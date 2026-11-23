"""Ground labels + ortho image of one scene: every camera keyframe of every drive that sees the grid is projected onto
the 3D map surface (map_surface.py).

Geometry from the LiDAR point cloud map, placement from localization (base_link snapped onto the map surface) x
official extrinsic + time offset (time_offset.py), classes from the per-image segmentation (common.GROUPS).
Several drives are fused (votes are summed); keyframes whose camera is farther than --rmax from the grid are skipped, and
cameras whose time offset could not be estimated (offset_status failed) are not used.
Per 0.1 m cell: class = argmax of the ground-class votes (0 = unobserved), ortho RGB = distance-weighted mean; with
--paint / --paint-crosswalk also cls_paint: the marking class following the paint seen in the ortho image, the gaps
between crosswalk stripes as road (hdlabel/paint.py; the training raster of export.py, while the vectors keep cls).
--mode gather looks every visible cell up in the image instead of splatting pixels onto cells (no holes or moire at
range). With --max-shift, a further drive whose markings line up with the first drive's only after a shift (time offset
or pose error of a drive) is not fused.
Writes OUT/ground_map.npz (cls 0.1 m raster: 0 unobserved, 1.. common.CLASSES; height / height_w surface grid; origin,
lo, res, surf_res; compact votes / n / cells), OUT/ground_map.json, OUT/stats.json; with --debug DIR:
DIR/ground_map.png (classes), DIR/ortho.jpg (also used by the review sheets)
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import CLASSES, GROUPS, PALETTE, PoseInterp
from hdlabel.mapproj import Camera, Projector, loader, snap_to_ground
from hdlabel.paint import XWALK_DEFAULTS


def project_drive(pj, data, calib, acc, a, stats):
    """every keyframe of every camera of one drive that can see the grid into acc."""
    poser = PoseInterp(f"{data}/gt.npy")
    name = os.path.basename(data.rstrip("/"))
    if a.snap:
        stats[f"{name}/snap"] = snap_to_ground(poser, a.ground)
    lo_m, hi_m = pj.lo - a.rmax, pj.lo + np.array(pj.Lshape) * pj.res + a.rmax  # camera positions that see the grid
    for f in sorted(glob.glob(f"{calib}/cam*.json")):
        c = int(os.path.basename(f)[3:-5])
        if not os.path.isdir(f"{data}/seg_cam{c}"):
            continue
        if json.load(open(f)).get("offset_status") == "failed":  # unknown time offset: would smear wrong labels
            stats[f"{name}/cam{c}"] = "skipped: time offset failed"
            print(f"{name} cam{c}: skipped (time offset failed)")
            continue
        cam = Camera(data, calib, c, poser)
        T = cam.T_map_cam(np.arange(len(cam.st)))
        xy = T[:, :2, 3] - pj.origin[:2]
        idx = np.flatnonzero(np.all((xy > lo_m) & (xy < hi_m), 1) & cam.on_level)[:: a.every]
        if not len(idx):
            continue
        t0 = time.time()
        n = 0
        for k, (seg, img) in zip(idx, loader([cam.seg[i] for i in idx], [cam.files[i] for i in idx])):
            if a.mode == "gather":
                n += pj.gather(T[k], cam.K, seg, img, acc, a.rmax, a.near, power=a.power, rgb_power=a.rgb_power)
            else:
                n += pj.project(T[k], cam.K, seg, img, acc, a.stride, a.rmax, a.near)
        stats[f"{name}/cam{c}"] = dict(images=len(idx), samples_M=round(n / 1e6, 1))
        print(f"{name} cam{c}: {len(idx)} images, {n / 1e6:.0f} M ground samples, {time.time() - t0:.0f} s", flush=True)


def drive_shift(pj, ref, new, min_mass, radius=1.5, min_cells=500):
    """how well the marking cells of two accumulators line up: the 2D shift (within radius m) that maximises their
    overlap on the cells both observed. Returns shift_m (0 without enough marking cells), the overlap at zero and at
    the best shift."""
    k = list(GROUPS).index("marking")

    def mask(acc):
        v = acc["votes"]
        m = v.sum(1)
        return m > min_mass, (m > min_mass) & (v.argmax(1) == k)

    (oa, ma), (ob, mb) = mask(ref), mask(new)
    both = oa & ob
    ma, mb = ma & both, mb & both
    if int(ma.sum()) < min_cells or int(mb.sum()) < min_cells:
        return dict(marking_cells=[int(ma.sum()), int(mb.sum())])
    A = torch.from_numpy(pj.dense(ma)).to(pj.dev).float()
    B = torch.from_numpy(pj.dense(mb)).to(pj.dev).float()
    rows, cols = torch.nonzero(A + B, as_tuple=True)
    R = int(round(radius / pj.res))
    A = A[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    B = torch.nn.functional.pad(B[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1], (R, R, R, R))
    h, w = A.shape
    score = torch.stack([(A * B[R + dx : R + dx + h, R + dy : R + dy + w]).sum()
                         for dx in range(-R, R + 1) for dy in range(-R, R + 1)]).view(2 * R + 1, 2 * R + 1)  # fmt: skip
    i = int(score.argmax())
    dx, dy = i // (2 * R + 1) - R, i % (2 * R + 1) - R
    return dict(marking_cells=[int(ma.sum()), int(mb.sum())], overlap_zero=int(score[R, R]),
                overlap_best=int(score.max()), shift_m=round(float(np.hypot(dx, dy)) * pj.res, 2))  # fmt: skip


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ground", required=True, help="map_surface.py map_ground.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--debug", help="dir for ground_map.png / ortho.jpg")
    p.add_argument(
        "--drive",
        nargs=2,
        action="append",
        required=True,
        metavar=("DATA", "CALIB"),
        help="data dir and calib dir of one drive (repeat to fuse drives)",
    )
    p.add_argument(
        "--snap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="base_link on the map surface (mapproj.snap_to_ground)",
    )
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--rmax", type=float, default=30.0)
    p.add_argument("--near", type=float, default=8.0)
    p.add_argument("--min-mass", type=float, default=0.05)
    p.add_argument(
        "--mode",
        choices=["splat", "gather"],
        default="splat",
        help="splat: every ground pixel votes for the cell its ray hits; gather: every visible cell looks up its pixel"
        " (no holes / moire at range, mapproj.Projector.gather)",
    )
    p.add_argument(
        "--power", type=float, default=2.0, help="gather: distance weight exponent (mapproj.Projector.gather)"
    )
    p.add_argument("--rgb-power", type=float, help="gather: ortho colour weight exponent (mapproj.Projector.gather)")
    p.add_argument(
        "--paint",
        nargs=5,
        type=float,
        metavar=("CONTRAST", "WINDOW_M", "GROW_M", "MIN_SHARE", "SHARE_M"),
        help="also write cls_paint: the marking class following the paint in the ortho image (hdlabel/paint.py)",
    )
    p.add_argument(
        "--paint-crosswalk",
        nargs=len(XWALK_DEFAULTS),
        type=float,
        metavar=tuple(k.upper() for k in XWALK_DEFAULTS),
        help="the unpainted gaps between crosswalk stripes become road (hdlabel/paint.py refine_crosswalk)",
    )
    p.add_argument(
        "--max-shift",
        type=float,
        default=0.0,
        help="a further drive is not fused when its markings fit the first drive's best shifted by more than this (m;"
        " 0: fuse every drive). Time offset or pose errors of a drive would otherwise put ghost lines in the map",
    )
    p.add_argument("--every", type=int, default=1, help="use every n-th keyframe (quick tests)")
    a = p.parse_args(argv)

    pj = Projector(a.ground)
    acc = None
    stats = {}
    for data, calib in a.drive:
        name = os.path.basename(data.rstrip("/"))
        new = pj.new_acc(rgb=True)
        project_drive(pj, data, calib, new, a, stats)
        if acc is None:  # the scene's own drive: the reference
            acc = new
            continue
        chk = stats[f"{name}/fusion"] = drive_shift(pj, acc, new, a.min_mass)
        if a.max_shift and chk.get("shift_m", 0) > a.max_shift:  # labels of this drive sit elsewhere: would ghost
            chk["fused"] = False
            print(f"{name}: not fused, its markings are {chk['shift_m']} m off the reference drive's", flush=True)
            continue
        chk["fused"] = True
        for k in acc:
            acc[k] += new[k]
        del new

    votes = acc["votes"]
    mass = votes.sum(1)
    cls_c = torch.where(mass > a.min_mass, votes.argmax(1) + 1, torch.zeros_like(mass, dtype=torch.long)).to(
        torch.uint8
    )
    cls = pj.dense(cls_c)
    rgb = pj.dense((acc["rgb"] / acc["w"].clamp(min=1e-9)[:, None]).clamp(0, 255).to(torch.uint8))
    extra = {}
    if a.paint:  # the vectors keep using cls (stroke widths of the segmentation), the training raster cls_paint
        from hdlabel.paint import DEFAULTS, refine

        extra["cls_paint"], stats["paint"] = refine(cls, rgb, pj.res, **dict(zip(DEFAULTS, a.paint)))
    if a.paint_crosswalk:
        from hdlabel.paint import refine_crosswalk

        extra["cls_paint"], stats["paint_crosswalk"] = refine_crosswalk(
            extra.get("cls_paint", cls), rgb, pj.res, **dict(zip(XWALK_DEFAULTS, a.paint_crosswalk))
        )
    groups = CLASSES[1:]
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(
        f"{a.out}/ground_map.npz",
        cls=cls,
        height=pj.H.astype(np.float32),
        height_w=np.load(a.ground)["height_w"],
        groups=np.array(groups),
        origin=pj.origin,
        lo=pj.lo,
        res=pj.res,
        surf_res=pj.sres,
        cells=pj.cells,
        votes=votes.cpu().numpy().astype(np.float16),
        n=acc["n"].cpu().numpy().astype(np.float32),
        **extra,
        snap=bool(a.snap),
    )
    obs = cls > 0
    stats["observed_km2"] = round(float(obs.sum()) * pj.res**2 / 1e6, 4)
    stats["class_share"] = {g: round(float((cls == j + 1).sum() / max(obs.sum(), 1)), 4) for j, g in enumerate(groups)}
    json.dump(
        dict(
            groups=["unobserved"] + groups,
            origin=pj.origin.tolist(),
            lo=pj.lo.tolist(),
            res=pj.res,
            surf_res=pj.sres,
            shape=list(cls.shape),
            drives=[d for d, _ in a.drive],
            note="cls[i, j] covers map x = origin.x + lo.x + i*res, y = origin.y + lo.y + j*res",
        ),
        open(f"{a.out}/ground_map.json", "w"),
        indent=1,
    )
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
        cv2.imwrite(f"{a.debug}/ground_map.png", np.rot90(PALETTE[cls])[..., ::-1])
        cv2.imwrite(f"{a.debug}/ortho.jpg", np.rot90(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps(stats["class_share"]), f"observed {stats['observed_km2']} km2")


if __name__ == "__main__":
    main()
