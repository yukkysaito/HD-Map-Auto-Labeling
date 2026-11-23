"""Ground labels + ortho image of one scene: every camera keyframe of every drive that sees the grid is projected onto
the 3D map surface (map_surface.py).

Geometry from the LiDAR point cloud map, placement from localization (base_link snapped onto the map surface) x
official extrinsic + time offset (time_offset.py), classes from the per-image segmentation (common.GROUPS).
Several drives are fused (votes are summed); keyframes whose camera is farther than --rmax from the grid are skipped, and
cameras whose time offset could not be estimated (offset_status failed) are not used.
Per 0.1 m cell: class = argmax of the ground-class votes (0 = unobserved), ortho RGB = distance-weighted mean; with
--paint / --paint-crosswalk also cls_paint: the marking class following the paint seen in the ortho image, the gaps
between crosswalk stripes as road (hdlabel/paint.py; the training raster of export.py, while the vectors keep cls).
With --road-level road cells raised behind a kerb step become sidewalk or unobserved first (hdlabel/roadlevel.py).
--mode gather looks every visible cell up in the image instead of splatting pixels onto cells (no holes or moire at
range). With --max-shift, a further drive whose markings line up with the first drive's only after a shift (time offset
or pose error of a drive) is not fused. With --align, such a drive is instead moved onto the first drive (a rigid 2D
correction of its camera poses for this scene, measured on the marking cells, drive_align) and projected again; it is
fused when the correction is reliable and leaves no residual shift, else the --max-shift rule decides.
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
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import CLASSES, GROUPS, PALETTE, PoseInterp
from hdlabel.mapproj import Camera, Projector, loader, snap_to_ground
from hdlabel.paint import DEFAULTS as PAINT_DEFAULTS
from hdlabel.paint import XWALK_DEFAULTS


def project_drive(pj, data, calib, acc, a, stats, G=None):
    """every keyframe of every camera of one drive that can see the grid into acc. G: 4x4 map-frame correction applied
    to every camera pose of the drive (drive_align)."""
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
        if G is not None:
            T = G @ T
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


def marking_masks(ref, new, min_mass):
    """marking cells of two accumulators, on the cells both observed."""
    k = list(GROUPS).index("marking")

    def mask(acc):
        v = acc["votes"]
        m = v.sum(1)
        return m > min_mass, (m > min_mass) & (v.argmax(1) == k)

    (oa, ma), (ob, mb) = mask(ref), mask(new)
    both = oa & ob
    return ma & both, mb & both


def drive_shift(pj, ref, new, min_mass, radius=1.5, min_cells=500):
    """how well the marking cells of two accumulators line up: the 2D shift (within radius m) that maximises their
    overlap on the cells both observed. Returns shift_m (0 without enough marking cells), the overlap at zero and at
    the best shift."""
    ma, mb = marking_masks(ref, new, min_mass)
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


# drive_align: a correction is used when it moves the markings by at least min_move_m, its correlation peak is unique (no
# other local maximum farther than sep_m reaches max_ambiguity of it: dashes or parallel lines that fit at two shifts)
# and, after the drive is projected again, the residual shift is at most max_residual_m
ALIGN = dict(rot_deg=0.6, rot_step=0.15, min_move_m=0.1, sep_m=0.4, max_ambiguity=0.9, max_residual_m=0.15)


def rigid(dx, dy, rot_deg, pivot):
    """4x4 map-frame transform: rotation by rot_deg about the map point pivot (x, y), then a shift by (dx, dy)."""
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    G = np.eye(4)
    G[:2, :2] = [[c, -s], [s, c]]
    G[:2, 3] = np.asarray(pivot) - G[:2, :2] @ np.asarray(pivot) + [dx, dy]
    return G


def _peak(v):
    """sub-sample offset of the maximum of a parabola through three values around a maximum (-0.5 .. 0.5)."""
    d = v[0] - 2 * v[1] + v[2]
    return float(np.clip(0.5 * (v[0] - v[2]) / d, -0.5, 0.5)) if d < 0 else 0.0


def drive_align(pj, ref, new, min_mass, max_m=2.0, rot_deg=0.0, rot_step=0.15, min_cells=500, sep_m=0.4):
    """the rigid 2D correction (rotation within +-rot_deg about the centre of the reference marking cells, then a shift
    of at most max_m) that moves the marking cells of the new accumulator onto the reference's, on the cells both
    observed: FFT cross-correlation over every shift within max_m + 0.5 m for each rotation, the peak refined to a
    fraction of a cell (and of rot_step). Returns marking_cells and, with enough of them, dx_m / dy_m / rot_deg / pivot
    (map frame, for rigid()), shift_m, the overlap at zero and at the correction, ambiguity (the highest other local
    maximum of the overlap farther than sep_m from the peak / the peak), move_m (largest displacement within the
    marking region) and in_range (peak inside the search window and shift_m <= max_m)."""
    ma, mb = marking_masks(ref, new, min_mass)
    out = dict(marking_cells=[int(ma.sum()), int(mb.sum())])
    if min(out["marking_cells"]) < min_cells:
        return out
    A, B = pj.dense(ma), pj.dense(mb)
    rows, cols = np.nonzero(A | B)
    r0, c0 = rows.min(), cols.min()
    A = torch.from_numpy(A[r0 : rows.max() + 1, c0 : cols.max() + 1]).to(pj.dev).float()
    B = torch.from_numpy(B[r0 : rows.max() + 1, c0 : cols.max() + 1]).to(pj.dev).float()
    h, w = A.shape
    R = int(np.ceil((max_m + 0.5) / pj.res))
    N = (h + 2 * R, w + 2 * R)  # no wrap-around for shifts within R
    FA = torch.fft.rfft2(A, s=N)
    ii, jj = torch.nonzero(A, as_tuple=True)
    piv = torch.stack([ii.float().mean(), jj.float().mean()])  # (row, col) of the rotation centre
    gi, gj = torch.meshgrid(*(torch.arange(n, device=pj.dev, dtype=torch.float32) for n in (h, w)), indexing="ij")

    def score(th):
        """overlap of A with B rotated by th (deg) about piv and moved by (si, sj), for every si, sj in -R..R."""
        Bt = B
        if th:
            c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
            u, v = gi - piv[0], gj - piv[1]  # rotated B: output cell p takes B at R(-th) (p - piv) + piv
            si, sj = c * u + s * v + piv[0], -s * u + c * v + piv[1]
            grid = torch.stack([sj / max(w - 1, 1) * 2 - 1, si / max(h - 1, 1) * 2 - 1], -1)[None]
            Bt = torch.nn.functional.grid_sample(B[None, None], grid, align_corners=True)[0, 0]
        S = torch.fft.irfft2(FA * torch.fft.rfft2(Bt, s=N).conj(), s=N)  # S[s] = sum_x A(x) Bt(x - s)
        return torch.roll(S, (R, R), (0, 1))[: 2 * R + 1, : 2 * R + 1].cpu().numpy()

    ths = [float(t) for t in np.linspace(-rot_deg, rot_deg, 2 * round(rot_deg / rot_step) + 1)] if rot_deg else [0.0]
    maps = [score(t) for t in ths]
    best = np.array([m.max() for m in maps])
    t = int(best.argmax())
    S = maps[t]
    i, j = divmod(int(S.argmax()), 2 * R + 1)
    di = _peak(S[i - 1 : i + 2, j]) if 0 < i < 2 * R else 0.0
    dj = _peak(S[i, j - 1 : j + 2]) if 0 < j < 2 * R else 0.0
    th = ths[t] + (_peak(best[t - 1 : t + 2]) * (ths[1] - ths[0]) if 0 < t < len(ths) - 1 else 0.0)
    si, sj = (i - R + di) * pj.res, (j - R + dj) * pj.res
    far = np.hypot(*(np.mgrid[: 2 * R + 1, : 2 * R + 1] - np.array([i, j])[:, None, None])) * pj.res > sep_m
    other = far & (S >= maximum_filter(S, size=5, mode="constant", cval=np.inf))  # peaks, not the ridge of a line
    pivot = pj.origin[:2] + pj.lo + (np.array([r0, c0]) + piv.cpu().numpy() + 0.5) * pj.res
    G = rigid(si, sj, th, pivot)
    corners = (np.array([[r0, c0], [r0 + h, c0], [r0, c0 + w], [r0 + h, c0 + w]]) * pj.res) + pj.origin[:2] + pj.lo
    move = np.linalg.norm(corners @ G[:2, :2].T + G[:2, 3] - corners, axis=1).max()
    zero = maps[int(np.argmin(np.abs(ths)))][R, R]
    out.update(dx_m=round(si, 3), dy_m=round(sj, 3), rot_deg=round(th, 3), pivot=[round(float(v), 3) for v in pivot],
               shift_m=round(float(np.hypot(si, sj)), 3), move_m=round(float(move), 3), overlap_zero=round(float(zero)),
               overlap_best=round(float(best[t])), ambiguity=round(float(S[other].max(initial=0) / max(best[t], 1e-9)), 3),
               in_range=bool(0 < i < 2 * R and 0 < j < 2 * R and np.hypot(si, sj) <= max_m))  # fmt: skip
    return out


def align_drive(pj, ref, new, data, calib, a, stats, chk):
    """the drive projected again with its drive_align correction when that is reliable (chk["aligned"]), else new
    unchanged. The correction and its checks go to chk["align"]."""
    al = chk["align"] = drive_align(pj, ref, new, a.min_mass, a.align, ALIGN["rot_deg"], ALIGN["rot_step"],
                                    sep_m=ALIGN["sep_m"])  # fmt: skip
    if "dx_m" not in al:
        al["used"] = "too few marking cells"
    elif not al["in_range"]:
        al["used"] = "out of range"
    elif al["ambiguity"] > ALIGN["max_ambiguity"]:
        al["used"] = "ambiguous"
    elif al["move_m"] < ALIGN["min_move_m"]:
        al["used"] = "not needed"
    else:
        moved = pj.new_acc(rgb=True)
        project_drive(pj, data, calib, moved, a, stats, G=rigid(al["dx_m"], al["dy_m"], al["rot_deg"], al["pivot"]))
        res = drive_align(pj, ref, moved, a.min_mass, a.align, sep_m=ALIGN["sep_m"])
        al.update(residual_m=res.get("shift_m"), overlap_aligned=res.get("overlap_zero"))
        if res.get("shift_m") is None or res["shift_m"] > ALIGN["max_residual_m"]:
            al["used"] = "residual too large"
        else:
            al["used"] = "aligned"
            chk["aligned"] = True
            name = os.path.basename(data.rstrip("/"))
            print(f"{name}: aligned by {al['shift_m']} m, {al['rot_deg']} deg", flush=True)
            return moved
    return new


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
        nargs=len(PAINT_DEFAULTS),
        type=float,
        metavar=tuple(k.upper() for k in PAINT_DEFAULTS),
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
        "--road-level",
        nargs=9,
        type=float,
        metavar="V",
        help="road cells raised behind a kerb step become sidewalk / unobserved; the values of hdlabel/roadlevel.py"
        " DEFAULTS in their order (grade, step_m, max_step_m, max_width_m, near_m, min_share, min_cells, keep_m, seed_dz)",
    )
    p.add_argument(
        "--max-shift",
        type=float,
        default=0.0,
        help="a further drive is not fused when its markings fit the first drive's best shifted by more than this (m;"
        " 0: fuse every drive). Time offset or pose errors of a drive would otherwise put ghost lines in the map",
    )
    p.add_argument(
        "--align",
        type=float,
        default=0.0,
        metavar="MAX_M",
        help="move a further drive onto the first drive's markings (rigid correction of its poses for this scene, at"
        " most MAX_M m) and project it again before fusing (0: off; drive_align)",
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
        if a.align:
            new = align_drive(pj, acc, new, data, calib, a, stats, chk)
        if a.max_shift and not chk.get("aligned") and chk.get("shift_m", 0) > a.max_shift:  # would ghost
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
    if a.road_level:
        from hdlabel.roadlevel import DEFAULTS as RL
        from hdlabel.roadlevel import refine_scene

        cls, stats["road_level"] = refine_scene(
            cls, a.ground, [d for d, _ in a.drive], pj.origin, pj.lo, pj.res, **dict(zip(RL, a.road_level))
        )
    extra = {}
    if a.paint:  # the vectors keep using cls (stroke widths of the segmentation), the training raster cls_paint
        from hdlabel.paint import refine

        extra["cls_paint"], stats["paint"] = refine(cls, rgb, pj.res, **dict(zip(PAINT_DEFAULTS, a.paint)))
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
