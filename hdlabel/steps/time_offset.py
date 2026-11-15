"""Camera calibration of one drive: official extrinsics (/tf_static) + image time offsets estimated on the map surface.

The mounting (R, t of each camera frame in base_link) is taken as-is from DATA/tf_static.json (or --tf, the file of
another drive of the same vehicle when the bag has none); camera frames come from camera_info. Only the image time
offset (image stamp + offset = pose time) is estimated, from multi-view consistency on the LiDAR map surface
(map_surface.py, built for the scenes with the most turning: --window, repeatable).
A wrong offset rotates / shifts the projection during turns and accelerations; frames with yaw rate > --turn deg/s are
used. Coarse scan (default +-1.5 s: whole-second stamp errors occur, e.g. one drive had a camera +1.0 s late), then a
fine scan. Cameras whose own curve is flat (downward / side views see little rotation) are aligned against the labels
of the accepted cameras over all frames (along-track errors become visible); still flat -> median of the accepted.
Score (lower = sharper): marking blur B = sum m p (1 - p) / sum m p over 0.1 m cells, p = marking share
(marking + crosswalk vs. road), m = vote mass.
--dz-check also scans a camera height residual dz (diagnostic only: applying per-camera or vehicle dz lowered the
cross-camera agreement and the drive-to-drive repeatability, so the extrinsics stay official).
--fixed S skips the estimation (every camera S seconds, status "fixed").
Writes CALIB/cam<N>.json (common.Extrinsic + frame, offset method / acceptance), CALIB/time_offset.json|png.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import GROUPS, Extrinsic, PoseInterp
from hdlabel.mapproj import Camera, Projector, loader, snap_to_ground


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--window",
        nargs=3,
        action="append",
        default=[],
        metavar=("GROUND", "T0", "T1"),
        help="map_surface.py map_ground.npz and the stamp range (ns) of the frames on it; repeat for more scenes",
    )
    p.add_argument("--data", required=True)
    p.add_argument("--calib", required=True, help="output dir for cam<N>.json")
    p.add_argument("--tf", help="tf_static.json to use instead of DATA's")
    p.add_argument("--dz", type=float, nargs=3, default=(-0.12, 0.12, 0.02), help="scan start stop step (m)")
    p.add_argument(
        "--coarse", type=float, nargs=3, default=(-1.5, 1.5, 0.1), help="coarse offset scan start stop step (s)"
    )
    p.add_argument(
        "--fine",
        type=float,
        nargs=2,
        default=(0.1, 0.01),
        help="fine scan half width and step around the coarse best (s)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=200,
        help="max frames per camera (turning frames; spread frames for the reference / dz); 400 gave the same offsets (within 4 ms) in twice the time",
    )
    p.add_argument("--turn", type=float, default=4.0, help="min yaw rate (deg/s) of frames for the offset scan")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--rmax", type=float, default=25.0)
    p.add_argument(
        "--snap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="base_link on the map surface (mapproj.snap_to_ground); dz is then a residual",
    )
    p.add_argument(
        "--min-rise", type=float, default=0.01, help="min relative curve rise over +-3 fine steps to accept an offset"
    )
    p.add_argument(
        "--min-coarse-rise",
        type=float,
        default=0.03,
        help="min relative coarse-curve rise over +-0.5 s: the offset is then "
        "taken from the fine-scan minimum even when the fine curve is flat (short drives; uncertain to a few 10 ms)",
    )
    p.add_argument("--dz-check", action="store_true")
    p.add_argument("--fixed", type=float, help="no estimation: every camera gets this offset (s), status 'fixed' "
                   "(hardware-synchronised cameras, synthetic data)")  # fmt: skip
    a = p.parse_args(argv)
    if a.fixed is None and not a.window:
        p.error("--window is required unless --fixed is given")
    os.makedirs(a.calib, exist_ok=True)
    out = f"{a.calib}/time_offset"
    if os.path.exists(f"{out}.json"):
        os.remove(f"{out}.json")  # marks the calibration as incomplete until the end
    tmp = f"{a.calib}/.official"
    os.makedirs(tmp, exist_ok=True)  # official extrinsics (offset 0) while estimating

    # official extrinsics (offset 0) for every camera with keyframes + segmentation
    tf = json.load(open(a.tf or f"{a.data}/tf_static.json"))
    info = json.load(open(f"{a.data}/camera_info.json"))
    cam_ids = sorted(
        int(c) for c in info if os.path.isdir(f"{a.data}/seg_cam{c}") and os.listdir(f"{a.data}/seg_cam{c}")
    )
    for c in cam_ids:
        T = np.array(tf[info[str(c)]["frame"]])
        Extrinsic(T[:3, :3], T[:3, 3], 0.0).save(
            f"{tmp}/cam{c}.json",
            frame=info[str(c)]["frame"],
            tf_static=os.path.abspath(a.tf or f"{a.data}/tf_static.json"),
        )

    if a.fixed is not None:  # no estimation
        for c in cam_ids:
            e = json.load(open(f"{tmp}/cam{c}.json"))
            e.update(time_offset_s=a.fixed, offset_status="fixed", offset_method="fixed (--fixed)")
            json.dump(e, open(f"{a.calib}/cam{c}.json", "w"), indent=1)
        res = dict(final={f"cam{c}": a.fixed for c in cam_ids}, status={f"cam{c}": "fixed" for c in cam_ids})
        json.dump(res, open(f"{out}.json", "w"), indent=1)
        print(f"time offsets fixed at {a.fixed} s for cameras {cam_ids}")
        return

    grounds = [g for g, _, _ in a.window]
    wins = np.array([[int(t0), int(t1)] for _, t0, t1 in a.window])
    pjs = [Projector(g) for g in grounds]
    poser = PoseInterp(f"{a.data}/gt.npy")
    snap = snap_to_ground(poser, grounds) if a.snap else dict(snapped=False)
    print("ground snap:", snap)
    G = list(GROUPS)
    MK = [G.index("marking"), G.index("crosswalk")]
    RD = G.index("road")
    classes = torch.tensor([RD] + MK, device=pjs[0].dev)

    def blur(accs, refs=None):
        """marking blur over all windows (+ refs: reference votes of other cameras, evaluated on the cells acc touches)."""
        num = den = 0.0
        for w, acc in enumerate(accs):
            v = acc["votes"]
            touched = v.sum(1) > 0
            if refs is not None:
                v = v + refs[w]
            v = v[touched]
            mk = v[:, MK].sum(1)
            m = mk + v[:, RD]
            ok = m > 0
            pm = mk[ok] / m[ok]
            num += float((m[ok] * pm * (1 - pm)).sum())
            den += float((m[ok] * pm).sum())
        return num / max(den, 1e-9)

    def project(cam, idx, imgs, dz=0.0, offset=None):
        """frames idx of cam onto the surface of their window -> one accumulator per window."""
        accs = [pj.new_acc() for pj in pjs]
        T = cam.T_map_cam(idx, offset)
        for k, i in enumerate(idx):
            w = cam.win[i]
            pjs[w].project(T[k], cam.K, imgs[k], None, accs[w], a.stride, a.rmax, dz=dz, classes=classes)
        return accs

    def score(cam, idx, imgs, dz=0.0, offset=None, ref=None):
        accs = project(cam, idx, imgs, dz, offset)
        return blur(accs, ref), float(sum(acc["votes"].sum() for acc in accs))

    def yaw_rate(cam):
        T = cam.poser(cam.st.astype(np.float64))
        yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
        return np.degrees(np.gradient(yaw, cam.st * 1e-9))

    def scan(cam, idx, imgs, ref=None):
        """coarse (half of the frames) then fine offset scan; returns result dict."""
        sub = slice(None, None, 2)
        bc = np.array([score(cam, idx[sub], imgs[sub], 0.0, o, ref)[0] for o in coarse])
        oc = float(coarse[bc.argmin()])
        offs = oc + fine
        bo = np.array([score(cam, idx, imgs, 0.0, o, ref)[0] for o in offs])
        j = int(bo.argmin())
        # sub-step refinement (parabola) and a flatness check: curve rise over +-3 steps relative to the minimum
        o_best = (
            offs[j]
            if j in (0, len(offs) - 1)
            else offs[j] + a.fine[1] * 0.5 * (bo[j - 1] - bo[j + 1]) / (bo[j - 1] - 2 * bo[j] + bo[j + 1])
        )
        rise = float((bo[max(0, j - 3)] + bo[min(len(bo) - 1, j + 3)]) / 2 / bo[j] - 1)
        k = int(bc.argmin())
        n5 = int(round(0.5 / a.coarse[2]))  # coarse sharpness over +-0.5 s
        coarse_rise = float((bc[max(0, k - n5)] + bc[min(len(bc) - 1, k + n5)]) / 2 / bc[k] - 1)
        return dict(
            best=round(float(o_best), 4),
            rise=round(rise, 4),
            coarse_rise=round(coarse_rise, 4),
            frames=len(idx),
            accepted=bool(rise > a.min_rise and 0 < j < len(offs) - 1),
            coarse_ok=bool(coarse_rise > a.min_coarse_rise and 0 < k < len(bc) - 1),
            coarse_best=round(oc, 2),
            coarse_curve=np.round(bc, 5).tolist(),
            offsets=np.round(offs, 4).tolist(),
            curve=np.round(bo, 5).tolist(),
        )

    res = {"snap": snap, "dz": {}, "offset": {}}
    dzs = np.arange(a.dz[0], a.dz[1] + 1e-9, a.dz[2])
    coarse = np.arange(a.coarse[0], a.coarse[1] + 1e-9, a.coarse[2])
    fine = np.arange(-a.fine[0], a.fine[0] + 1e-9, a.fine[1])
    fig, ax = plt.subplots(1, 3 if a.dz_check else 2, figsize=(17 if a.dz_check else 12, 4.5))
    cams = {c: Camera(a.data, tmp, c, poser) for c in cam_ids}
    for cam in cams.values():  # keep the frames inside the windows
        w = np.full(len(cam.st), -1)
        for j, (t0, t1) in enumerate(wins):
            w[(cam.st >= t0) & (cam.st <= t1)] = j
        k = w >= 0
        cam.st, cam.win = cam.st[k], w[k]
        cam.files = [f for f, kk in zip(cam.files, k) if kk]
        cam.seg = [f for f, kk in zip(cam.seg, k) if kk]
    spread = {}
    for c, cam in cams.items():  # frames spread over the drive (dz scan, reference)
        idx = np.linspace(0, len(cam.st) - 1, min(a.frames, len(cam.st))).astype(int)
        spread[c] = (idx, [s for s, _ in loader([cam.seg[i] for i in idx])])

    # pass 1: each camera alone, turning frames (rotation errors are visible without any reference)
    for c, cam in cams.items():
        turn = np.flatnonzero(np.abs(yaw_rate(cam)) > a.turn)
        if len(turn) < 20:
            print(f"cam{c}: too few turning frames")
            continue
        ti = turn[np.linspace(0, len(turn) - 1, min(len(turn), a.frames)).astype(int)]
        r = scan(cam, ti, [s for s, _ in loader([cam.seg[i] for i in ti])])
        r.update(method="self (turning frames)", calib=cam.ext.offset)
        res["offset"][f"cam{c}"] = r
        ax[0].plot(coarse, np.array(r["coarse_curve"]) / min(r["coarse_curve"]), label=f"cam{c}")
        ax[1].plot(np.array(r["offsets"]) * 1000, np.array(r["curve"]) / min(r["curve"]), label=f"cam{c}")
        print(
            f"cam{c}: self offset {r['best']:+.3f} s (coarse {r['coarse_best']:+.1f}, rise {r['rise']:.3f}{'' if r['accepted'] else ', not accepted'})",
            flush=True,
        )

    # pass 2: cameras with a flat self curve (e.g. downward views) against the accepted cameras' labels, all frames
    # (along-track time errors shift the projection against the reference even when driving straight)
    good = [c for c in cams if res["offset"].get(f"cam{c}", {}).get("accepted")]
    weak = [c for c in cams if c not in good]
    if good and weak:
        ref = [
            sum(x)
            for x in zip(
                *[
                    [acc["votes"] for acc in project(cams[c], *spread[c], 0.0, res["offset"][f"cam{c}"]["best"])]
                    for c in good
                ]
            )
        ]
        for c in weak:
            r = scan(cams[c], *spread[c], ref=ref)
            r.update(method=f"against cams {good}", calib=cams[c].ext.offset)
            res["offset"][f"cam{c}"] = r
            ax[1].plot(
                np.array(r["offsets"]) * 1000, np.array(r["curve"]) / min(r["curve"]), "--", label=f"cam{c} (ref)"
            )
            print(
                f"cam{c}: offset vs reference {r['best']:+.3f} s (coarse {r['coarse_best']:+.1f}, rise {r['rise']:.3f}"
                f"{'' if r['accepted'] else ', not accepted'})",
                flush=True,
            )
        del ref
    acc_off = [v["best"] for v in res["offset"].values() if v["accepted"]]
    res["offset_fallback"] = round(float(np.median(acc_off)), 4) if acc_off else None
    # final offset and its status per camera: fine (sharp own / reference curve) > coarse (only the coarse curve is sharp) >
    # fallback (median of the fine ones) > failed (offset 0, the drive's samples fail QA in export.py)
    final, status = {}, {}
    for c in cams:
        r = res["offset"].get(f"cam{c}", {})
        if r.get("accepted"):
            final[c], status[c] = r["best"], "reference" if r["method"].startswith("against") else "fine"
        elif r.get("coarse_ok"):
            final[c], status[c] = r["best"], "coarse"
        elif res["offset_fallback"] is not None:
            final[c], status[c] = res["offset_fallback"], "fallback"
        else:
            final[c], status[c] = 0.0, "failed"
        print(f"cam{c}: final offset {final[c]:+.3f} s ({status[c]})")

    # pass 3 (diagnostic): camera height residual with the final offsets
    for c, cam in cams.items() if a.dz_check else []:
        sc = np.array([score(cam, *spread[c], dz, final[c]) for dz in dzs])
        b = sc[:, 0]
        best = float(dzs[b.argmin()])
        res["dz"][f"cam{c}"] = dict(best=round(best, 3), mass=sc[0, 1], curve=np.round(b, 5).tolist())
        ax[-1].plot(dzs * 100, b / b.min(), label=f"cam{c}")
        print(f"cam{c}: final offset {final[c]}, dz {best:+.2f} m", flush=True)

    if res["dz"]:
        dd = res["dz"]
        w = np.array([v["mass"] for v in dd.values()])
        b = np.array([v["best"] for v in dd.values()])
        o = np.argsort(b)
        res["dz_vehicle"] = round(float(b[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2)]), 3)
    print(
        f"offsets accepted for {[c for c in cams if res['offset'].get(f'cam{c}', {}).get('accepted')]}; fallback {res['offset_fallback']}"
    )
    for x, t, u in list(
        zip(
            ax,
            (
                "coarse: blur vs time offset (turning frames)",
                "fine: blur vs time offset",
                "blur vs camera height correction dz",
            ),
            ("offset (s)", "offset (ms)", "dz (cm)"),
        )
    ):
        x.set_title(t)
        x.set_xlabel(u)
        x.set_ylabel("blur / min")
        x.legend(fontsize=7)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=110)
    res["final"] = {f"cam{c}": final[c] for c in cams}
    res["status"] = {f"cam{c}": status[c] for c in cams}
    res["windows"] = a.window
    for c in cams:  # calibration files first, time_offset.json last
        e = json.load(open(f"{tmp}/cam{c}.json"))
        r = res["offset"].get(f"cam{c}")
        e.update(
            time_offset_s=final[c],
            offset_status=status[c],
            offset_method=f"map surface marking sharpness ({r['method'] if r else 'no turning frames'})",
        )
        json.dump(e, open(f"{a.calib}/cam{c}.json", "w"), indent=1)
    json.dump(res, open(f"{out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
