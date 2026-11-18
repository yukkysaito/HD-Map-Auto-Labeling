"""Road-level check of marking elements (vectorize.py --marking-gate): paint lies on the road surface the vehicle drove on.

False markings from the segmentation lie on something else: a ledge or rail base above the road, a raised sidewalk with
tactile blocks or a brick plaza behind a kerb step. Per element (lane_line, stop_line, marking_area, arrow outline) its
0.1 m cells are compared with the LiDAR ground height Hc (nan = no LiDAR ground there):
  reference road  road / manhole cells with LiDAR ground (the local plane); for the step profiles only those of the
                  drivable regions reachable from the trajectories (curbcheck.reachable; without a trajectory: all)
  local plane     per 4 m block of the element, a robust plane fitted to the reference road cells within ref_m of the
                  block's cells (slopes and camber are followed); residual = cell height - plane
  known           cells with LiDAR ground, not within exempt_m of crosswalk cells (raised crosswalks / humps are paint
                  above the road); cells without LiDAR are unknown and never count against an element
Rules (an element with fewer than min_known of its cells known, or fewer than min_cells, is not judged):
  off_level  less than min_on_level of the known cells within tol m of the local road plane and the median residual
             beyond tol (a consistent offset, not scatter)
  raised     median residual above step_m (or below -step_m): the element sits on a ledge / behind a step or in a gap
  cut_off    less than min_on_level of the known cells on the road level and, on more than half of the profiles from
             the nearest trajectory-reachable road cell (within cut_m) to element cells, the ground rises or falls by
             more than step_m (the step test of geom_check: 7 cm). Paint on the level of the road beyond a median
             (the profile crosses its kerbs) is not cut off
An element without reference road within ref_m (no plane) or reachable road within cut_m is not judged by that rule:
roads across a median, side streets and car parks are not reachable from the trajectories but carry real paint.
vectorize.py turns a failing element into kind marking_candidate (props.candidate_of = its kind, props.marking_check =
the failed rules).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from hdlabel.curbcheck import reachable, resample

DEFAULTS = dict(tol=0.04, min_on_level=0.7, step_m=0.07, ref_m=3.0, block_m=4.0, min_known=0.3, min_cells=5,
                min_ref=20, exempt_m=1.0, min_area=50.0, profiles=30, cut_m=5.0)  # fmt: skip
RULES = ("off_level", "raised", "cut_off")


def upsample(H: np.ndarray, k: int, shape) -> np.ndarray:
    """surface grid (surf_res = k * res) -> label grid of `shape` (nearest; outside: nan)."""
    U = np.repeat(np.repeat(H, k, 0), k, 1)[: shape[0], : shape[1]]
    return np.pad(U, [(0, shape[0] - U.shape[0]), (0, shape[1] - U.shape[1])], constant_values=np.nan)


def reference_road(cls, C, Hc, lo, res, trajs=(), min_area=50.0):
    """-> (road / manhole cells with LiDAR ground, those of the drivable regions reachable from the trajectories)."""
    known = np.isfinite(Hc)
    road = np.isin(cls, [C["road"], C["manhole"]]) & known
    trajs = [np.asarray(t, float) for t in trajs if len(t) >= 2]
    if not trajs:
        return road, road
    surf = cv2.dilate(known.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    seeds = np.floor((np.concatenate([resample(t, res) for t in trajs]) - lo) / res).astype(int)
    reach = reachable(cls, C, seeds, res, surf, 0.0, min_area)
    return road, (road & reach if reach.any() else road)


def fit_plane(ij: np.ndarray, z: np.ndarray, c: np.ndarray):
    """robust plane z = a + b (i - ci) + d (j - cj) (cells) -> coefficients; one refit without residuals > 5 cm
    over the median (cars / vegetation left in the ground)."""
    A = np.c_[np.ones(len(z)), ij - c]
    keep = np.ones(len(z), bool)
    for _ in range(2):
        if keep.sum() < 3:
            break
        w, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ w
        keep = np.abs(r - np.median(r[keep])) < 0.05
    return w


def check_markings(elems, cls, C, lo, res, Hc, trajs=(), params=None):
    """elems: list of (N, 2) int cell arrays (row, col of the 0.1 m label grid) per element; Hc: LiDAR ground height on
    that grid (nan = none). -> (failed rules per element (empty: on the road), stats per element, summary)."""
    p = dict(DEFAULTS, **(params or {}))
    ref, rref = reference_road(cls, C, Hc, lo, res, trajs, p["min_area"])
    known = np.isfinite(Hc)
    if p["exempt_m"] > 0 and (cls == C["crosswalk"]).any():
        known &= distance_transform_edt(cls != C["crosswalk"]) * res > p["exempt_m"]
    d_ref, near = distance_transform_edt(~rref, return_indices=True) if rref.any() else (None, None)
    R = int(np.ceil(p["ref_m"] / res))
    blk = max(1, int(round(p["block_m"] / res)))
    shape = np.array(cls.shape)
    whys, stats = [], []
    for ij in elems:
        ij = np.asarray(ij, int).reshape(-1, 2)
        ij = ij[np.all((ij >= 0) & (ij < shape), 1)]
        kn = known[ij[:, 0], ij[:, 1]] if len(ij) else np.zeros(0, bool)
        st = dict(cells=int(len(ij)), known=round(float(kn.mean()), 3) if len(ij) else 0.0)
        if len(ij) < p["min_cells"] or kn.sum() < p["min_cells"] or kn.mean() < p["min_known"]:
            whys.append([])
            stats.append(st)
            continue
        q = ij[kn]
        z = Hc[q[:, 0], q[:, 1]]
        resid = np.full(len(q), np.nan)
        for b in np.unique(q // blk, axis=0):  # local plane per block of the element
            inb = np.all(q // blk == b, 1)
            w0, w1 = np.maximum(q[inb].min(0) - R, 0), np.minimum(q[inb].max(0) + R + 1, shape)
            win = np.zeros(w1 - w0, np.uint8)
            win[q[inb, 0] - w0[0], q[inb, 1] - w0[1]] = 1
            close = cv2.dilate(win, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))) > 0
            rr = np.argwhere(close & rref[w0[0] : w1[0], w0[1] : w1[1]])  # the driven road first
            if len(rr) < p["min_ref"]:
                rr = np.argwhere(close & ref[w0[0] : w1[0], w0[1] : w1[1]])
            if len(rr) < p["min_ref"]:
                continue
            c = q[inb].mean(0)
            w = fit_plane(rr + w0, Hc[rr[:, 0] + w0[0], rr[:, 1] + w0[1]], c)
            resid[inb] = z[inb] - (w[0] + (q[inb] - c) @ w[1:])
        why = []
        fit = np.isfinite(resid)
        st["plane"] = round(float(fit.mean()), 3)
        if fit.mean() >= 0.5:
            r = resid[fit]
            on = float((np.abs(r) < p["tol"]).mean())
            med = float(np.median(r))
            st.update(on_level=round(on, 3), median_cm=round(100 * med, 1))
            if on < p["min_on_level"] and abs(med) > p["tol"]:
                why.append("off_level")
            if abs(med) > p["step_m"]:
                why.append("raised")
        if d_ref is not None and st.get("on_level", 1.0) < p["min_on_level"]:  # step to the driven road
            sel = q[np.linspace(0, len(q) - 1, min(len(q), p["profiles"])).astype(int)]
            sel = sel[d_ref[sel[:, 0], sel[:, 1]] * res <= p["cut_m"]]
            cut = 0
            for a in sel:
                b = np.array([near[0][a[0], a[1]], near[1][a[0], a[1]]])
                n = max(2, int(np.ceil(np.abs(a - b).max())) + 1)
                pth = np.round(np.linspace(b, a, n)).astype(int)
                zp = Hc[pth[:, 0], pth[:, 1]]
                zp = zp[np.isfinite(zp)]
                if len(zp) and np.abs(zp - zp[0]).max() > p["step_m"]:
                    cut += 1
            if len(sel) >= max(3, p["profiles"] // 3):
                st["cut_share"] = round(cut / len(sel), 3)
                if cut / len(sel) > 0.5:
                    why.append("cut_off")
        whys.append(why)
        stats.append(st)
    return (
        whys,
        stats,
        dict(
            ref_cells=int(ref.sum()),
            reachable_cells=int(rref.sum()),
            trajectories=len([t for t in trajs if len(t) >= 2]),
        ),
    )
