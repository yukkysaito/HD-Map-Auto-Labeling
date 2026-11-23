"""Road-level refinement of the road class: road-coloured surfaces behind a kerb step are not the road.

The segmentation labels any asphalt-like surface road: a strip between a kerb and a wall, a paved verge or a plaza
next to the carriageway. The LiDAR map surface (map_surface.py, height grid of the ego road level) shows the kerb step
between them and the road the vehicles drove on. Steps (on the surface grid, surf_res):
  level road  drivable cells (road, marking, crosswalk) with LiDAR ground, connected to the trajectory cells (seeds:
              trajectory points whose height is within seed_dz of the surface) through cells whose slope is at most
              `grade` (rise / run; a kerb, even smoothed over ~1 m by the surface fit, is steeper than a road)
  candidates  road cells (label grid) with LiDAR ground, not on the level road, grouped into 8-connected components
  raised      a component whose flat cells (slope <= grade: not the kerb face) within near_m of the level road lie
              above the nearest level road cell by more than step_m on their median and on at least min_share of
              them, with at least min_cells such cells (a sliver of kerb face alone is never raised), and the median
              of all its flat cells is above step_m too (a lower road across a median, its near edge on the kerb
              face, is not raised). Both medians at most max_step_m: higher is no kerb but a wall, an embankment or
              another road level (a diverging ramp), left undecided
The parts of a raised component at most max_width_m wide (a strip behind a kerb, between a kerb and a wall) become
sidewalk; wider parts (a plaza, a car park or a side street up a lowered kerb: the height cannot tell them apart) and
their ragged rims (within max_width_m / 2) become unobserved (abstain, excluded from the loss). Cells less than
step_m / 2 above the level road (the foot of a kerb that the surface fit smooths over ~1 m) stay road. Components
within keep_m of a trajectory (the vehicle drove there: humps, ramps) and components without enough known cells near
the level road are left as they are (undecidable). Road below the level road (a lower street or a sunken plaza) is
never changed. Only road cells change.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from hdlabel.common import CLASSES

ROAD, SIDEWALK = CLASSES.index("road"), CLASSES.index("sidewalk")
DRIVABLE = [CLASSES.index(c) for c in ("road", "marking", "crosswalk")]
DEFAULTS = dict(
    grade=0.1,
    step_m=0.07,
    max_step_m=0.3,
    max_width_m=3.0,
    near_m=3.0,
    min_share=0.6,
    min_cells=20,
    keep_m=1.0,
    seed_dz=1.0,
)


def _down(mask: np.ndarray, k: int, shape) -> np.ndarray:
    """label-grid bool mask -> surface grid (a surface cell is set when any of its k x k label cells is)."""
    h, w = shape
    m = np.zeros((h * k, w * k), bool)
    s = mask[: h * k, : w * k]
    m[: s.shape[0], : s.shape[1]] = s
    return m.reshape(h, k, w, k).any((1, 3))


def _up(a: np.ndarray, k: int, shape) -> np.ndarray:
    """surface grid -> label grid of `shape` (nearest; outside: the dtype's zero / False)."""
    U = np.repeat(np.repeat(a, k, 0), k, 1)[: shape[0], : shape[1]]
    return np.pad(U, [(0, shape[0] - U.shape[0]), (0, shape[1] - U.shape[1])])


def flat(H: np.ndarray, sres: float, grade: float) -> np.ndarray:
    """cells with LiDAR ground (finite H) and a slope of at most `grade` (rise / run, central differences; next to
    cells without ground the slope is unknown: flat)."""
    gi, gj = np.gradient(H, sres)
    return np.isfinite(H) & ~(np.hypot(gi, gj) > grade)


def level_road(ok: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """cells of `ok` (bool) 8-connected to a seed cell (row, col)."""
    n, lab = cv2.connectedComponents(ok.astype(np.uint8), connectivity=8)
    sd = seeds[np.all((seeds >= 0) & (seeds < ok.shape), 1)]
    ids = np.unique(lab[sd[:, 0], sd[:, 1]]) if len(sd) else np.zeros(0, int)
    return np.isin(lab, ids[ids > 0])


def _medians(c: np.ndarray, r: np.ndarray, n: int):
    """count and median of r per label c (0..n-1; nan median for an empty label)."""
    cnt = np.bincount(c, minlength=n)
    med = np.full(n, np.nan)
    order = np.argsort(c, kind="stable")
    rs = r[order]
    ends = np.cumsum(cnt)
    for i in np.flatnonzero(cnt):
        med[i] = np.median(rs[ends[i] - cnt[i] : ends[i]])
    return cnt, med


def refine(cls, H, k: int, lo, res: float, trajs=(), **kw):
    """cls (label grid, common.CLASSES ids), H (surface-grid height, nan = no LiDAR ground; surf_res = k * res), lo
    (grid corner, the trajectories' frame), trajs: list of (N, 3) xyz (grid-local xy, surface-frame z) -> (refined cls,
    stats). Without a trajectory on the level road nothing changes."""
    p = dict(DEFAULTS, **kw)
    st = dict(components=0, raised=0, changed_m2=0.0)
    trajs = [np.asarray(t, float) for t in trajs if len(t) >= 2]
    if not trajs:
        return cls, st
    sres = k * res
    P = np.concatenate(trajs)
    sij = np.floor((P[:, :2] - lo) / sres).astype(int)
    inb = np.all((sij >= 0) & (sij < H.shape), 1)
    sij, z = sij[inb], P[inb, 2]
    on = np.abs(z - H[sij[:, 0], sij[:, 1]]) <= p["seed_dz"]  # nan (no ground) is not on the level
    dr = _down(np.isin(cls, DRIVABLE), k, H.shape)
    fl = flat(H, sres, p["grade"])
    L = level_road(dr & fl, sij[on])
    st["seeds"] = int(on.sum())
    if not L.any():
        return cls, st
    d, near = distance_transform_edt(~L, return_indices=True)
    rel = H - H[near[0], near[1]]  # height above the nearest level road cell (surface grid)
    relc = _up(rel.astype(np.float32), k, cls.shape)
    relc[_up(~np.isfinite(H), k, cls.shape)] = np.nan
    dc = _up((d * sres).astype(np.float32), k, cls.shape)
    Lc = _up(L, k, cls.shape)
    cand = (cls == ROAD) & ~Lc & np.isfinite(relc)  # cells without LiDAR ground are never decided
    n, comp = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    st["components"] = int(n - 1)
    if n <= 1:
        return cls, st
    tr = np.zeros(cls.shape, np.uint8)
    for t in trajs:
        q = np.floor((t[:, :2] - lo) / res).astype(int)
        q = q[np.all((q >= 0) & (q < cls.shape), 1)]
        tr[q[:, 0], q[:, 1]] = 1
    keep = distance_transform_edt(tr == 0) * res <= p["keep_m"] if tr.any() else np.zeros(cls.shape, bool)
    flc = cand & _up(fl, k, cls.shape)  # the kerb face (steep) is never judged
    judged = flc & (dc <= p["near_m"])
    cnt, med = _medians(comp[judged], relc[judged], n)
    up = np.bincount(comp[judged], weights=relc[judged] > p["step_m"], minlength=n)
    _, med_all = _medians(comp[flc], relc[flc], n)  # all of it: a lower road across a median is not raised
    raised = (cnt >= p["min_cells"]) & (up >= p["min_share"] * np.maximum(cnt, 1))
    for m in (np.nan_to_num(med), np.nan_to_num(med_all)):
        raised &= (m > p["step_m"]) & (m <= p["max_step_m"])
    raised[0] = False
    raised[np.unique(comp[keep & cand])] = False
    out = cls.copy()
    ch = raised[comp] & cand & (relc > p["step_m"] / 2)  # not the foot of the kerb ramp (smoothed surface)
    w = int(round(p["max_width_m"] / res)) | 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
    core = cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_OPEN, disc)  # the parts a max_width_m disc fits in
    wide = ch & (cv2.dilate(core, disc) > 0)  # and their ragged rims
    out[ch] = SIDEWALK
    out[wide] = 0
    lower = np.nan_to_num(med) < -p["step_m"]
    m2 = lambda m: round(float(m.sum()) * res**2, 1)
    st.update(raised=int(raised.sum()), changed_m2=m2(ch), abstain_m2=m2(wide), off_level_m2=m2(cand),
              level_m2=round(float(L.sum()) * sres**2, 1), lower_m2=m2(lower[comp] & cand),
              undecided=int((cnt[1:] < p["min_cells"]).sum()))  # fmt: skip
    return out, st


def refine_scene(cls, ground: str, drives, origin, lo, res: float, **kw):
    """refine() with the surface grid of map_surface.py's npz `ground` (no LiDAR ground where height_w <= 0) and the
    trajectories of the drive data dirs (gt.npy: t, x, y, z, ... in the map frame; origin: the grid's map origin)."""
    g = np.load(ground)
    H = g["height"].astype(float)
    H[g["height_w"] <= 0] = np.nan
    k = max(1, int(round(float(g["surf_res"]) / res)))
    trajs = [np.load(f"{d}/gt.npy")[:, 1:4] - np.asarray(origin, float)[:3] for d in drives]
    return refine(cls, H, k, lo, res, trajs, **kw)
