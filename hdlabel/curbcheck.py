"""Road-adjacency, first-edge and shape checks of curb lines (vectorize.py --curb-filter).

A road kerb is the first raised edge met going sideways from the driven lanes, and it has road on one side. Rules per
curb line (grid-local xy, metres; label map cls with label ids C); a line that fails any of them is listed with the
failed rules (vectorize.py turns it into a curb_candidate):
  not_road_adjacent  less than min_adjacent of its length is ON THE ROAD EDGE: within adj_m of the reachable road
                     going only through curb (or unobserved) cells - a planter / lawn border behind a sidewalk, hedge
                     or verge strip, a wall base or a piece on the sidewalk at bollards / trees is not. Reachable road
                     = road, marking, crosswalk, manhole cells (limited to the LiDAR road surface) of the connected
                     regions of >= min_area m2 within reach_m of the trajectories of all drives (so the opposite
                     carriageway across a median and side streets count, road-coloured patches on a plaza do not).
  not_first_edge     rays cast sideways from the trajectories (every ray_step m, up to ray_m) give the points where
                     they leave the reachable road. A line with less than min_first of its length within first_tol of
                     such points and less than strong_adjacent of it on the road edge is not the first edge.
  loop               shorter than loop_max_m and closed (ends < loop_gap m apart) or turning >= 300 deg in total:
                     drainage gratings, patches (longer rings are islands / medians)
  wavy               mean absolute turning > wavy_deg_per_m (resampled at 1 m), lines >= 3 m: vegetation edges
  edge_of_view       shorter than edge_short m and within 0.5 m of unobserved cells over >= 30 % of its length
Without a usable trajectory only the shape / edge rules apply.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

DEFAULTS = dict(adj_m=2.0, min_adjacent=0.3, strong_adjacent=0.6, reach_m=0.0, min_area=50.0, ray_step=1.0,
                ray_m=30.0, first_tol=1.5, min_first=0.15, loop_gap=0.5, loop_max_m=10.0, wavy_deg_per_m=30.0,
                edge_short=3.0)  # fmt: skip


def resample(xy: np.ndarray, step: float = 0.5) -> np.ndarray:
    xy = np.asarray(xy, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if s[-1] <= 0:
        return xy[:1]
    at = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.stack([np.interp(at, s, xy[:, k]) for k in (0, 1)], 1)


def turning(xy: np.ndarray, step: float) -> np.ndarray:
    """signed heading changes (rad) between consecutive segments of the line resampled at step m."""
    q = resample(xy, step)
    if len(q) < 3:
        return np.zeros(0)
    d = np.diff(q, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    return (np.diff(h) + np.pi) % (2 * np.pi) - np.pi


def shape_reasons(xy: np.ndarray, p: dict) -> list[str]:
    xy = np.asarray(xy, float)
    L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    out = []
    closed = L >= 1.5 and np.linalg.norm(xy[-1] - xy[0]) < p["loop_gap"]
    if L < p["loop_max_m"] and (closed or abs(turning(xy, 0.5).sum()) >= np.radians(300)):
        out.append("loop")
    t = turning(xy, 1.0)
    if L >= 3.0 and len(t) and np.degrees(np.abs(t).sum()) / L > p["wavy_deg_per_m"]:
        out.append("wavy")
    return out


def reachable(cls, C, seeds_ij, res, surface=None, reach_m=15.0, min_area=50.0):
    """drivable cells (closed over 0.2 m gaps, limited to the surface mask if given) of the connected regions of at
    least min_area m2 that come within reach_m of a seed cell."""
    dr = np.isin(cls, [C["road"], C["marking"], C["crosswalk"], C["manhole"]]).astype(np.uint8)
    dr = cv2.morphologyEx(dr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if surface is not None:
        dr &= surface.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dr, connectivity=8)
    f = max(1, int(round(1.0 / res)))  # seed neighbourhood on a 1 m grid
    near = np.zeros((cls.shape[0] // f + 1, cls.shape[1] // f + 1), np.uint8)
    ok = np.all((seeds_ij >= 0) & (seeds_ij < cls.shape), 1)
    near[seeds_ij[ok, 0] // f, seeds_ij[ok, 1] // f] = 1
    r = int(round(reach_m / (f * res)))
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ii, jj = np.nonzero(dr)
    ids = np.unique(lab[ii, jj][near[ii // f, jj // f]])
    ids = ids[(ids > 0) & (st[ids, cv2.CC_STAT_AREA] * res**2 >= min_area)]
    return np.isin(lab, ids) if len(ids) else np.zeros(cls.shape, bool)


def road_edge(cls, C, reach, res, adj_m):
    """cells within adj_m of the reachable road on paths through curb / unobserved cells only (geodesic dilation)."""
    through = ((cls == C["curb"]) | (cls == 0)).astype(np.uint8)
    edge = cv2.dilate(reach.astype(np.uint8), np.ones((3, 3), np.uint8))
    for _ in range(max(1, int(round(adj_m / res)))):
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8)) & (through | edge)
    return edge > 0


def first_hits(cls, C, trajs, lo, res, p, reach):
    """points (grid-local xy) where rays cast sideways from the trajectories leave the reachable road: a non-drivable
    observed cell (curb, sidewalk, off-road, other) within 0.3 m after a reachable road cell. The first one is the edge
    of the driven road, later ones edges of the roads across a median; a planter behind a sidewalk is never one."""
    barrier = np.isin(cls, [C["curb"], C["sidewalk"], C["offroad"], C["other"]])
    hits = []
    steps = (np.arange(int(p["ray_m"] / res)) * res)[:, None]
    for P in trajs:
        q = resample(P, p["ray_step"])
        if len(q) < 2:
            continue
        d = np.gradient(q, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        for side in (1.0, -1.0):
            nrm = side * np.c_[-d[:, 1], d[:, 0]]
            ray = q[:, None, :] + steps[None] * nrm[:, None, :]  # (rays, steps, 2)
            ij = np.floor((ray - lo) / res).astype(int)
            ok = np.all((ij >= 0) & (ij < cls.shape), 2)
            b, rr = np.zeros(ok.shape, bool), np.zeros(ok.shape, bool)
            b[ok] = barrier[ij[ok][:, 0], ij[ok][:, 1]]
            rr[ok] = reach[ij[ok][:, 0], ij[ok][:, 1]]
            after = np.zeros_like(rr)  # a reachable road cell within the last 3 steps
            for s in (1, 2, 3):
                after[:, s:] |= rr[:, :-s]
            hits.append(ray[b & after])
    return np.concatenate(hits).reshape(-1, 2) if hits else np.zeros((0, 2))


def check_curbs(lines, cls, C, lo, res, trajs=(), surface=None, params=None):
    """lines: list of (N, 2) grid-local xy curb lines -> (list of failed rules per line (empty: a kerb), stats)."""
    from scipy.spatial import cKDTree

    p = dict(DEFAULTS, **(params or {}))
    trajs = [np.asarray(t, float) for t in trajs if len(t) >= 2]
    obs = gaussian_filter((cls > 0).astype(np.float32), 1.5) > 0.5
    d_unobs = distance_transform_edt(obs) * res
    reach = None
    if trajs:
        seeds = np.floor((np.concatenate([resample(t, res) for t in trajs]) - lo) / res).astype(int)
        reach = reachable(cls, C, seeds, res, surface, p["reach_m"], p["min_area"])
        if not reach.any():
            reach = None
    if reach is not None:
        d_edge = distance_transform_edt(~road_edge(cls, C, reach, res, p["adj_m"])) * res
        hits = first_hits(cls, C, trajs, lo, res, p, reach)
        tree = cKDTree(hits) if len(hits) else None
    out = []
    for xy in lines:
        xy = np.asarray(xy, float)
        L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        q = resample(xy, 0.5)
        ij = np.clip(np.floor((q - lo) / res).astype(int), 0, np.array(cls.shape) - 1)
        why = shape_reasons(xy, p)
        if L < p["edge_short"] and (d_unobs[ij[:, 0], ij[:, 1]] < 0.5).mean() >= 0.3:
            why.append("edge_of_view")
        if reach is not None:
            adj = float((d_edge[ij[:, 0], ij[:, 1]] <= 0.3).mean())
            first = 0.0
            if tree is not None:
                first = float((tree.query(q, distance_upper_bound=p["first_tol"])[0] <= p["first_tol"]).mean())
            if adj < p["min_adjacent"]:
                why.append("not_road_adjacent")
            elif first < p["min_first"] and adj < p["strong_adjacent"]:
                why.append("not_first_edge")
        out.append(why)
    return out, dict(reach_cells=int(reach.sum()) if reach is not None else 0, trajectories=len(trajs))
