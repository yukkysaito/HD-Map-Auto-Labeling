"""3D road surface of the ego road level from the LiDAR point cloud map (no HD / vector map), multi-level safe.

A height field H(x, y) cannot hold two road levels at one place (bridges, elevated roads, pedestrian decks, ramps
crossing over), so the level is chosen by connectivity, not by a height band:
  1. points within `corridor` m (horizontal) of the trajectories,
  2. per 0.5 m cell the points are split into vertical layers (gaps > `gap` m); each layer's height is its 10th
     percentile,
  3. layers are graph nodes, joined to the layers of the 8 neighbour cells when the height step is drivable / walkable
     (< `step` m per 0.5 m: curbs pass, walls and deck edges do not); Dijkstra from the trajectory seeds (the layer
     under each trajectory point) and every cell takes its layer reached first (shortest path). Under a deck the road
     is reached long before the deck (only via ramps / stairs far away), so the deck is never chosen; on a bridge the
     road below is not connected at all,
  4. robust coarse surface from the chosen layers (normalised convolution, cells off by > 0.3 / 0.15 m dropped:
     cars, vegetation), ground points = |z - coarse| < 0.15 m, splatted and filled by multi-scale normalised
     convolution (finest scale with >= 3 points within ~sigma wins).
Grid convention: cell (i, j) covers map x = origin.x + lo.x + i * res, y = origin.y + lo.y + j * res.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CELL = 0.5                                                          # layer cell size (m)


def read_pcd(path: str, fields=("x", "y", "z")) -> np.ndarray:
    """binary float PCD (Autoware map tiles) -> [N, len(fields)] float32."""
    b = open(path, "rb").read(); h = b[:b.index(b"\nDATA ") + 1].decode()
    names = re.search(r"^FIELDS (.*)$", h, re.M).group(1).split(); data = b[len(h):]
    kind, data = data.split(b"\n", 1)
    if kind.strip() != b"DATA binary" or re.search(r"^TYPE (.*)$", h, re.M).group(1).split() != ["F"] * len(names):
        raise ValueError(f"{path}: only binary float PCD is supported")
    arr = np.frombuffer(data, np.float32).reshape(-1, len(names))
    return arr[:, [names.index(f) if f in names else 0 for f in fields]] * np.array([f in names for f in fields], np.float32)


def write_pcd(path: str, xyz: np.ndarray, fields: dict | None = None) -> None:
    """binary PCD with x y z (float32) and extra per-point fields {name: array}; 'rgb' is packed as uint32."""
    fields = fields or {}; cols = [np.asarray(xyz, np.float32)]; names, sizes, types = ["x", "y", "z"], [4, 4, 4], ["F", "F", "F"]
    for k, v in fields.items():
        v = np.asarray(v)
        if k == "rgb":
            v = v.astype(np.uint32); v = (v[:, 0] << 16 | v[:, 1] << 8 | v[:, 2]).view(np.float32)[:, None]; t = "U"
        else:
            t = "F" if v.dtype.kind == "f" else "U"; v = v.astype(np.float32 if t == "F" else np.uint32).view(np.float32)[:, None]
        cols.append(v); names.append(k); sizes.append(4); types.append(t)
    arr = np.ascontiguousarray(np.concatenate(cols, 1).astype(np.float32)); n = len(arr)
    head = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS {' '.join(names)}\nSIZE {' '.join(map(str, sizes))}\nTYPE {' '.join(types)}\n"
            f"COUNT {' '.join('1' * len(names))}\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f: f.write(head.encode()); f.write(arr.tobytes())


def map_files(map_path: str, xy: np.ndarray, reach: float) -> list[str]:
    """Autoware map tiles (<name>_<x>_<y>.pcd, tile origin in the name) within `reach` m of any of the points xy (map
    frame); a single .pcd file is returned as is."""
    if not os.path.isdir(map_path): return [map_path]
    files = sorted(glob.glob(f"{map_path}/*.pcd"))
    org = np.array([[float(v) for v in re.findall(r"_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.pcd$", f)[0]] for f in files])
    size = float(np.min(np.diff(np.unique(org[:, 0])))) if len(org) > 1 else 1e9
    d, _ = cKDTree(xy).query(org + size / 2)
    return [f for f, k in zip(files, d < reach + size) if k]


def densify(g: np.ndarray, step: float = 0.2) -> np.ndarray:
    """polyline [N, 3] resampled every `step` m (horizontal arc length)."""
    if len(g) < 2: return g
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g[:, :2], axis=0), axis=1))]; s = np.arange(0, d[-1] + 1e-9, step)
    return np.stack([np.interp(s, d, g[:, k]) for k in range(g.shape[1])], 1)


def layers(P: np.ndarray, gap: float = 0.4, min_pts: int = 3, max_layers: int = 4):
    """points -> per 0.5 m cell up to max_layers vertical layers (lowest first).
    Returns cells [M, 2] (int cell coords), heights [M, max_layers] (NaN = none)."""
    q = np.floor(P[:, :2] / CELL).astype(np.int64); key = q[:, 0] * 4_000_037 + q[:, 1]
    o = np.lexsort((P[:, 2], key)); key, z, q = key[o], P[o, 2], q[o]
    new = np.r_[True, (key[1:] != key[:-1]) | (np.diff(z) > gap)]   # new layer: new cell or a vertical gap
    st = np.flatnonzero(new); cnt = np.diff(np.r_[st, len(z)])
    lz = z[st + (cnt * 0.1).astype(int)]; good = cnt >= min_pts
    st, lz = st[good], lz[good]; lkey = key[st]
    ukey, first, nper = np.unique(lkey, return_index=True, return_counts=True)
    rank = np.arange(len(lkey)) - np.repeat(first, nper)
    H = np.full((len(ukey), max_layers), np.nan); keep = rank < max_layers
    H[np.repeat(np.arange(len(ukey)), nper)[keep], rank[keep]] = lz[keep]
    return q[st[first]], H


def connected_level(cells: np.ndarray, Hl: np.ndarray, seeds: np.ndarray, step: float = 0.25, limit: float = 80.0):
    """Dijkstra over (cell, layer) nodes from seed points [S, 3] (xyz, layer = nearest within 1 m of the seed z).
    Returns per cell the chosen layer height (NaN = not connected within `limit` m) and the path distance."""
    M, L = Hl.shape
    node = lambda k, l: k * L + l
    rows, cols, w = [], [], []
    offs = [(di, dj) for di in range(3) for dj in range(-2, 3) if di > 0 or dj > 0]   # 5 x 5 (bridges 1-cell holes), each pair once
    for di, dj in offs:
        nb = _lookup(cells, cells + [di, dj]); a_ = np.flatnonzero(nb >= 0); b_ = nb[a_]
        dist = CELL * np.hypot(di, dj)
        for la in range(L):
            for lb in range(L):
                ha, hb = Hl[a_, la], Hl[b_, lb]; e = np.isfinite(ha) & np.isfinite(hb) & (np.abs(ha - hb) < step * np.hypot(di, dj))
                rows.append(node(a_[e], la)); cols.append(node(b_[e], lb)); w.append(np.full(e.sum(), dist))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    G = coo_matrix((w, (rows, cols)), shape=(M * L, M * L)).tocsr()
    k = _lookup(cells, np.floor(seeds[:, :2] / CELL).astype(np.int64)); s = k >= 0
    dz = np.abs(Hl[k[s]] - seeds[s, 2:3]); dz[~np.isfinite(dz)] = np.inf; near = dz.min(1) < 1.0
    src = np.unique(node(k[s][near], dz[near].argmin(1)))
    if not len(src): return np.full(M, np.nan), np.full(M, np.inf)
    D = dijkstra(G, directed=False, indices=src, min_only=True, limit=limit).reshape(M, L)
    best = np.argmin(D, 1); dist = D[np.arange(M), best]
    h = np.where(np.isfinite(dist), Hl[np.arange(M), best], np.nan)
    return h, dist


class Grid:
    def __init__(self, lo, shape, res, origin):
        self.lo, self.shape, self.res, self.origin = np.asarray(lo, float), tuple(int(v) for v in shape), float(res), np.asarray(origin, float)

    def splat(self, P, w=None):
        ij = np.clip(np.floor((P[:, :2] - self.lo) / self.res).astype(int), 0, np.array(self.shape) - 1)
        c = np.ravel_multi_index(tuple(ij.T), self.shape); w = np.ones(len(P)) if w is None else w
        n = int(np.prod(self.shape))
        return np.bincount(c, P[:, 2] * w, n).reshape(self.shape), np.bincount(c, w, n).reshape(self.shape)

    def at(self, G, P, cval=None):
        g = ((P[:, :2] - self.lo) / self.res).T - 0.5
        return map_coordinates(G, g, order=1, mode="nearest" if cval is None else "constant", cval=np.nan if cval is None else cval)

    def norm_conv(self, Z, W, sig_m):
        sw = gaussian_filter(W, sig_m / self.res); return gaussian_filter(Z, sig_m / self.res) / np.maximum(sw, 1e-9), sw


def build(P: np.ndarray, trajs: list[np.ndarray], grid: Grid, corridor: float = 40.0, step: float = 0.25) -> dict:
    """P: map points relative to grid.origin [N, >=3]; trajs: trajectories (base_link xyz, same frame).
    Returns dict(height, height_w (grid), ground (indices into P of the ground points), info)."""
    traj = np.concatenate([densify(t) for t in trajs]); tree = cKDTree(traj[:, :2])
    dist, _ = tree.query(P[:, :2], workers=-1); P = P[dist < corridor]; idx_in = np.flatnonzero(dist < corridor)
    cells, Hl = layers(P[:, :3])
    lvl, path = connected_level(cells, Hl, traj, step=step, limit=2 * corridor)
    ok = np.isfinite(lvl); low = np.c_[(cells[ok] + 0.5) * CELL, lvl[ok]]
    keep = np.ones(len(low), bool)
    for thr, sig in ((0.3, 3.0), (0.15, 2.0)):                     # robust coarse surface of the chosen level
        coarse, _ = grid.norm_conv(*grid.splat(low[keep]), sig); keep = np.abs(low[:, 2] - grid.at(coarse, low)) < thr
    coarse, cw = grid.norm_conv(*grid.splat(low[keep]), 2.0)
    # ground points: near the coarse surface and in a cell whose chosen level they belong to (no deck points above a road)
    lk = _lookup(cells, np.floor(P[:, :2] / CELL).astype(np.int64))
    lv = np.where(lk >= 0, lvl[np.maximum(lk, 0)], np.nan)
    g = (np.abs(P[:, 2] - grid.at(coarse, P)) < 0.15) & (np.abs(P[:, 2] - lv) < 0.5)
    Z, W = grid.splat(P[g])
    H = np.full(grid.shape, np.nan); HW = np.zeros(grid.shape)
    for sig in (2.5, 1.0, 0.4):                                     # finest scale with >= 3 ground points within ~sigma wins
        h, sw = grid.norm_conv(Z, W, sig); good = sw * 2 * np.pi * (sig / grid.res) ** 2 > 3; H[good] = h[good]; HW[good] = sw[good]
    multi = int((np.isfinite(Hl).sum(1) > 1).sum())
    info = dict(points=int(len(P)), ground_points=int(g.sum()), layer_cells=int(len(cells)), multi_layer_cells=multi,
                connected_cells=int(ok.sum()), unconnected_cells=int((~ok).sum()), covered=round(float(np.isfinite(H).mean()), 3))
    return dict(height=H, height_w=HW, ground=idx_in[g], info=info)


def _lookup(cells, pc):
    """cell coords -> row in cells (-1 if absent), vectorised."""
    key = cells[:, 0] * 4_000_037 + cells[:, 1]; o = np.argsort(key); sk = key[o]
    q = pc[:, 0] * 4_000_037 + pc[:, 1]; pos = np.minimum(np.searchsorted(sk, q), len(sk) - 1)
    return np.where(sk[pos] == q, o[pos], -1)
