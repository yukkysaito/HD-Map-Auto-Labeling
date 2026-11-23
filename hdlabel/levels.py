"""Other levels around the ego road level (overpasses, elevated roads and rails, pedestrian decks above; the road below
a bridge) from the LiDAR point cloud map (no HD / vector map).

The labels hold only the ego road level (map.npz `height` / `cls`, hdlabel/surface.py). A scene's levels.npz marks
where another level lies above or below it, so samples can be tagged (`multilevel`) and scored separately:
  1. ego surface height on RES m cells, filled up to fill_m m around the surface (no surface off the road),
  2. per cell the nearest map point min_dz..max_dz m above it (and below it). The underside of a deck is sparse (a few
     points per m2, too few for surface.layers), so single points count; a cell whose nearest point is closer than
     `clearance` m (tree crowns, signs, vehicles left in the map) is not under / over another level,
  3. neighbouring cells are joined when their heights differ by < step m per RES m; a connected part is another level
     when it covers min_area m2 and is thin (median vertical extent of the points within `window` m beyond the nearest
     one <= max_spread m: a deck, a road, a roof or a tunnel ceiling, not a tree crown).
So `over` also marks noise enclosures, tunnel ceilings and roofs over the road: everything a camera sees as a level
above. `under` marks the ground below a bridge / viaduct deck.
levels.npz: over / under float16 [nx, ny] (RES m cells: dz of that level above / below the ego surface, m; NaN =
none), lo, res, origin (cell (i, j) covers map x = origin.x + lo.x + i * res, as map.npz).

usage: python -m hdlabel.levels <dataset.json> --map <pointcloud_map.pcd> [--out DIR]
       writes <scene>/levels.npz for every scene (into DIR/<scene>/ with --out)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .bev import SceneMap, grid
from .surface import map_files, read_pcd

RES = 1.0  # m, levels.npz cell


def ego_reference(H, sres, shape, fill_m=20.0):
    """ego surface height [X, Y] (sres cells) -> RES m cells of `shape`, filled from the nearest surface cell up to
    fill_m m away (NaN beyond)."""
    ci, cj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    g = np.stack([(ci + 0.5) * RES / sres - 0.5, (cj + 0.5) * RES / sres - 0.5])
    ref = map_coordinates(H, g, order=0, mode="constant", cval=np.nan)
    if not np.isfinite(ref).any():
        return ref
    d, (fi, fj) = distance_transform_edt(~np.isfinite(ref), return_indices=True)
    return np.where(d * RES <= fill_m, ref[fi, fj], np.nan)


def level_parts(Z, spread, step=1.5, min_area=150.0, max_spread=2.0):
    """Z [X, Y] distance of the nearest point to the ego surface (NaN = none), spread [X, Y] vertical extent of the
    points within window m beyond it -> bool [X, Y]: cells of connected parts (8-neighbours whose Z differ by < step per
    RES m) of min_area m2 or more with a median spread <= max_spread m (a deck / road is a thin slab; a tree crown is
    deep)."""
    X, Y = Z.shape
    idx = np.arange(X * Y).reshape(X, Y)
    rows, cols = [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = (slice(0, X - di), slice(max(0, -dj), Y - max(0, dj)))
        b = (slice(di, X), slice(max(0, dj), Y - max(0, -dj)))
        dz = np.abs(Z[a] - Z[b])
        e = np.isfinite(dz) & (dz < step * np.hypot(di, dj))
        rows.append(idx[a][e])
        cols.append(idx[b][e])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, comp = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(X * Y, X * Y)), directed=False)
    have = np.isfinite(Z).ravel()
    c, sp = comp[have], spread.ravel()[have]
    n = comp.max() + 1
    area = np.bincount(c, minlength=n) * RES * RES
    med = np.full(n, np.inf)
    if len(c):  # median spread per part
        o = np.lexsort((sp, c))
        c, sp = c[o], sp[o]
        u, first, cnt = np.unique(c, return_index=True, return_counts=True)
        med[u] = sp[first + cnt // 2]
    ok = (area >= min_area) & (med <= max_spread)
    return (have & ok[comp]).reshape(X, Y)


def other_levels(P, H, lo, sres, min_dz=2.0, clearance=4.0, max_dz=30.0, window=10.0, fill_m=20.0, **kw):
    """P: map points relative to origin [N, >=3]; H: ego surface height [X, Y] (relative to origin, NaN = none) on
    sres m cells from lo. Returns over, under float32 [nx, ny] on RES m cells from lo (dz of the other level to the ego
    surface, NaN = none). kw: level_parts()."""
    shape = tuple(np.ceil(np.array(H.shape) * sres / RES).astype(int))
    ref = ego_reference(H, sres, shape, fill_m)
    q = np.floor((P[:, :2] - lo) / RES).astype(np.int64)
    ins = np.all((q >= 0) & (q < shape), 1)
    q = q[ins]
    dz = P[ins, 2] - ref[q[:, 0], q[:, 1]]
    out = []
    for sign in (1, -1):
        v = sign * dz
        s = np.isfinite(v) & (v >= min_dz) & (v <= max_dz)
        Z = np.full(shape, np.inf)
        np.minimum.at(Z, (q[s, 0], q[s, 1]), v[s])  # nearest point on that side (a crown low over the road counts)
        far = np.full(shape, -np.inf)
        s &= v <= Z[q[:, 0], q[:, 1]] + window
        np.maximum.at(far, (q[s, 0], q[s, 1]), v[s])
        Z[~np.isfinite(Z) | (Z < clearance)] = np.nan  # something lower than a clearance: not a level above / below
        out.append(np.where(level_parts(Z, far - Z, **kw), sign * Z, np.nan).astype(np.float32))
    return out[0], out[1]


def scene_levels(scene_dir: str, map_path: str, **kw) -> dict:
    """levels.npz content of a dataset scene (its map.npz + the point cloud map tiles over its extent)."""
    m = np.load(f"{scene_dir}/map.npz")
    H, lo, sres, origin = m["height"], m["lo"], float(m["surf_res"]), m["origin"]
    ext = np.array(H.shape) * sres
    xs, ys = np.meshgrid(np.arange(0, ext[0] + 20, 20), np.arange(0, ext[1] + 20, 20), indexing="ij")
    files = map_files(map_path, np.c_[xs.ravel(), ys.ravel()] + lo + origin[:2], 20.0)
    P = np.concatenate([read_pcd(f) for f in files]).astype(np.float64) - origin
    over, under = other_levels(P, H, lo, sres, **kw)
    return dict(over=over.astype(np.float16), under=under.astype(np.float16), lo=lo, res=RES, origin=origin)


def write(scene_dir: str, map_path: str, out_dir: str | None = None) -> str:
    """scene_levels -> OUT_DIR/levels.npz (default: the scene directory); returns a summary line."""
    out_dir = out_dir or scene_dir
    os.makedirs(out_dir, exist_ok=True)
    r = scene_levels(scene_dir, os.path.expanduser(map_path))
    np.savez_compressed(f"{out_dir}/levels.npz", **r)
    n = np.isfinite(r["over"]).sum(), np.isfinite(r["under"]).sum()
    return f"other level above {n[0] * RES**2:.0f} m2, below {n[1] * RES**2:.0f} m2"


class Levels:
    """a scene's levels.npz, looked up for the BEV grid of a pose."""

    def __init__(self, path: str):
        m = np.load(path)
        self.over, self.under = np.isfinite(m["over"]), np.isfinite(m["under"])
        self.lo, self.res, self.origin = m["lo"], float(m["res"]), m["origin"]

    def crop(self, smap: SceneMap, T: np.ndarray, x, y, res: float) -> np.ndarray:
        """uint8 [nx, ny] on the grid of smap.crop(T, x, y, res) (the scene's SceneMap, same origin): bit 1 = another
        level above, bit 2 = below, at the map xy where each cell meets the ego surface."""
        Xg, Yg = grid(x, y, res)
        P, _ = smap.surface_points(T, np.stack([Xg.ravel(), Yg.ravel()], 1))
        ij = np.floor((P[:, :2] - self.lo) / self.res).astype(int)
        ins = np.all((ij >= 0) & (ij < self.over.shape), 1)
        out = np.zeros(len(P), np.uint8)
        i, j = ij[ins, 0], ij[ins, 1]
        out[ins] = self.over[i, j].astype(np.uint8) + 2 * self.under[i, j].astype(np.uint8)
        return out.reshape(Xg.shape)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset.json")
    p.add_argument("--map", required=True, help="pointcloud_map.pcd directory (tiles) or a single .pcd file")
    p.add_argument("--out", help="write DIR/<scene>/levels.npz instead of into the dataset")
    p.add_argument("--scenes", nargs="*", help="only these scenes")
    a = p.parse_args(argv)
    root = os.path.dirname(os.path.abspath(a.dataset))
    for s in json.load(open(a.dataset))["scenes"]:
        if not a.scenes or s["scene"] in a.scenes:
            d = os.path.join(root, s["path"])
            print(f"{s['scene']}: {write(d, a.map, os.path.join(a.out, s['path']) if a.out else None)}")


if __name__ == "__main__":
    main()
