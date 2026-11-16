"""Ego-frame (BEV) labels from a scene's map-frame labels: used by export (QA, previews), the training loader and the
review sheets, so every consumer crops the same way.

A dataset scene stores its labels once, in the map frame (map.npz: class raster 0.1 m + road surface height 0.2 m,
vectors.geojson: 3D vectors). A sample stores only its pose T_map_base. crop() / vectors() turn them into ego-frame
labels for ANY range and resolution:
  raster   grid x in [x0, x1], y in [y0, y1] of base_link (x forward, y left), row 0 = far (x max), col 0 = left (y max).
           The BEV plane is base_link's x-y plane; each cell takes the class where the line along base_link z meets
           the 3D road surface (slopes / banking handled). height = that surface point's z in base_link (NaN = none).
  vectors  elements transformed in 3D into base_link and clipped to the range (lines split where they leave it,
           polygons intersected); lane line attributes (style, painted segments) are cut to the visible part.
"""

from __future__ import annotations

import json

import numpy as np
import shapely
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

LINE_ATTRS = ("observed", "type", "directions", "shape", "style", "line_id", "width", "confidence", "observations")
POLY_ATTRS = ("type", "confidence", "observations")


class SceneMap:
    def __init__(self, map_npz: str, vectors_geojson: str | None = None):
        m = np.load(map_npz)
        self.cls, self.H = m["cls"], m["height"]
        self.lo, self.res, self.sres, self.origin = m["lo"], float(m["res"]), float(m["surf_res"]), m["origin"]
        self.lines, self.polys = [], []
        if vectors_geojson:
            for f in json.load(open(vectors_geojson))["features"]:
                g, pr = f["geometry"], f["properties"]
                if g["type"] == "LineString":
                    self.lines.append((*_dens(g["coordinates"]), pr))
                else:
                    self.polys.append(([_xyz(r) for r in g["coordinates"]], pr))

    def surface_points(self, T: np.ndarray, B: np.ndarray, iters: int = 3):
        """base_link xy grid points B [N, 2] -> map points (relative to origin) on the surface along base_link z, and
        their base_link z (NaN where there is no surface)."""
        R, t = T[:3, :3], T[:3, 3] - self.origin
        h = np.zeros(len(B))
        for _ in range(iters):
            P = np.c_[B, h] @ R.T + t
            zs = map_coordinates(
                self.H, ((P[:, :2] - self.lo) / self.sres).T - 0.5, order=1, mode="constant", cval=np.nan
            )
            h = np.where(np.isfinite(zs), (zs - t[2] - R[2, 0] * B[:, 0] - R[2, 1] * B[:, 1]) / R[2, 2], 0.0)
        return P, np.where(np.isfinite(zs), h, np.nan)

    def crop(self, T: np.ndarray, x=(-20.0, 60.0), y=(-30.0, 30.0), res: float = 0.2):
        """-> cls uint8 [nx, ny] (0 = unobserved), height float32 [nx, ny] (base_link z of the surface, NaN = none)."""
        Xg, Yg = grid(x, y, res)
        B = np.stack([Xg.ravel(), Yg.ravel()], 1)
        P, hz = self.surface_points(T, B)
        ij = np.floor((P[:, :2] - self.lo) / self.res).astype(int)
        inside = np.all((ij >= 0) & (ij < self.cls.shape), 1)
        c = np.zeros(len(B), np.uint8)
        c[inside] = self.cls[ij[inside, 0], ij[inside, 1]]
        return c.reshape(Xg.shape), hz.astype(np.float32).reshape(Xg.shape)

    def vectors(self, T: np.ndarray, x=(-20.0, 60.0), y=(-30.0, 30.0), min_len: float = 0.5) -> list[dict]:
        """map vectors in base_link, clipped to the range. Lines: points (thinned at 5 cm), plus lane line length /
        painted share / painted segments (m along the visible part); polygons: rings (outer first)."""
        Ti = np.linalg.inv(T)
        X0, X1 = x
        Y0, Y1 = y
        els = []
        for P, s, pr in self.lines:
            Q = P @ Ti[:3, :3].T + Ti[:3, 3]
            ins = (Q[:, 0] >= X0) & (Q[:, 0] <= X1) & (Q[:, 1] >= Y0) & (Q[:, 1] <= Y1)
            if not ins.any():
                continue
            for run in np.split(np.arange(len(Q)), np.flatnonzero(np.diff(ins.astype(int))) + 1):
                if not ins[run[0]] or len(run) < 2 or s[run[-1]] - s[run[0]] < min_len:
                    continue
                sr = s[run]
                L = float(sr[-1] - sr[0])
                e = dict(kind=pr["kind"], points=_rnd(_thin(Q[run])))
                e.update({k: pr[k] for k in LINE_ATTRS if k in pr})
                if pr["kind"] == "lane_line":
                    sg = [
                        [max(u, sr[0]) - sr[0], min(v, sr[-1]) - sr[0]]
                        for u, v in pr["segments"]
                        if v > sr[0] and u < sr[-1]
                    ]
                    e.update(
                        painted=round(sum(v - u for u, v in sg) / L, 3),
                        length=round(L, 2),
                        segments=[[round(u, 2), round(v, 2)] for u, v in sg],
                    )
                if pr["kind"] == "arrow":  # paint outline (all pieces) in base_link
                    e["outlines"] = [_rnd(_xyz(o) @ Ti[:3, :3].T + Ti[:3, 3]) for o in pr.get("outlines", [])]
                els.append(e)
        BOX = box(X0, Y0, X1, Y1)
        for rings, pr in self.polys:
            Qr = [r @ Ti[:3, :3].T + Ti[:3, 3] for r in rings]
            q = Qr[0]
            if q[:, 0].max() < X0 or q[:, 0].min() > X1 or q[:, 1].max() < Y0 or q[:, 1].min() > Y1:
                continue
            g = Polygon(q[:, :2], [r[:, :2] for r in Qr[1:]]).buffer(0).intersection(BOX)
            allp = np.concatenate(Qr)
            tree = cKDTree(allp[:, :2])
            for part in getattr(g, "geoms", [g]):
                if part.is_empty or part.geom_type != "Polygon" or part.area < 0.2:
                    continue
                rr = []
                for ring in [part.exterior] + list(part.interiors):
                    xy = np.array(ring.coords)
                    z = allp[tree.query(xy)[1], 2]
                    rr.append(_rnd(np.c_[xy, z]))
                e = dict(kind=pr["kind"], rings=rr)
                e.update({k: pr[k] for k in POLY_ATTRS if k in pr})
                els.append(e)
        return els


def grid(x, y, res):
    """BEV cell centres (base_link): Xg, Yg [nx, ny], row 0 = far, col 0 = left."""
    xs = np.arange(x[1] - res / 2, x[0], -res)
    ys = np.arange(y[1] - res / 2, y[0], -res)
    return np.meshgrid(xs, ys, indexing="ij")


def resample(P: np.ndarray, n: int) -> np.ndarray:
    """polyline -> n points equally spaced along its horizontal length."""
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, d]) for d in range(P.shape[1])], 1)


def _xyz(c):
    c = np.array([[p[0], p[1], np.nan if len(p) < 3 or p[2] is None else p[2]] for p in c], float)
    if np.isnan(c[:, 2]).all():
        c[:, 2] = 0.0
    elif np.isnan(c[:, 2]).any():
        k = np.isfinite(c[:, 2])
        c[~k, 2] = np.interp(np.flatnonzero(~k), np.flatnonzero(k), c[k, 2])
    return c


def _dens(c, step: float = 0.1):
    """map polyline -> points every `step` m (for clipping) and their arc length."""
    c = _xyz(c)
    s0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))]
    s = np.arange(0, s0[-1], step)
    if len(s) == 0 or s[-1] < s0[-1]:
        s = np.r_[s, s0[-1]]
    return np.stack([np.interp(s, s0, c[:, d]) for d in range(3)], 1), s


def _thin(P, tol: float = 0.05):
    if len(P) < 3:
        return P
    keep = shapely.simplify(shapely.LineString(P[:, :2]), tol).coords
    return P[sorted({int(np.argmin(np.linalg.norm(P[:, :2] - q, axis=1))) for q in keep})]


_rnd = lambda P: np.round(np.asarray(P), 3).tolist()
