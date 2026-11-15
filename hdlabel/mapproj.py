"""Projection of camera pixels onto the 3D ground surface (map frame), on the GPU. Used by time_offset, project_labels
and review (labels rendered into camera images).

A pixel ray from the camera is intersected with the surface height field H(x, y) (fixed-point iteration
t = (H(C + t d) - C_z) / d_z, which converges for rays that are not grazing; slopes and banking are handled).
Votes are accumulated on a 0.1 m grid restricted to cells where the surface exists (compact index).
Per image and cell, one unit of vote mass is split over the classes seen there (so near-range pixels do not
dominate), times a distance weight (near / max(r, near))^2.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import CLASSES, GROUPS, VERTICAL, Extrinsic, PoseInterp, stamps_of
from .common import device as default_device

GROUND_GROUPS = ["road", "marking", "crosswalk", "manhole", "curb", "sidewalk", "offroad"]
NG = len(GROUPS) + 1  # + other (never voted on the ground)


def seg_lut() -> np.ndarray:
    lut = np.full(256, len(GROUPS), np.int64)
    for g, ids in enumerate(GROUPS.values()):
        lut[ids] = g
    lut[VERTICAL] = len(GROUPS)
    return lut


class Camera:
    """keyframes of one camera of one drive: stamps, image / seg paths, intrinsics, extrinsic, poses."""

    def __init__(self, data: str, calib: str, cam: int, poser: PoseInterp):
        self.cam, self.data = cam, data
        self.files, self.st = stamps_of(f"{data}/kf_cam{cam}")
        self.seg = [f"{data}/seg_cam{cam}/{s}.png" for s in self.st]
        kd = os.path.dirname(os.path.realpath(self.files[0]))
        self.K = np.array(json.load(open(f"{kd}/K.json"))["K"])
        self.ext = Extrinsic.load(f"{calib}/cam{cam}.json")
        self.poser = poser
        # frames taken on another road level than the surface (bridge above / road below, see snap_to_ground) are not used
        off = getattr(poser, "off_level", None)
        self.on_level = (
            np.ones(len(self.st), bool)
            if off is None
            else ~off[np.clip(np.searchsorted(poser.g[:, 0], self.st + self.ext.offset * 1e9), 0, len(off) - 1)]
        )

    def T_map_cam(self, idx, offset: float | None = None) -> np.ndarray:
        off = self.ext.offset if offset is None else offset
        return self.poser(self.st[idx].astype(np.float64) + off * 1e9) @ self.ext.T


class Projector:
    def __init__(self, ground_npz: str, res: float = 0.1, device: str | None = None):
        g = np.load(ground_npz)
        self.H, self.lo, self.sres, self.origin = g["height"], g["lo"], float(g["surf_res"]), g["origin"]
        self.res, self.dev = res, torch.device(device or default_device())
        self.Ht = torch.from_numpy(np.nan_to_num(self.H, nan=-1e4)).float().to(self.dev)[None, None]
        self.Lshape = tuple(int(round(v * self.sres / res)) for v in self.H.shape)
        valid = np.repeat(np.repeat(np.isfinite(self.H), round(self.sres / res), 0), round(self.sres / res), 1)
        self.cells = np.flatnonzero(valid)  # compact index -> flat 0.1 m cell
        idx = np.full(valid.size, -1, np.int64)
        idx[self.cells] = np.arange(len(self.cells))
        self.cidx = torch.from_numpy(idx).to(self.dev)
        self.lut = torch.from_numpy(seg_lut()).to(self.dev)
        self.ground = torch.tensor([list(GROUPS).index(g) for g in GROUND_GROUPS], device=self.dev)

    def height_at(self, xy: torch.Tensor) -> torch.Tensor:
        """bilinear surface height at map-frame xy (relative to origin); -1e4 outside the surface."""
        g = (xy - torch.tensor(self.lo, device=self.dev, dtype=xy.dtype)) / self.sres - 0.5  # (row, col) = (x, y)
        n = torch.tensor(self.H.shape, device=self.dev, dtype=xy.dtype)
        grid = torch.stack([g[:, 1] / (n[1] - 1) * 2 - 1, g[:, 0] / (n[0] - 1) * 2 - 1], 1)[None, None]
        return F.grid_sample(self.Ht, grid.float(), mode="bilinear", align_corners=True, padding_mode="border")[0, 0, 0]

    def intersect(self, C: torch.Tensor, d: torch.Tensor):
        """rays C + t d (map frame relative to origin) with the surface: fixed point t = (H(C + t d) - C_z) / d_z.
        Returns points, converged mask, t."""
        h0 = self.height_at(C[None, :2].double())[0]
        t = (h0 - C[2]) / d[:, 2]
        conv = torch.zeros(len(d), dtype=torch.bool, device=self.dev)
        for _ in range(10):
            P = C + t[:, None] * d
            t_new = (self.height_at(P[:, :2]) - C[2]) / d[:, 2]
            conv = (t_new - t).abs() * d[:, :2].norm(dim=1) < 0.01
            t = t_new
        return C + t[:, None] * d, conv, t

    def new_acc(self, rgb: bool = False) -> dict:
        acc = dict(
            votes=torch.zeros(len(self.cells), NG, device=self.dev), n=torch.zeros(len(self.cells), device=self.dev)
        )
        if rgb:
            acc.update(
                rgb=torch.zeros(len(self.cells), 3, device=self.dev, dtype=torch.float64),
                w=torch.zeros(len(self.cells), device=self.dev, dtype=torch.float64),
            )  # float64: see project()
        return acc

    def project(
        self,
        T_map_cam: np.ndarray,
        K: np.ndarray,
        seg: np.ndarray,
        img: np.ndarray | None,
        acc: dict,
        stride: int = 2,
        rmax: float = 30.0,
        near: float = 8.0,
        dz: float = 0.0,
        classes=None,
    ):
        """add one image's ground pixels to acc. dz is added to the camera height (vehicle height correction)."""
        dev = self.dev
        s = torch.from_numpy(seg[::stride, ::stride]).to(dev).long()
        grp = self.lut[s]
        keep = torch.isin(grp, self.ground if classes is None else classes)
        v, u = torch.nonzero(keep, as_tuple=True)
        grp = grp[v, u]
        uv1 = torch.stack(
            [u * stride + 0.5 * (stride - 1), v * stride + 0.5 * (stride - 1), torch.ones_like(u)], 1
        ).double()
        Rm = torch.from_numpy(T_map_cam[:3, :3]).to(dev)
        C = torch.from_numpy(T_map_cam[:3, 3] - self.origin).to(dev)
        C = C + torch.tensor([0, 0, dz], device=dev, dtype=C.dtype)
        d = uv1 @ torch.from_numpy(np.linalg.inv(K)).to(dev).T @ Rm.T
        down = d[:, 2] < -0.02
        d, grp, v, u = d[down], grp[down], v[down], u[down]
        P, conv, t = self.intersect(C, d)
        r = (P[:, :2] - C[:2]).norm(dim=1)
        ok = conv & (t > 0) & (r < rmax) & (P[:, 2] > -1e3)
        P, grp, r, v, u = P[ok], grp[ok], r[ok], v[ok], u[ok]
        ij = torch.floor((P[:, :2] - torch.tensor(self.lo, device=dev)) / self.res).long()
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.Lshape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.Lshape[1])
        ij, grp, r, v, u = ij[inside], grp[inside], r[inside], v[inside], u[inside]
        ci = self.cidx[ij[:, 0] * self.Lshape[1] + ij[:, 1]]
        m = ci >= 0
        ci, grp, r, v, u = ci[m], grp[m], r[m], v[m], u[m]
        if not len(ci):
            return 0
        uc, inv, cnt = torch.unique(ci, return_inverse=True, return_counts=True)
        w = (near / torch.clamp(r, min=near)) ** 2
        frac = torch.zeros(len(uc), NG, device=dev)
        frac.index_put_((inv, grp), torch.ones_like(r).float(), accumulate=True)
        # sums over repeated indices run in an arbitrary order on the GPU: float64 makes the order-dependent rounding
        # vanish when cast back, so reruns are byte-identical (frac counts are exact integers; uc is unique)
        wc = (torch.zeros(len(uc), device=dev, dtype=torch.float64).index_add_(0, inv, w.double()) / cnt).float()
        acc["votes"].index_add_(0, uc, frac / cnt[:, None] * wc[:, None])
        acc["n"].index_add_(0, uc, torch.ones_like(wc))
        if img is not None and "rgb" in acc:
            col = torch.from_numpy(img[::stride, ::stride]).to(dev)[v, u].float()
            acc["rgb"].index_add_(0, ci, col.double() * w[:, None].double())
            acc["w"].index_add_(0, ci, w.double())
        return int(len(ci))

    def render(
        self, T_map_cam: np.ndarray, K: np.ndarray, size, cls: np.ndarray, stride: int = 2, rmax: float = 40.0
    ) -> np.ndarray:
        """label map (0.1 m raster, this grid) seen from a camera: per pixel the class where its ray meets the surface
        (0 = sky / beyond rmax / unobserved). Occluders (vehicles, buildings) are not modelled."""
        dev = self.dev
        w, h = size
        v, u = torch.meshgrid(
            torch.arange(0, h, stride, device=dev), torch.arange(0, w, stride, device=dev), indexing="ij"
        )
        uv1 = torch.stack(
            [u.ravel() + 0.5 * (stride - 1), v.ravel() + 0.5 * (stride - 1), torch.ones(u.numel(), device=dev)], 1
        ).double()
        Rm = torch.from_numpy(T_map_cam[:3, :3]).to(dev)
        C = torch.from_numpy(T_map_cam[:3, 3] - self.origin).to(dev)
        d = uv1 @ torch.from_numpy(np.linalg.inv(K)).to(dev).T @ Rm.T
        out = torch.zeros(len(d), dtype=torch.uint8, device=dev)
        down = torch.nonzero(d[:, 2] < -0.02)[:, 0]
        P, conv, t = self.intersect(C, d[down])
        ok = conv & (t > 0) & ((P[:, :2] - C[:2]).norm(dim=1) < rmax) & (P[:, 2] > -1e3)
        ij = torch.floor((P[:, :2] - torch.tensor(self.lo, device=dev)) / self.res).long()
        ok &= (ij[:, 0] >= 0) & (ij[:, 0] < cls.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < cls.shape[1])
        ct = torch.from_numpy(cls).to(dev)
        out[down[ok]] = ct[ij[ok, 0], ij[ok, 1]]
        return out.reshape(u.shape).cpu().numpy()

    def dense(self, x: torch.Tensor, fill=0) -> np.ndarray:
        """compact per-cell array -> full 0.1 m grid."""
        x = x.cpu().numpy()
        out = np.full((int(np.prod(self.Lshape)),) + x.shape[1:], fill, x.dtype)
        out[self.cells] = x
        return out.reshape(self.Lshape + x.shape[1:])


def snap_to_ground(poser: PoseInterp, ground_npz, win_s: float = 2.0, level_tol: float = 1.0) -> dict:
    """Put base_link on the map surface: z += running median (+-win_s/2) of (surface height - localization z) under
    base_link. Localization z is only loosely tied to the ground (here ~10-15 cm below the map surface); the
    calibrated camera heights are relative to base_link on the road. ground_npz: one surface or a list (several
    scenes); poses outside every surface get the nearest covered correction. Poses more than level_tol m off the
    surface are on another road level (on a bridge over it / on a road below it): poser.off_level marks them, they do
    not enter the correction and Camera.on_level excludes their frames. Modifies poser in place."""
    from scipy.ndimage import map_coordinates, median_filter

    d = np.full(len(poser.g), np.nan)
    for gf in [ground_npz] if isinstance(ground_npz, str) else ground_npz:
        g = np.load(gf)
        H, lo, sres, origin = g["height"], g["lo"], float(g["surf_res"]), g["origin"]
        xy = poser.g[:, 1:3] - origin[:2]
        h = map_coordinates(H, ((xy - lo) / sres).T - 0.5, order=1, mode="constant", cval=np.nan) + origin[2]
        d = np.where(np.isnan(d), h - poser.g[:, 3], d)
    off = np.isfinite(d) & (np.abs(d) > level_tol)
    poser.off_level = off
    d[off] = np.nan
    ok = np.isfinite(d)
    if ok.sum() < 10:
        return dict(snapped=False, off_level=int(off.sum()))
    d = np.interp(poser.g[:, 0], poser.g[ok, 0], d[ok])
    n = max(1, int(round(win_s / np.median(np.diff(poser.g[:, 0]) * 1e-9))) | 1)
    ds = median_filter(d, n, mode="nearest")
    poser.g[:, 3] += ds
    return dict(
        snapped=True,
        dz_median_cm=round(float(np.median(ds[ok])) * 100, 1),
        dz_p05_cm=round(float(np.percentile(ds[ok], 5)) * 100, 1),
        dz_p95_cm=round(float(np.percentile(ds[ok], 95)) * 100, 1),
        covered=round(float(ok.mean()), 3),
        off_level=int(off.sum()),
    )


def loader(paths_seg, paths_img=None, workers=8):
    """yields (seg, rgb or None) in order, decoded in a thread pool."""
    rd = lambda i: (
        cv2.imread(paths_seg[i], cv2.IMREAD_GRAYSCALE),
        cv2.imread(paths_img[i])[..., ::-1].copy() if paths_img is not None else None,
    )
    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(rd, range(len(paths_seg)))


def load_arrows(labels_dir: str) -> list:
    """arrows.json of a label map (steps/arrows.py); [] if there is none."""
    f = os.path.join(labels_dir, "arrows.json")
    return json.load(open(f))["arrows"] if os.path.exists(f) else []


def arrow_cls(cls: np.ndarray, arrows: list, lo, res: float, origin, margin: float = 0.3) -> np.ndarray:
    """label map with clean arrows: inside each arrow outline (all pieces) + margin, 'marking' cells (the blobby sliver of the
    segmentation) become 'road', then the outline (measured in sharp single views) is filled as 'marking'."""
    if not arrows:
        return cls
    from shapely.geometry import Polygon

    mk, road = CLASSES.index("marking"), CLASSES.index("road")
    out = cls.copy()
    foot = np.zeros(cls.shape, np.uint8)
    paint = np.zeros(cls.shape, np.uint8)
    ij = lambda xy: np.round((np.asarray(xy)[:, :2] - np.asarray(origin)[:2] - lo) / res - 0.5).astype(np.int32)[
        :, ::-1
    ]  # (col = y, row = x)
    for a in arrows:
        for ring in (np.array([q[:2] for q in o], float) for o in a["outlines"]):
            cv2.fillPoly(foot, [ij(np.array(Polygon(ring).buffer(margin).exterior.coords))], 1)
            # shafts are ~0.15 m wide: at 0.1 m cells a slanted shaft would break up, so the outline is widened by
            # 5 cm and its edge drawn too (keeps the arrow one connected shape, like the lane lines of the label map)
            q = ij(np.array(Polygon(ring).buffer(0.05).exterior.coords))
            cv2.fillPoly(paint, [q], 1)
            cv2.polylines(paint, [q], True, 1, 1)
    out[(foot > 0) & (out == mk)] = road
    out[(paint > 0) & (out > 0)] = mk
    return out
