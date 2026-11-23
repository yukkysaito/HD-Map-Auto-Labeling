"""Evaluation: how well the marking class of a label raster follows the road paint in the fused ortho image.

Independent of the paint refinement (hdlabel/paint.py, a top-hat threshold): the scores use image edges and local
intensity statistics, not a paint mask. On the ortho brightness g (brightest colour channel: white and yellow paint):
  - boundary F-score: label boundary (marking cells with a non-marking 4-neighbour) vs Canny edges of g, within tol
    cells, in the road / marking area within reach_m of the marking (curb and unobserved borders excluded). A blob
    drawn around a text or a hatched area has no image edge on its outline (low precision) and leaves the stroke
    edges inside it unmatched (low recall).
  - per piece (a marking blob, 8-connected, cut into piece_m tiles so that the levels stay local) against its ring
    (road cells within ring_m, each taken by the nearest marking cell): contrast = mean g of the piece minus the ring
    median. With the split level halfway between the ring median and the piece's 90th percentile (its paint):
    dark = share of marking cells below it (asphalt labelled marking), missed = share of the cells above it in
    marking + ring that are road (paint labelled road), paint_iou = IoU of the marking with the cells above it. Pieces whose paint does
    not stand out by min_contrast (blurred views) count as low_contrast and are left out of these three.
Each score is given for all marking cells, for "line" cells (thin parts: lane lines, stop lines) and for "area" cells
(text, symbols, arrows, hatched areas: parts thicker than line_width_m after a closing over close_m, which merges
letters and hatch stripes into one area whether the label draws them as strokes or as a blob). Every cell takes
the group of the nearest marking cell of the reference labelling (--ref; default: the labels themselves). Give two
labellings of a scene the same --ref so that both are split into the same lines and areas.

Usage: python -m hdlabel.eval.paint_fidelity ground_map.npz ortho.jpg [--ref ground_map.npz] [--out scores.json]
(ortho.jpg = np.rot90 of the label grid, as project_labels.py --debug writes it). Compare two labellings on the same
ortho image: a sharper ortho alone changes the scores.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from scipy import ndimage

from hdlabel.common import CLASSES

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")
DEFAULTS = dict(
    tol=1, reach_m=1.0, ring_m=0.5, piece_m=4.0, close_m=2.0, line_width_m=0.7, canny=(40, 80), min_contrast=10.0
)


def _disc(d: int):
    d = max(3, d | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def area_mask(seg, res: float, close_m: float, line_width_m: float):
    """bool mask of the marking cells in areas (text, symbols, hatched areas): the closing of the marking over close_m
    opened by a disc of line_width_m (removes the thin lines) and grown by that disc again (corners); the other
    marking cells are lines."""
    closed = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_CLOSE, _disc(int(round(close_m / res))))
    disc = _disc(int(round(line_width_m / res)))
    return cv2.dilate(cv2.morphologyEx(closed, cv2.MORPH_OPEN, disc), disc).astype(bool) & seg


def _r(x: float, nd: int = 4):
    return round(float(x), nd) if np.isfinite(x) else None


def score(cls, rgb, res: float, ref=None, **kw) -> dict:
    """cls (class raster, common.CLASSES ids), rgb (ortho on the same grid, HxWx3), ref (class raster that defines
    the line / area groups; default cls) -> scores (see module doc)."""
    p = dict(DEFAULTS, **kw)
    g = rgb.astype(np.float32).max(2)
    seg = cls == MARKING
    out = dict(res=res, params={k: list(v) if isinstance(v, tuple) else v for k, v in p.items()})
    if not seg.any():
        return dict(out, all=None, line=None, area=None)
    d_mark, (ii, jj) = ndimage.distance_transform_edt(~seg, return_indices=True)
    rseg = seg if ref is None else ref == MARKING
    if not rseg.any():
        rseg = seg
    area = area_mask(rseg, res, p["close_m"], p["line_width_m"])
    ri, rj = (ii, jj) if rseg is seg else ndimage.distance_transform_edt(~rseg, return_indices=True)[1]
    near_area = area[ri, rj]  # group of every cell: that of the nearest reference marking cell

    # boundary vs Canny edges of the ortho
    surface = (cls == ROAD) | seg
    inner = cv2.erode(surface.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # away from curbs, holes
    zone = inner & (d_mark * res <= p["reach_m"])
    blur = cv2.GaussianBlur(np.clip(g, 0, 255).astype(np.uint8), (0, 0), 1.0)
    edge = (cv2.Canny(blur, *p["canny"]) > 0) & zone
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    bnd = seg & (cv2.erode(seg.astype(np.uint8), cross, borderType=cv2.BORDER_REPLICATE) == 0) & zone
    hit_b = ndimage.distance_transform_edt(~edge) <= p["tol"] if edge.any() else np.zeros_like(seg)
    hit_e = ndimage.distance_transform_edt(~bnd) <= p["tol"] if bnd.any() else np.zeros_like(seg)

    # pieces: blobs cut into tiles; ring cells belong to the piece of their nearest marking cell
    _, blob = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    t = max(1, int(round(p["piece_m"] / res)))
    r, c = np.indices(seg.shape)
    nr, nc = seg.shape[0] // t + 1, seg.shape[1] // t + 1
    key = blob.astype(np.int64) * (nr * nc) + (r // t) * nc + c // t
    u, piece = np.unique(np.where(seg, key + 1, 0), return_inverse=True)
    piece = piece.reshape(seg.shape) + int(u[0] != 0)  # 0: not marking
    n = int(piece.max()) + 1
    ring = (cls == ROAD) & (d_mark * res <= p["ring_m"])
    ring_id = np.where(ring, piece[ii, jj], 0)
    idx = np.arange(1, n)
    has_ring = np.bincount(ring_id.ravel(), minlength=n)[1:] > 0
    ring_med = np.full(n, np.nan)
    if has_ring.any():
        ring_med[idx[has_ring]] = ndimage.median(g, ring_id, idx[has_ring])
    mean = np.concatenate([[np.nan], ndimage.mean(g, piece, idx)])
    p90 = np.concatenate([[np.nan], ndimage.labeled_comprehension(g, piece, idx, _p90, float, np.nan)])
    good = (p90 - ring_med) >= p["min_contrast"]  # NaN (no ring) -> False
    thr = np.where(good, (p90 + ring_med) / 2, np.nan)
    contrast = mean - ring_med

    res_out = {}
    for name, sel in (("all", np.ones_like(seg)), ("line", ~near_area), ("area", near_area)):
        m, b, e = seg & sel, bnd & sel, edge & sel
        prec = hit_b[b].mean() if b.any() else np.nan
        rec = hit_e[e].mean() if e.any() else np.nan
        pm = piece[m]
        ok = good[pm]
        above_m = g[m][ok] >= thr[pm[ok]]
        rm = ring & sel & good[ring_id]
        above_r = g[rm] >= thr[ring_id[rm]]
        cm = contrast[pm]
        union = above_m.size + int(above_r.sum())  # marking (contrasted pieces) + paint labelled road
        res_out[name] = dict(
            cells=int(m.sum()),
            pieces=int(np.unique(pm).size),
            low_contrast_pieces=int(np.unique(pm[~ok]).size),
            boundary_precision=_r(prec),
            boundary_recall=_r(rec),
            boundary_f=_r(2 * prec * rec / (prec + rec)) if prec + rec > 0 else None,
            contrast=_r(np.nanmean(cm), 2) if np.isfinite(cm).any() else None,
            dark=_r(1 - above_m.mean()) if above_m.size else None,
            missed=_r(above_r.sum() / (above_m.sum() + above_r.sum())) if above_m.any() or above_r.any() else None,
            paint_iou=_r(above_m.sum() / union) if union else None,
        )
    return dict(out, **res_out)


def _p90(x):
    return np.percentile(x, 90)


def load(npz: str, ortho: str):
    """(cls, rgb on the label grid, res) from a ground_map.npz and its ortho.jpg (np.rot90 of the grid)."""
    z = np.load(npz)
    img = cv2.imread(ortho)
    if img is None:
        raise SystemExit(f"cannot read {ortho}")
    rgb = np.ascontiguousarray(np.rot90(img[..., ::-1], -1))
    if rgb.shape[:2] != z["cls"].shape:
        raise SystemExit(f"ortho {img.shape[:2]} does not match the label grid {z['cls'].shape} (rot90)")
    return z["cls"], rgb, float(z["res"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", help="ground_map.npz")
    ap.add_argument("ortho", help="ortho.jpg of the same grid")
    ap.add_argument("--ref", help="ground_map.npz whose marking defines the line / area groups (default: labels)")
    ap.add_argument("--out", help="write the scores to this JSON file")
    ap.add_argument("--tol", type=int, default=DEFAULTS["tol"], help="boundary tolerance in cells")
    a = ap.parse_args(argv)
    cls, rgb, res = load(a.labels, a.ortho)
    ref = None
    if a.ref:
        ref = np.load(a.ref)["cls"]
        if ref.shape != cls.shape:
            raise SystemExit(f"--ref grid {ref.shape} differs from the labels {cls.shape}")
    s = dict(labels=a.labels, ortho=a.ortho, ref=a.ref, **score(cls, rgb, res, ref=ref, tol=a.tol))
    txt = json.dumps(s, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
