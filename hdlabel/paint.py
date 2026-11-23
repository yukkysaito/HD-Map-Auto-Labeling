"""Paint refinement of the marking class: the segmentation outlines small markings as blobs (text and symbols turn
into rectangles, hatched areas into solid triangles), the fused ortho image shows the paint itself.

Within reach of the segmentation's marking cells, a cell is marking when it is brighter than the asphalt around it
(white top-hat of the brightest colour channel: white and yellow paint), else road. Where the segmented marking within
share_m shows little paint contrast (blurred or far views, worn paint) it is kept as the segmentation drew it - decided
locally, so one long line refines where it is sharp and stays where it is not.

Only the raster for training uses the result (ground_map.npz `cls_paint`); the vectors keep the segmentation's shapes,
whose stroke widths tell lines, stop lines and areas apart (text and hatched areas stay marking_area)."""

from __future__ import annotations

import cv2
import numpy as np

from hdlabel.common import CLASSES

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")
DEFAULTS = dict(contrast=18.0, window_m=1.0, grow_m=0.4, min_share=0.1, share_m=2.0)


def paint_mask(rgb, observed, res: float, contrast: float, window_m: float):
    """bool mask of the cells brighter by more than `contrast` (0..255) than the background, the grey opening over a
    disc of window_m (removes bright structures narrower than it: lines, letters, stripes)."""
    g = rgb.astype(np.float32).max(2)
    g[~observed] = np.median(g[observed]) if observed.any() else 0
    w = max(3, int(round(window_m / res)) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)))
    m = ((g - bg) > contrast) & observed
    n, comp, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return (st[:, cv2.CC_STAT_AREA] >= 4)[comp] & m  # specks of 1-3 cells are noise


def refine(cls, rgb, res: float, **kw):
    """cls (class raster, common.CLASSES ids), rgb (same grid) -> (refined cls, stats). Only road / marking cells
    near segmented marking change; other classes and crosswalks are untouched."""
    p = dict(DEFAULTS, **kw)
    paint = paint_mask(rgb, cls > 0, res, p["contrast"], p["window_m"])
    seg = cls == MARKING
    g = max(1, int(round(p["grow_m"] / res)))
    zone = cv2.dilate(seg.astype(np.uint8), np.ones((2 * g + 1, 2 * g + 1), np.uint8)).astype(bool)
    zone &= (cls == ROAD) | seg
    k = max(3, int(round(p["share_m"] / res)) | 1)  # local share of the segmented marking that shows paint
    n_seg = cv2.boxFilter(seg.astype(np.float32), -1, (k, k), normalize=False)
    n_paint = cv2.boxFilter((seg & paint).astype(np.float32), -1, (k, k), normalize=False)
    use = zone & (n_seg > 0) & (n_paint >= p["min_share"] * n_seg)
    out = cls.copy()
    out[use & paint] = MARKING
    out[use & ~paint & seg] = ROAD
    st = dict(refined_cells=int(use.sum()), kept_cells=int((zone & ~use).sum()), marking_before=int(seg.sum()),
              marking_after=int((out == MARKING).sum()))  # fmt: skip
    return out, st
