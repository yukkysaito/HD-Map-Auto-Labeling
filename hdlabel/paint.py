"""Paint refinement of the marking class: the segmentation outlines small markings as blobs (text and symbols turn
into rectangles, hatched areas into solid triangles), the fused ortho image shows the paint itself.

Within reach of the segmentation's marking cells, a cell is marking when it is brighter than the asphalt around it
(white top-hat of the brightest colour channel: white and yellow paint), else road. A marking blob whose cells show
little paint contrast (blurred or far views) is kept as the segmentation drew it."""

from __future__ import annotations

import cv2
import numpy as np

from hdlabel.common import CLASSES

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")
DEFAULTS = dict(contrast=18.0, window_m=1.5, grow_m=0.4, min_share=0.1)


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
    n, comp = cv2.connectedComponents(zone.astype(np.uint8), connectivity=8)
    n_seg = np.bincount(comp[seg], minlength=n)
    n_paint = np.bincount(comp[zone & paint & seg], minlength=n)
    ok = n_paint >= p["min_share"] * np.maximum(n_seg, 1)
    ok[0] = False
    use = ok[comp] & zone
    out = cls.copy()
    out[use & paint] = MARKING
    out[use & ~paint & seg] = ROAD
    st = dict(blobs=int(n - 1), refined=int(ok.sum()), marking_before=int(seg.sum()),
              marking_after=int((out == MARKING).sum()))  # fmt: skip
    return out, st
