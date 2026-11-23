"""Paint refinement of the marking and crosswalk classes: the segmentation outlines small markings as blobs (text
and symbols turn into rectangles, hatched areas into solid triangles) and crosswalks as solid areas (stripes and the
gaps between them), the fused ortho image shows the paint itself.

refine: within reach of the segmentation's marking cells, a cell is marking when it is brighter than the asphalt
around it (white top-hat of the brightest colour channel: white and yellow paint), else road. Where the segmented
marking within share_m shows little paint contrast (blurred or far views, worn paint) it is kept as the segmentation
drew it - decided locally, so one long line refines where it is sharp and stays where it is not.

refine_crosswalk: inside a segmented crosswalk area, a cell stays crosswalk when it is brighter than its
surroundings (white top-hat of the darkest colour channel: white stripes stand out on grey and on red pavement
alike), the unpainted gaps between the stripes become road. An area whose cells show little paint (blurred, far or
worn) is kept whole.

Only the raster for training uses the results (ground_map.npz `cls_paint`); the vectors keep the segmentation's
shapes, whose stroke widths tell lines, stop lines and areas apart (text and hatched areas stay marking_area,
crosswalks whole areas)."""

from __future__ import annotations

import cv2
import numpy as np

from hdlabel.common import CLASSES

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")
CROSSWALK = CLASSES.index("crosswalk")
DEFAULTS = dict(contrast=18.0, window_m=1.0, grow_m=0.4, min_share=0.1, share_m=2.0)
XWALK_DEFAULTS = dict(contrast=30.0, rel=0.5, window_m=1.5, min_share=0.25, hole_m2=0.2, min_zone_m2=3.0)


def tophat(rgb, observed, res: float, window_m: float, channel=np.max):
    """how much brighter (0..255) each cell is than the background, the grey opening over a disc of window_m (removes
    bright structures narrower than it: lines, letters, stripes). channel: how the colour channels are combined
    (np.max: white and yellow paint; np.min: white paint only, also on coloured pavement)."""
    g = channel(rgb.astype(np.float32), axis=2)
    g[~observed] = np.median(g[observed]) if observed.any() else 0
    w = max(3, int(round(window_m / res)) | 1)
    return g - cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)))


def paint_mask(rgb, observed, res: float, contrast: float, window_m: float):
    """bool mask of the cells brighter by more than `contrast` (0..255) than the background (tophat)."""
    m = (tophat(rgb, observed, res, window_m) > contrast) & observed
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


def refine_crosswalk(cls, rgb, res: float, **kw):
    """cls, rgb (same grid) -> (refined cls, stats). Painted: brighter than the background by > contrast and by
    > rel x the brightest cell within window_m (the stripes there: robust to blur and to low contrast on coloured
    pavement), smoothed by a 3-cell closing and opening (ragged edges); unpainted holes < hole_m2 (dirt, worn spots in
    a stripe) are filled. Where >= min_share of the crosswalk cells within window_m are painted, in a connected zone of
    >= min_zone_m2, the unpainted crosswalk cells become road (the gaps between the stripes); elsewhere (blurred, far
    or worn parts) the area is kept as segmented. Other classes are untouched."""
    p = dict(XWALK_DEFAULTS, **kw)
    seg = cls == CROSSWALK
    th = tophat(rgb, cls > 0, res, p["window_m"], channel=np.min)
    w = max(3, int(round(p["window_m"] / res)) | 1)
    peak = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)))  # the stripes' contrast nearby
    paint = ((th > p["contrast"]) & (th > p["rel"] * peak)).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    paint = cv2.morphologyEx(cv2.morphologyEx(paint, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k).astype(bool) & seg
    n, comp, st, _ = cv2.connectedComponentsWithStats((seg & ~paint).astype(np.uint8), connectivity=4)
    paint |= (st[:, cv2.CC_STAT_AREA] * res**2 < p["hole_m2"])[comp] & (comp > 0)
    share = cv2.boxFilter(paint.astype(np.float32), -1, (w, w), normalize=False) / np.maximum(
        cv2.boxFilter(seg.astype(np.float32), -1, (w, w), normalize=False), 1
    )
    zone = (seg & (share >= p["min_share"])).astype(np.uint8)  # where the stripes are seen
    n, comp, st, _ = cv2.connectedComponentsWithStats(zone, connectivity=8)
    gap = (st[:, cv2.CC_STAT_AREA] * res**2 >= p["min_zone_m2"])[comp] & zone.astype(bool) & ~paint
    out = cls.copy()
    out[gap] = ROAD
    st = dict(crosswalk_before=int(seg.sum()), to_road=int(gap.sum()))
    return out, st
