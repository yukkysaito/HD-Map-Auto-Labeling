"""Arrow shape from a sharp ground patch (steps/arrows.py): paint segmentation and template matching.

Patch frame: rows = +u (along the lane axis), cols = +v (driver's left of +u), RES m per pixel. Templates
(assets/arrow_templates.npz, cut from real arrows at 2 cm): tip at the last row, driver's left at the last column; keys
class[_variant] with class S (straight), R (right), SL (straight + left). Mirroring gives L / SR, reversing the tip
direction gives arrows pointing to -u.
"""
import os

import cv2
import numpy as np

RES = 0.02


def nan_median(x):
    """median over axis 0 ignoring NaN (= np.nanmedian, much faster for a few images); all-NaN -> 0."""
    k = np.isfinite(x).sum(0); s = np.sort(x, 0)                       # NaN sort last
    lo_, hi_ = np.maximum((k - 1) // 2, 0)[None], np.maximum(k // 2, 0)[None]
    m = (np.take_along_axis(s, lo_, 0)[0] + np.take_along_axis(s, hi_, 0)[0]) / 2
    return np.where(k > 0, m, 0.0)


def paint_masks(med, r=RES):
    """paint (brighter than the local road) -> list of arrow-candidate masks: first the paint through the patch centre,
    then other compact paint in the patch with a head (e.g. a second arrow 1 m beside it: their blobs merge in the label
    map, so it gets no candidate of its own). [] if there is no paint at the centre."""
    g = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY); valid = med.sum(-1) > 0
    if valid.mean() < 0.5: return []
    d = g.astype(np.float32) - cv2.medianBlur(g, 101).astype(np.float32)
    strip = valid.copy(); strip[:, :g.shape[1] // 2 - int(0.8 / r)] = False; strip[:, g.shape[1] // 2 + int(0.8 / r):] = False
    if strip.sum() < 100: return []
    # threshold from the central strip only: bright zebra zones / crosswalks beside the arrow would raise it
    # arrows are white paint: saturated pixels (red / green coloured pavement, which often borders an arrow) are not
    # paint; the border of the rendered area (no view beyond) looks brighter than its local median: not paint either
    sat = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[..., 1] < 70
    inner = cv2.erode(valid.astype(np.uint8), np.ones((int(0.3 / r) | 1,) * 2, np.uint8)) > 0
    pt = ((d > max(14.0, 0.45 * float(np.percentile(d[strip], 99)))) & inner & sat).astype(np.uint8)
    pt = cv2.morphologyEx(cv2.morphologyEx(pt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labm, st, _ = cv2.connectedComponentsWithStats(pt, connectivity=8)
    nu, nv = g.shape; cu, cv_ = nu // 2, nv // 2; wu, wv = int(1.2 / r), int(0.3 / r)
    cen = labm[cu - wu:cu + wu, cv_ - wv:cv_ + wv]
    ids = [i for i in np.unique(cen) if i and st[i, cv2.CC_STAT_AREA] * r * r > 0.08]
    if not ids: return []

    def grow(k):
        # pieces of the same arrow that are not connected (worn paint, or dashed shafts: the branch head of a
        # straight+left arrow can sit 1-2.5 m behind the head): components within 0.9 m of the axis that are inside the
        # length range of the seed, or within 2.5 m of it and wider than 0.3 m (a head, not a dash of the shaft / tail)
        m = labm == k; iu, iv = np.nonzero(m); u0, u1, vc = iu.min(), iu.max(), np.median(iv)
        for i in range(1, n):
            if i == k or st[i, cv2.CC_STAT_AREA] * r * r < 0.03: continue
            x, y, w, h = st[i, :4]                                      # x = col (v), y = row (u)
            if max(abs(x - vc), abs(x + w - vc)) * r >= 0.9: continue
            inside = y >= u0 - 5 and y + h <= u1 + 5
            near_head = w * r >= 0.3 and st[i, cv2.CC_STAT_AREA] * r * r >= 0.05 and max(y - u1, u0 - (y + h)) * r <= 2.5
            if inside or near_head: m |= labm == i
        return m
    out = [grow(max(ids, key=lambda i: st[i, cv2.CC_STAT_AREA]))]; used = out[0].copy()
    for i in range(1, n):                                               # other arrows in the patch
        x, y, w, h = st[i, :4]
        if used[labm == i].any() or st[i, cv2.CC_STAT_AREA] * r * r < 0.25 or w * r < 0.3 or w * r > 1.2: continue
        if y < 3 or y + h > nu - 3 or abs(x + w / 2 - cv_) * r > 1.7: continue
        m = grow(i); used |= m; out.append(m)
    return out


def uv_of(mask, r=RES):
    nu, nv = mask.shape; iu, iv = np.nonzero(mask); return (iu - nu / 2 + 0.5) * r, (iv - nv / 2 + 0.5) * r


def axis_fit(mask, r=RES):
    """shaft line v = a u + b of the mask (patch frame) -> angle (rad), centre (u, v) of the component."""
    u, v = uv_of(mask, r); sh = np.abs(v - np.median(v)) < 0.2
    if sh.sum() < 20: return 0.0, (0.0, 0.0)
    A = np.polyfit(u[sh], v[sh], 1); uc = (u.min() + u.max()) / 2; return float(np.arctan(A[0])), (uc, float(A[0] * uc + A[1]))


TPL = np.load(os.path.join(os.path.dirname(__file__), "assets", "arrow_templates.npz"))
TEMPLATES = {k: TPL[k].astype(np.float32) for k in TPL.files if k not in ("res", "source", "note")}  # key = class[_variant];
# tip at the last row, +v (driver's left) at the last col
NAMES = {"S": ["straight"], "R": ["right"], "L": ["left"], "SL": ["straight", "left"], "SR": ["straight", "right"]}


def off_shaft(m, half=6):
    o = m.copy(); c = int(np.median(np.nonzero(m)[1])); o[:, max(0, c - half):c + half + 1] = 0; return o


def _half(m): return (cv2.resize(m, (max(1, m.shape[1] // 2), max(1, m.shape[0] // 2)), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.float32)


def _variants():
    """every template x length scale (0.6-1.2) x mirror (left <-> right) x reverse (tip at -u):
    (class, scale, reversed, (t, t off the shaft) at full and at half resolution)."""
    out = []
    for name, t in TEMPLATES.items():
        cl = name.split("_")[0]
        for sy in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            ts = cv2.resize(t, (t.shape[1], max(5, int(t.shape[0] * sy))), interpolation=cv2.INTER_NEAREST)
            for tv, rev, mir in ((ts, False, False), (ts[:, ::-1], False, True), (ts[::-1, ::-1], True, False), (ts[::-1], True, True)):
                tv = np.ascontiguousarray(tv); th = _half(tv)
                out.append((cl.translate(str.maketrans("LR", "RL")) if mir else cl, sy, rev, (tv, off_shaft(tv)), (th, off_shaft(th, 3))))
    return out


VARIANTS = _variants()


def _match(M, Mo, t, to):
    """max over positions of Dice(paint) x Dice(paint off the shaft) (cross-correlation of binary masks)."""
    if t.shape[0] >= M.shape[0] or t.shape[1] >= M.shape[1]: return -1.0
    d = (2 * cv2.matchTemplate(M, t, cv2.TM_CCORR) / (t.sum() + M.sum())) * (2 * cv2.matchTemplate(Mo, to, cv2.TM_CCORR) / (to.sum() + Mo.sum() + 1e-6))
    return float(d.max())


def shape(mask, r=RES, min_score=0.5, refine=16):
    """template match in the aligned patch frame (rows = +u, cols = +v) over the templates, their mirror image (left <->
    right), their reverse (tip at -u) and the length scale. score = Dice(all paint) x Dice(paint off the shaft), so
    plain lines (no head / branch) score ~0. All variants are scored at half resolution, the best `refine` again at full
    resolution (the half-resolution score alone confuses e.g. a reversed right arrow with a straight one).
    -> dict (directions, sign: +1 tip at +u, -1 tip at -u, score, ...)"""
    M = mask.astype(np.float32); Mo = off_shaft(M); Mh = _half(M); Mho = off_shaft(Mh, 3)
    coarse = sorted(range(len(VARIANTS)), key=lambda i: -_match(Mh, Mho, *VARIANTS[i][4]))[:refine]
    best = (0.0, None)
    for i in coarse:
        cl, sy, rev, full, _ = VARIANTS[i]; sc = _match(M, Mo, *full)
        if sc > best[0]: best = (sc, (cl, sy, rev))
    score, b = best
    u, v = uv_of(mask, r); u0, u1 = float(u.min()), float(u.max()); L = u1 - u0
    if b is None: return dict(ok=False, directions=[], score=0.0, length=round(L, 2))
    # the tip is pointed: within 0.15 m of the tip end the paint is much narrower than the widest part (stripes of
    # crosswalks / zebra zones are flat-ended rectangles and can otherwise match a head)
    ut = (u > u1 - 0.15) if not b[2] else (u < u0 + 0.15)
    rows = np.round(u / r).astype(int); wmax = max(np.ptp(v[rows == q]) for q in np.unique(rows)) + r
    wtip = max(np.ptp(v[ut & (rows == q)]) for q in np.unique(rows[ut])) + r if ut.any() else 0.0
    pointed = wtip / wmax < 0.6
    return dict(ok=bool(score >= min_score and 1.8 < L < 7.5 and pointed), pointed=round(float(wtip / wmax), 2), directions=NAMES[b[0]], shape=b[0], scale=b[1], sign=-1 if b[2] else 1,
                score=round(score, 3), u0=u0, u1=u1, v=float(np.median(v[np.abs(v - np.median(v)) < 0.2])), length=round(L, 2), width=round(float(np.ptp(v)), 2), area=round(float(len(u) * r * r), 2))
