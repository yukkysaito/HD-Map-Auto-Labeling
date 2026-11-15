"""Road arrows (lane direction markings) of a scene from sharp single camera frames, without any HD map.

Why: the segmentation (Vistas "Lane Marking - General", images at half resolution) gives blobby masks, so in the fused
label map an arrow is a thin sliver without its head, and vectorize.py turned it into a short lane line. The direction
(straight / left / right) is lost, which a planner needs (e.g. right-turn-only lanes).

How (per scene):
 1. candidates: compact marking components of the label map (ground_map.npz), 1-7.5 m long, on the road (wide
    painted areas cut away first: an arrow touching a zone would otherwise be a spur of one big component)
 2. for each, a ground patch (8 x 3.8 m, 2 cm, aligned with the lane = nearest trajectory heading) is rendered from the
    closest camera views (2.5-11 m, every camera of every drive of the scene; cameras with a failed time offset are
    skipped): each view is warped onto the map surface and the patch is the median of the 5 closest views (removes
    cars / occlusions); the axis is re-fitted on the paint and the patch rendered again
 3. paint = brighter than the local road (median over 2 m); the component through the patch centre is the candidate
 4. template match (hdlabel/assets/arrow_templates.npz: straight, right, straight+left cut from real arrows; mirrored
    for left / straight+right, reversed for the tip direction, scaled 0.6-1.2 in length): score = Dice of the paint x
    Dice of the paint off the shaft (heads, branches; plain lines score ~0); accepted at >= --min-score
 5. other arrows in the same patch (an arrow 1 m beside another merges with it in the label map) become candidates
    of their own (patch centred on them);
    duplicates (several candidates of one arrow: middle within 0.5 m of a better one's axis) are merged; arrows on
    crosswalk cells of the label map are dropped (stripes)
Writes OUT/arrows.json: [{id, tail, tip (map xyz), directions, shape, score, length, width, outlines (paint outline rings,
map xyz; several for dashed shafts / detached branches), views, auto}] and
DEBUG/arrows.jpg (review sheet: patch, paint mask and the result per arrow, tip up). Consumers: mapproj.apply_arrows
(clean raster: the sliver is replaced by the outline), vectorize.py (arrow elements, no lane-line pieces under arrows).
Arrows the detection missed can be added in review ({"kind": "arrow", "add": {"tail", "tip", "directions"}}, --fixes).
usage: arrows.py --labels OUT_OF_project_labels --drive DATA CALIB [--drive ...] [--fixes review/<name>/*.json]
"""
import argparse, json, os, sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import PoseInterp
from hdlabel.arrowshape import axis_fit, nan_median, paint_masks, shape, uv_of
from hdlabel.mapproj import Camera, Projector, snap_to_ground

p = argparse.ArgumentParser()
p.add_argument("--labels", required=True, help="dir with ground_map.npz; arrows.json / arrows.jpg are written there")
p.add_argument("--drive", nargs=2, action="append", required=True, metavar=("DATA", "CALIB"))
p.add_argument("--min-score", type=float, default=0.5, help="template match score to accept")
p.add_argument("--res", type=float, default=0.02); p.add_argument("--views", type=int, default=5)
p.add_argument("--debug", help="dir for arrows.jpg (review sheet)")
p.add_argument("--debug-candidates", action="store_true", help="also DEBUG/arrows_debug.jpg + arrow_cand_<n>.npz: every candidate patch")
p.add_argument("--fixes", nargs="*", default=[], help="review json files: {\"kind\": \"arrow\", \"add\": {\"tail\": [x, y], \"tip\": [x, y], "
               "\"directions\": [...]}} adds an arrow the detection missed (measured in the sharp patch like the others)")
p.add_argument("--range", type=float, nargs=2, default=[2.5, 11.0], help="camera distance of the views (m)")
a = p.parse_args()

lab = f"{a.labels}/ground_map.npz"; m = np.load(lab); cls = m["cls"]; lo = m["lo"]; gres = float(m["res"])
G = [str(g) for g in m["groups"]]; MK = G.index("marking") + 1; ROAD = G.index("road") + 1
pj = Projector(lab); org = pj.origin; r = a.res

# ---- views: every camera of every drive (poses relative to the label map origin)
views, traj = [], []
for data, calib in a.drive:
    poser = PoseInterp(f"{data}/gt.npy"); snap_to_ground(poser, lab); traj.append(poser.g[~poser.off_level, 1:3] - org[:2])
    for f in sorted(os.listdir(calib)):
        if not (f.startswith("cam") and f.endswith(".json")): continue
        c = int(f[3:-5])
        if json.load(open(f"{calib}/{f}")).get("offset_status") == "failed" or not os.path.isdir(f"{data}/kf_cam{c}"): continue
        cam = Camera(data, calib, c, poser); T = cam.T_map_cam(np.arange(len(cam.st))); T[:, :3, 3] -= org
        views.append((cam, np.linalg.inv(T)))
traj = np.concatenate(traj); hd = np.gradient(traj, axis=0); ttree = cKDTree(traj)


def heading(c):
    d, i = ttree.query(c)
    if d > 25: return None
    v = hd[max(0, i - 5):i + 6].sum(0); return v / (np.linalg.norm(v) + 1e-9)


def patch(c, e, hu=4.0, hv=1.9):
    """median of the closest views of the ground rectangle c +- hu e +- hv n -> (nu, nv, 3) uint8 (rows: u, cols: v)."""
    nu, nv = int(2 * hu / r), int(2 * hv / r); n_ = np.array([-e[1], e[0]])
    U, V = np.meshgrid((np.arange(nu) - nu / 2 + 0.5) * r, (np.arange(nv) - nv / 2 + 0.5) * r, indexing="ij")
    P = c + U[..., None] * e + V[..., None] * n_
    z = pj.height_at(torch.tensor(P.reshape(-1, 2), device=pj.dev)).cpu().numpy()
    if (z < -1e3).mean() > 0.3: return None, 0
    Pm = np.c_[P.reshape(-1, 2), z, np.ones(z.size)].T; zc = float(np.median(z[z > -1e3]))
    cand = []
    for vi, (cam, Ti) in enumerate(views):
        pc = Ti[:, :3, :3] @ np.r_[c, zc] + Ti[:, :3, 3]; d = np.linalg.norm(pc, axis=1)
        W2, H2 = 2 * cam.K[0, 2], 2 * cam.K[1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = cam.K[0, 0] * pc[:, 0] / pc[:, 2] + cam.K[0, 2]; vv = cam.K[1, 1] * pc[:, 1] / pc[:, 2] + cam.K[1, 2]
        ok = cam.on_level & (pc[:, 2] > 1) & (d > a.range[0]) & (d < a.range[1]) & (uu > 0.1 * W2) & (uu < 0.9 * W2) & (vv > 0.3 * H2) & (vv < 0.97 * H2)
        cand += [(d[i], vi, i) for i in np.flatnonzero(ok)]
    cand.sort(); used, ims = [], []
    for _, vi, i in cand:
        if any(vi == u and abs(i - j) < 3 for u, j in used): continue    # not consecutive frames of one camera
        cam, Ti = views[vi]; q = Ti[i] @ Pm
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = (cam.K[0, 0] * q[0] / q[2] + cam.K[0, 2]).reshape(nu, nv).astype(np.float32)
            mv = (cam.K[1, 1] * q[1] / q[2] + cam.K[1, 2]).reshape(nu, nv).astype(np.float32)
        im = cv2.remap(cv2.imread(cam.files[i]), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        im[((q[2] < 0.5) | (z < -1e3)).reshape(nu, nv) | (im.sum(-1) == 0)] = np.nan
        ims.append(im); used.append((vi, i))
        if len(ims) >= a.views: break
    if not ims: return None, 0
    med = nan_median(np.stack(ims))
    return np.nan_to_num(med).astype(np.uint8), len(ims)


# ---- candidates: compact marking components on the road
mk = (cls == MK).astype(np.uint8)
# arrows often touch wide painted areas (coloured / hatched zones, stop bars, crosswalk-like pavement labelled as marking):
# then the arrow is a thin spur of one big component. Wide parts (> 0.7 m across) are cut away (+ 0.4 m), so the spurs
# become candidates of their own.
wide = cv2.dilate((cv2.distanceTransform(mk, cv2.DIST_L2, 5) * gres > 0.35).astype(np.uint8),
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.75 / gres) + 1,) * 2))
mk[wide > 0] = 0
n, labm, st, cen = cv2.connectedComponentsWithStats(mk, connectivity=8)
cands = []
for k in range(1, n):
    if st[k, cv2.CC_STAT_AREA] * gres ** 2 < 0.1: continue
    x0, y0, w, h = st[k, :4]
    if max(w, h) * gres > 12: continue                               # long lines
    ii, jj = np.nonzero(labm[y0:y0 + h, x0:x0 + w] == k); xy = np.c_[ii + y0, jj + x0] * gres + lo   # rows = x, cols = y
    c = xy.mean(0); ev = np.linalg.eigh(np.cov((xy - c).T))[1][:, -1] if len(xy) > 2 else np.array([1.0, 0.0])
    L = np.ptp((xy - c) @ ev)
    if 1.0 < L < 7.5: cands.append((c, ev))

# arrows added in review (missed by the detection): measured like the others, the reviewer's axis / directions win
adds = []
for fx in (fx for f_ in a.fixes if os.path.exists(f_) for fx in json.load(open(f_)).get("fixes", []) if fx.get("kind") == "arrow" and "add" in fx):
    t0, t1 = (np.array(fx["add"][k][:2], float) - org[:2] for k in ("tail", "tip"))
    if np.all(((t0 + t1) / 2 >= lo) & ((t0 + t1) / 2 < lo + np.array(cls.shape) * gres)): adds.append((t0, t1, fx["add"]))
found, dbg, done = [], [], []
queue = [((t0 + t1) / 2, (t1 - t0) / np.linalg.norm(t1 - t0), True, ad) for t0, t1, ad in adds] + [(c, ev, True, None) for c, ev in cands]
n_done = 0
while queue:
    c, ev, first, add = queue.pop(0); n_done += 1
    if n_done % 50 == 0: print(f"{n_done} candidates done, {len(queue)} queued, {len(found)} arrows", flush=True)
    if not first and any(np.linalg.norm(c - q) < 0.7 for q in done): continue
    e = ev if add is not None or not first else heading(c); e = ev if e is None else e
    med, good = None, None
    for it in range(3):                                              # align with the paint, re-render
        med, nviews = patch(c, e)
        ms = paint_masks(med, r) if med is not None else []
        if not ms: break
        good = (med, nviews, c, e)
        ang, (uc, vc) = axis_fit(ms[0], r); n_ = np.array([-e[1], e[0]]); ang = float(np.clip(ang, -0.17, 0.17))
        if abs(ang) < np.radians(1) and abs(vc) < 0.05 and abs(uc) < 0.3: break
        c = c + uc * e + vc * n_; e = np.cos(ang) * e + np.sin(ang) * n_
    if good is None: continue
    med, nviews, c, e = good                                           # last patch with paint at the centre
    done.append(c); n_ = np.array([-e[1], e[0]]); ms = paint_masks(med, r)
    for msk in (ms[1:] if first else []):     # other arrows in the patch: own candidate, centred (spawned ones do not spawn again:
                                              # they would re-spawn their neighbour at a slightly other centre, forever)
        u, v = uv_of(msk, r); queue.append((c + (u.min() + u.max()) / 2 * e + float(np.median(v)) * n_, e, False, None))
    if not ms: continue
    msk = ms[0]; s = shape(msk, r, a.min_score)
    if a.debug and a.debug_candidates:
        np.savez_compressed(f"{a.debug}/arrow_cand_{len(dbg)}.npz", med=med, mask=msk, c=c, e=e)
        t = med.copy(); t[msk] = (0, 0, 255) if s["ok"] else (255, 0, 0); t = cv2.resize(np.concatenate([med, t], 1)[::-1], (160, 200))
        cv2.putText(t, f"{len(dbg)}", (2, 14), 0, 0.5, (0, 255, 0), 1); dbg.append(t); print("debug", len(dbg) - 1, s, flush=True)
    if add is not None:                     # reviewer's arrow: tip along +e (tail -> tip), their directions if given
        u, v = uv_of(msk, r); s.update(ok=True, sign=1, u0=float(u.min()), u1=float(u.max()), v=float(np.median(v)), score=2.0, review=True,
                                      directions=add.get("directions", s["directions"]), shape=add.get("shape", s.get("shape", "?")),
                                      width=round(float(np.ptp(v)), 2), area=round(float(len(u) * r * r), 2), length=round(float(np.ptp(u)), 2))
    if not s["ok"]: continue
    c_ = c + s["v"] * n_
    tail2, tip2 = (c_ + s["u0"] * e, c_ + s["u1"] * e) if s["sign"] > 0 else (c_ + s["u1"] * e, c_ + s["u0"] * e)
    found.append(dict(c=c, e=e, med=med, mask=msk, nviews=nviews, tail2=tail2, tip2=tip2, **s))

# ---- merge duplicates (keep the best match)
def seg_dist(p, a0, a1):
    d = a1 - a0; t = np.clip((p - a0) @ d / max(d @ d, 1e-9), 0, 1); return float(np.linalg.norm(p - (a0 + t * d)))


CW = G.index("crosswalk") + 1
def on_crosswalk(p):                                                   # crosswalk stripes look like arrow shafts
    i, j = np.floor((p - lo) / gres).astype(int)
    return 0 <= i < cls.shape[0] and 0 <= j < cls.shape[1] and (cls[max(0, i - 3):i + 4, max(0, j - 3):j + 4] == CW).mean() > 0.3


found = [f for f in found if f.get("review") or not on_crosswalk((f["tail2"] + f["tip2"]) / 2)]
found.sort(key=lambda f: -f["score"]); keep = []
for f in found:                                                        # same arrow: middle within 0.5 m of a kept axis
    mid = (f["tail2"] + f["tip2"]) / 2                                 # (arrows side by side are ~1 m apart)
    if all(seg_dist(mid, g["tail2"], g["tip2"]) > 0.5 for g in keep): keep.append(f)


def to_map(xy):
    z = pj.height_at(torch.tensor(np.atleast_2d(xy), device=pj.dev)).cpu().numpy()
    return [[round(float(x + org[0]), 3), round(float(y + org[1]), 3), None if h < -1e3 else round(float(h + org[2]), 3)]
            for (x, y), h in zip(np.atleast_2d(xy), z)]


out, tiles = [], []
for k, f in enumerate(keep):
    msk = cv2.morphologyEx(f["mask"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [q for q in cs if cv2.contourArea(q) * r * r >= 0.01]          # all pieces (dashed shafts, detached branches)
    nu, nv = msk.shape; e = f["e"]; n_ = np.array([-e[1], e[0]]); rings = []
    for q in cs:
        poly = cv2.approxPolyDP(q, 0.015 / r, True).reshape(-1, 2)      # (col = v, row = u) pixels
        rings.append(to_map(f["c"] + ((poly[:, 1] - nu / 2 + 0.5) * r)[:, None] * e + ((poly[:, 0] - nv / 2 + 0.5) * r)[:, None] * n_))
    out.append(dict(id=k, tail=to_map(f["tail2"])[0], tip=to_map(f["tip2"])[0], directions=f["directions"], shape=f["shape"], score=f["score"],
                    length=f["length"], width=f["width"], area=f["area"], views=f["nviews"], auto=not f.get("review"), outlines=rings))
    t = f["med"].copy(); cv2.drawContours(t, cs, -1, (0, 0, 255), 1)
    t = np.concatenate([f["med"], t], 1)
    t = t[::-1, ::-1] if f["sign"] > 0 else t                       # tip up (rows: u), left on the left
    t = cv2.resize(t, (2 * 95, 200), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 18, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(t, f"{k} {f['shape']} {f['score']:.2f}", (2, 13), 0, 0.42, (0, 255, 0), 1); tiles.append(t)
if dbg:
    dbg += [np.zeros_like(dbg[0])] * (-len(dbg) % 12)
    cv2.imwrite(f"{a.debug}/arrows_debug.jpg", np.concatenate([np.concatenate(dbg[i:i + 12], 1) for i in range(0, len(dbg), 12)], 0))
json.dump(dict(arrows=out, candidates=len(cands), params=vars(a)), open(f"{a.labels}/arrows.json", "w"), indent=1)
if tiles and a.debug:
    os.makedirs(a.debug, exist_ok=True); per = 10; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per)
    cv2.imwrite(f"{a.debug}/arrows.jpg", np.concatenate([np.concatenate(tiles[i:i + per], 1) for i in range(0, len(tiles), per)], 0),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
print(json.dumps(dict(candidates=len(cands), arrows=len(out), directions={"+".join(o["directions"]): sum(1 for q in out if q["directions"] == o["directions"]) for o in out})))
