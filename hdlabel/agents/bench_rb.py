"""Road boundary group of the protected benchmark: reference lines made without the pipeline's own road_boundary
output, reviewed by a person, then added to bench/<name>/ and locked. Kept out of bench.py (the locked evaluator).

Propose (`./autolabel bench rb-propose <config> [--scene S ..] [--out DIR]`, CPU only, writes only into the draft
directory, default output/<name>/bench_draft/road_boundary/):
  every 1 m along the driven path of the bench scenes (localization poses), a height profile across the road in the
  raw LiDAR point cloud map (0.25 m bins, 20th percentile height). Walking out from the vehicle on each side, the first
  kerb - a height change of 7-35 cm within 0.5 m with a flat top behind it - is a boundary point; a tall obstacle (wall,
  parked vehicle), a drop, an unobserved gap or the end of the reach stop the walk without a point. Points of
  consecutive profiles at a similar offset are chained into lines (>= 6 m). No camera, segmentation or vector of the
  pipeline is used, so the reference does not inherit the errors of the tools it measures.
  The path is cut into windows (~40 m; a window over a road an earlier window already covers is skipped). A window's
  region is the area between the first boundaries on both sides (+1 m), or the reach where there is none.
  DIR/draft.json, DIR/evidence/rb_<window>.jpg (camera ortho image | LiDAR height image), DIR/review.html.
Review (DIR/review.html, the person): per window, each proposed line correct / wrong, boundaries without a kerb (flat
  edge to grass, wall, lowered kerb) drawn by clicking, and the window marked complete (every road boundary in its
  region is drawn) or not. Only complete windows are scored. -> rb_review.json
Apply (`./autolabel bench rb-apply <config> rb_review.json --by person:NAME [--out DIR]`): the decisions ->
  bench/<name>/road_boundary.json (windows: id, scene, split, complete, region, lines) + evidence/rb_<window>.jpg (with
  the final lines), then the benchmark is locked by that person. Refused if the locked data changed since the lock.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os

import cv2
import numpy as np

BIN = 0.25  # profile bin (m)
STEP_MIN, STEP_MAX = 0.07, 0.35  # kerb height range (m)
FLAT = 0.08  # max height spread of the kerb top over 1 m (m)
MAX_GAP_BINS = 3  # unobserved bins in a row that stop the walk
PX = 0.1  # evidence image resolution (m / px)


def draft_dir(area) -> str:
    return f"{area.out}/bench_draft/road_boundary"


# ---------------------------------------------------------------- LiDAR profiles
def low_bins(v: np.ndarray, z: np.ndarray, reach: float, q=0.2, min_n=3) -> np.ndarray:
    """per BIN of v in [-reach, reach): the q-quantile of z (NaN with fewer than min_n points)."""
    nb = int(round(2 * reach / BIN))
    b = np.floor((v + reach) / BIN).astype(int)
    k = (b >= 0) & (b < nb)
    b, z = b[k], z[k]
    out = np.full(nb, np.nan)
    if not len(b):
        return out
    o = np.lexsort((z, b))
    b, z = b[o], z[o]
    st = np.searchsorted(b, np.arange(nb))
    en = np.searchsorted(b, np.arange(nb), side="right")
    n = en - st
    ok = n >= min_n
    out[ok] = z[st[ok] + np.floor(q * (n[ok] - 1)).astype(int)]
    return out


def walk(h: np.ndarray, side: int) -> tuple[float | None, str, float]:
    """profile h (bins over [-reach, reach), the vehicle at the middle) -> (offset of the first kerb on that side, m,
    positive; or None), what stopped the walk (kerb | high | drop | rough | gap | reach), road height there."""
    nb = len(h)
    c = nb // 2
    road = [x for x in h[c - 4 : c + 4] if np.isfinite(x)]
    if len(road) < 3:
        return None, "gap", float("nan")
    road = [float(np.median(road))]
    k = c if side > 0 else c - 1
    gap = 0
    center = lambda i: (i + 0.5) * BIN - nb * BIN / 2  # noqa: E731
    while 0 <= k + side < nb:
        j = k + side
        if not np.isfinite(h[j]):
            gap += 1
            if gap > MAX_GAP_BINS:
                return None, "gap", road[-1]
            k = j
            continue
        gap = 0
        r = float(np.median(road[-3:]))
        if abs(h[j] - r) <= STEP_MIN:
            road.append(float(h[j]))
            k = j
            continue
        top = [h[i] for i in range(j + side, j + 5 * side, side) if 0 <= i < nb and np.isfinite(h[i])]
        if len(top) < 2:
            return None, "gap", r
        step = float(np.median(top[:3])) - r
        if step < -STEP_MIN:
            return None, "drop", r
        if step > STEP_MAX or max(top) - r > STEP_MAX + FLAT:
            return None, "high", r
        if step < STEP_MIN or max(top) - min(top) > FLAT:
            return None, "rough", r
        # sub-bin position: where the profile crosses the middle of the step (from the last road bin outward)
        half = r + step / 2
        i0 = j - side
        while 0 <= i0 - side < nb and not np.isfinite(h[i0]):
            i0 -= side
        x = [center(i) for i in (i0, j, j + side) if 0 <= i < nb and np.isfinite(h[i])]
        y = [h[i] for i in (i0, j, j + side) if 0 <= i < nb and np.isfinite(h[i])]
        pos = center(j)
        for a in range(len(x) - 1):
            if (y[a] - half) * (y[a + 1] - half) <= 0 and y[a + 1] != y[a]:
                pos = x[a] + (half - y[a]) / (y[a + 1] - y[a]) * (x[a + 1] - x[a])
                break
        return abs(pos), "kerb", r
    return None, "reach", float(np.median(road[-3:]))


def path_profiles(poses: np.ndarray, every=1.0) -> tuple[np.ndarray, np.ndarray]:
    """poses [N, >=3] (x y z, time order) -> profile origins [M, 3] every `every` m of the path and their left normals
    [M, 2] (heading from +-2 m of path)."""
    xy = poses[:, :2]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-3]
    poses, s = poses[keep], s[keep]
    if len(s) < 2 or s[-1] < 2 * every:
        return np.zeros((0, 3)), np.zeros((0, 2))
    t = np.arange(0, s[-1], every)
    O = np.stack([np.interp(t, s, poses[:, d]) for d in range(3)], 1)
    a = np.stack([np.interp(np.clip(t + 2, 0, s[-1]), s, poses[:, d]) for d in range(2)], 1)
    b = np.stack([np.interp(np.clip(t - 2, 0, s[-1]), s, poses[:, d]) for d in range(2)], 1)
    tan = a - b
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    return O, np.c_[-tan[:, 1], tan[:, 0]]


def profiles(O, N, Q, qtree, reach=12.0, along=0.75) -> list[dict]:
    """per profile (origin O[i], normal N[i]) and side (+1 left, -1 right): the walk result {off, stop, z}."""
    out = []
    for o, n in zip(O, N):
        idx = qtree.query_ball_point(o[:2], float(np.hypot(reach, along)) + 0.1)
        d = Q[idx, :2] - o[:2]
        t = np.array([n[1], -n[0]])
        u, v, z = d @ t, d @ n, Q[idx, 2]
        k = (np.abs(u) < along) & (z > o[2] - 1.0) & (z < o[2] + 1.0)
        h = low_bins(v[k], z[k], reach)
        out.append({side: dict(zip(("off", "stop", "z"), walk(h, side))) for side in (1, -1)})
    return out


def chains(O, N, prof, side: int, min_len=6.0, max_jump=0.4, max_skip=2) -> list[np.ndarray]:
    """kerb points of consecutive profiles on one side at a similar offset -> lines [K, 3] (median-smoothed offsets)
    of at least min_len m; up to max_skip profiles without a point are bridged."""
    pts = [(i, p[side]["off"], p[side]["z"]) for i, p in enumerate(prof) if p[side]["off"] is not None]
    groups, cur = [], []
    for i, off, z in pts:
        if cur and (i - cur[-1][0] > max_skip + 1 or abs(off - cur[-1][1]) > max_jump * (i - cur[-1][0])):
            groups.append(cur)
            cur = []
        cur.append((i, off, z))
    groups.append(cur)
    out = []
    for g in groups:
        if len(g) < 2:
            continue
        idx = np.array([x[0] for x in g])
        off = np.array([x[1] for x in g])
        z = np.array([x[2] for x in g])
        sm = np.array([np.median(off[max(0, a - 2) : a + 3]) for a in range(len(off))])
        P = np.c_[O[idx, :2] + side * sm[:, None] * N[idx], z]
        if np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum() >= min_len:
            out.append(np.c_[P, idx])  # last column: profile index
    return out


# ---------------------------------------------------------------- windows and regions
def region(O, N, lines: list, reach: float, margin=1.0) -> list:
    """the window's scored area: between the first reference line on each side of every profile (+margin), the reach
    where a side has none. -> exterior ring [[x, y], ...] of the largest polygon."""
    from shapely.geometry import LineString, MultiPoint, Point
    from shapely.ops import unary_union

    ref = unary_union([LineString([p[:2] for p in c]) for c in lines if len(c) > 1]) if lines else None
    ext = []
    for o, n in zip(O, N):
        e = []
        for side in (1, -1):
            d = reach
            if ref is not None and not ref.is_empty:
                hit = LineString([o[:2], o[:2] + side * reach * n]).intersection(ref)
                if not hit.is_empty:
                    d = min(reach, float(Point(o[:2]).distance(hit)) + margin)
            e.append(o[:2] + side * d * n)
        ext.append(e)
    quads = [MultiPoint([a[0], a[1], b[0], b[1]]).convex_hull for a, b in zip(ext[:-1], ext[1:])]
    g = unary_union(quads).buffer(0.05).buffer(-0.05)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda x: x.area)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in g.exterior.coords] if g.area > 0 else []


def windows(n: int, length=40.0, every=1.0) -> list[tuple[int, int]]:
    """the path cut into pieces of about `length` m: [(first, last + 1)] profile index ranges (a short tail joins the
    previous piece)."""
    k = max(2, int(round(length / every)))
    cuts = list(range(0, n, k))
    if len(cuts) > 1 and n - cuts[-1] < k / 2:
        cuts.pop()
    return [(a, b) for a, b in zip(cuts, cuts[1:] + [n])]


def split_of(scene: str, holdout: list | None, share=0.35) -> str:
    """the benchmark's split rule (bench.build): the reference's holdout scenes, else the same hash of the name."""
    if holdout is not None:
        return "holdout" if scene in holdout else "dev"
    return "holdout" if int(hashlib.sha1(scene.encode()).hexdigest(), 16) % 100 < share * 100 else "dev"


# ---------------------------------------------------------------- evidence images
def view_of(O, N, reach, pad=2.0) -> dict:
    E = np.concatenate([O[:, :2] + reach * N, O[:, :2] - reach * N])
    x0, y0 = E.min(0) - pad
    x1, y1 = E.max(0) + pad
    return dict(x0=round(float(x0), 2), y1=round(float(y1), 2), px=PX, w=int((x1 - x0) / PX), h=int((y1 - y0) / PX))


def to_px(view, xy) -> np.ndarray:
    xy = np.array([q[:2] for q in xy], float)
    return np.c_[(xy[:, 0] - view["x0"]) / view["px"], (view["y1"] - xy[:, 1]) / view["px"]]


def ortho_panel(view, ortho) -> np.ndarray:
    """camera ortho image of a scene (rot90 of its label grid: (img, lo, res, origin)) resampled to the view."""
    img, lo, res, origin = ortho
    cc, rr = np.meshgrid(np.arange(view["w"]), np.arange(view["h"]))
    x = view["x0"] + (cc + 0.5) * view["px"]
    y = view["y1"] - (rr + 0.5) * view["px"]
    i = (x - origin[0] - lo[0]) / res
    j = (y - origin[1] - lo[1]) / res
    J = img.shape[0]
    return cv2.remap(img, i.astype(np.float32), (J - 1 - j).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))  # fmt: skip


def lidar_panel(view, Q, O) -> np.ndarray:
    """LiDAR map points in the view: lowest height per 0.2 m cell relative to the road at the nearest path point, -0.1
    m (blue) to +0.3 m (red); a kerb is a sharp colour edge. Black: no points."""
    from scipy.spatial import cKDTree

    c = 0.2
    W, H = int(view["w"] * view["px"] / c) + 1, int(view["h"] * view["px"] / c) + 1
    ci = ((Q[:, 0] - view["x0"]) / c).astype(int)
    ri = ((view["y1"] - Q[:, 1]) / c).astype(int)
    k = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    dist, nn = cKDTree(O[:, :2]).query(Q[k, :2])
    dz = Q[k, 2] - O[nn, 2]
    near = (dist < 1.5) & (np.abs(dz) < 0.5)
    dz -= np.percentile(dz[near], 20) if near.sum() > 10 else 0.0  # road level (the pose is not exactly on it)
    keep = np.abs(dz) < 1.0
    g = np.full(H * W, np.inf)
    np.minimum.at(g, (ri[k] * W + ci[k])[keep], dz[keep])
    g = g.reshape(H, W)
    g = np.where(np.isfinite(g), g, cv2.erode(np.where(np.isfinite(g), g, 1e3).astype(np.float32), np.ones((3, 3))))
    g[g > 100] = np.inf  # cells without points and without a neighbour with points
    v = np.clip((g + 0.1) / 0.4, 0, 1)
    v[~np.isfinite(g)] = 0
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~np.isfinite(g)] = 0
    return cv2.resize(img, (view["w"], view["h"]), interpolation=cv2.INTER_NEAREST)


def overlay(panel, view, O, lines, reg) -> np.ndarray:
    """path (white), region (cyan) and lines (green, numbered) drawn on a panel."""
    img = panel.copy()
    cv2.polylines(img, [to_px(view, O).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    if reg:
        cv2.polylines(img, [to_px(view, reg).astype(np.int32)], True, (255, 220, 0), 1, cv2.LINE_AA)
    for k, ln in enumerate(lines):
        p = to_px(view, ln).astype(np.int32)
        cv2.polylines(img, [p], False, (0, 255, 0), 2, cv2.LINE_AA)
        m = p[len(p) // 2]
        cv2.putText(img, str(k), (int(m[0]) + 4, int(m[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def composite(out: str, w: dict, lines: list, reg: list) -> np.ndarray:
    """a draft window's panels side by side with the lines and the region drawn."""
    O = np.array(w["rays"]["o"])
    return np.concatenate([overlay(cv2.imread(f"{out}/{f}"), w["view"], O, lines, reg) for f in w["panels"]], 1)


# ---------------------------------------------------------------- proposal
def propose(map_path: str, scenes: list[dict], poses, out: str, ortho=None, holdout=None, reach=12.0,
            win_m=40.0, every=1.0) -> dict:  # fmt: skip
    """scenes [{id, drive, t_range}], poses(drive) -> [N, 8] (stamp x y z q..), ortho(scene) -> (img, lo, res, origin)
    or None -> out/draft.json + out/evidence/rb_<window>_<panel>.jpg (bare panels) + out/review.html."""
    from scipy.spatial import cKDTree

    from hdlabel.surface import map_files, read_pcd

    os.makedirs(f"{out}/evidence", exist_ok=True)
    seen = None  # path points of the windows kept so far (a road already covered is not proposed twice)
    wins = []
    stats = dict(profiles=0, sides=0, kerb=0, skipped_windows=0)
    for s in scenes:
        g = poses(s["drive"])
        g = g[(g[:, 0] >= s["t_range"][0]) & (g[:, 0] <= s["t_range"][1])]
        O, N = path_profiles(g[:, 1:4], every)
        if len(O) < 3:
            continue
        lo, hi = O[:, :2].min(0) - reach - 2, O[:, :2].max(0) + reach + 2
        files = map_files(map_path, O[:, :2], reach + 2)
        Q = np.concatenate([read_pcd(f) for f in files]).astype(np.float64)
        Q = Q[np.all((Q[:, :2] > lo) & (Q[:, :2] < hi), 1)]
        if not len(Q):
            continue
        qtree = cKDTree(Q[:, :2])
        prof = profiles(O, N, Q, qtree, reach)
        stats["profiles"] += len(prof)
        stats["sides"] += 2 * len(prof)
        stats["kerb"] += sum(p[sd]["stop"] == "kerb" for p in prof for sd in (1, -1))
        lines = [(sd, c) for sd in (1, -1) for c in chains(O, N, prof, sd)]
        orth = ortho(s["id"]) if ortho else None
        for wi, (a, b) in enumerate(windows(len(O), win_m, every)):
            if seen is not None and (cKDTree(seen).query(O[a:b, :2])[0] < 5.0).mean() > 0.5:
                stats["skipped_windows"] += 1
                continue
            seen = O[a:b, :2] if seen is None else np.r_[seen, O[a:b, :2]]
            wid = f"{s['id']}_w{wi:02d}"
            wl = []
            for sd, c in lines:
                k = (c[:, 3] >= a) & (c[:, 3] < b)
                if k.sum() >= 2:
                    wl.append(dict(id=f"{wid}_{len(wl)}", side="left" if sd > 0 else "right",
                                   coords=[[round(float(v), 3) for v in p[:3]] for p in c[k]], source="lidar_step"))  # fmt: skip
            Ow, Nw = O[a:b], N[a:b]
            view = view_of(Ow, Nw, reach)
            stops = {}
            for p in prof[a:b]:
                for sd in (1, -1):
                    stops[p[sd]["stop"]] = stops.get(p[sd]["stop"], 0) + 1
            pan = ([ortho_panel(view, orth)] if orth is not None else []) + [lidar_panel(view, Q, Ow)]
            ev = [f"evidence/rb_{wid}_{k}.jpg" for k in range(len(pan))]
            for f, im in zip(ev, pan):
                cv2.imwrite(f"{out}/{f}", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            wins.append(dict(id=wid, scene=s["id"], split=split_of(s["id"], holdout), view=view, panels=ev,
                             rays=dict(o=np.round(Ow, 3).tolist(), n=np.round(Nw, 4).tolist(), reach=reach),
                             lines=wl, stops=stops,
                             region=region(Ow, Nw, [x["coords"] for x in wl], reach)))  # fmt: skip
    draft = dict(built=datetime.datetime.now().isoformat(timespec="seconds"), scenes=[s["id"] for s in scenes],
                 params=dict(reach=reach, window_m=win_m, every=every, bin=BIN, step=[STEP_MIN, STEP_MAX], flat=FLAT),
                 stats=stats, windows=wins)  # fmt: skip
    json.dump(draft, open(f"{out}/draft.json", "w"), indent=1)
    review_page(out)
    return draft


def propose_area(area, out: str, scene_ids: list | None = None, bench_ref: str | None = None) -> dict:
    """propose() for an area's scenes: the given ones, else the scenes the benchmark reference (bench_ref:
    reference.json) names that the area still has, else all of them; split by the reference's holdout scenes. Reads
    the drives' poses and the owner scenes' camera ortho images (debug), writes only into `out`."""
    ref = json.load(open(bench_ref)) if bench_ref and os.path.exists(bench_ref) else {}
    all_ = {s["id"]: s for s in area.scenes()}
    ids = scene_ids or [s for s in sorted({e["scene"] for e in ref.get("items", [])}) if s in all_] or sorted(all_)
    scenes = [all_[s] for s in ids if s in all_]

    def ortho(sid):
        o = area.owners.get(sid, sid)
        f, g = f"{area.dbg_dir(o)}/ortho.jpg", f"{area.scene_dir(o)}/labels/ground_map.npz"
        if not (os.path.exists(f) and os.path.exists(g)):
            return None
        m = np.load(g)
        return cv2.imread(f), m["lo"], float(m["res"]), m["origin"]

    return propose(area.map, scenes, lambda d: np.load(f"{area.data(d)}/gt.npy"), out, ortho,
                   ref.get("holdout_scenes"))  # fmt: skip


def render_overlays(out: str, which: list | None = None) -> list[str]:
    """out/check/rb_<window>.jpg: the draft's panels with its proposed lines (numbered) and region, for a quick look."""
    d = json.load(open(f"{out}/draft.json"))
    os.makedirs(f"{out}/check", exist_ok=True)
    res = []
    for w in d["windows"]:
        if which and w["id"] not in which:
            continue
        f = f"{out}/check/rb_{w['id']}.jpg"
        img = composite(out, w, [x["coords"] for x in w["lines"]], w["region"])
        cv2.imwrite(f, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        res.append(f)
    return res


# ---------------------------------------------------------------- person review
def review_page(out: str) -> str:
    """out/review.html: per window the panels with the proposed lines (click: correct <-> wrong), drawing of missing
    boundaries (click points on either panel, finish with Enter or the button) and complete / not complete; saves
    rb_review.json for `bench rb-apply`."""
    d = json.load(open(f"{out}/draft.json"))
    data = [dict(id=w["id"], scene=w["scene"], split=w["split"], view=w["view"], panels=w["panels"],
                 path=[p[:2] for p in w["rays"]["o"]], region=w["region"], stops=w["stops"],
                 lines=[dict(id=x["id"], side=x["side"], xy=[p[:2] for p in x["coords"]]) for x in w["lines"]])
            for w in d["windows"]]  # fmt: skip
    page = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace(
        "__KEY__", f"rb_review_{d['built']}"
    )
    open(f"{out}/review.html", "w").write(page)
    return f"{out}/review.html"


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Road boundary reference</title>
<style>body{font:14px sans-serif;margin:16px;background:#fafafa}.card{background:#fff;border:2px solid #ddd;margin:0 0 16px;padding:6px}
.card.complete{border-color:#2a2}.card.incomplete{border-color:#999}.wrap{position:relative;display:inline-block;max-width:100%}
.wrap img{display:block;width:100%}.wrap svg{position:absolute;left:0;top:0;width:100%;height:100%}
.ln{fill:none;stroke-width:3;cursor:pointer;vector-effect:non-scaling-stroke}.ln.ok{stroke:#0f0}.ln.wrong{stroke:#f22;stroke-dasharray:6 4}
.dr{fill:none;stroke:#f0f;stroke-width:3;vector-effect:non-scaling-stroke}.rg{fill:none;stroke:#0df;stroke-width:1;vector-effect:non-scaling-stroke}
.pa{fill:none;stroke:#fff;stroke-width:1;vector-effect:non-scaling-stroke}.bar{position:sticky;top:0;background:#fafafa;padding:6px 0;z-index:2}
button{margin:4px 6px 0 0;padding:4px 10px}button.on{font-weight:bold;outline:2px solid #333}.guide{background:#fff;border:1px solid #ccc;padding:8px 12px;margin:0 0 14px}
.hide .card.complete,.hide .card.incomplete{display:none}</style>
<div class="bar"><b>Road boundary reference review</b> <label><input type="checkbox" id="h"> hide decided windows</label>
<button id="s">save decisions (JSON)</button> <span id="n"></span>
<small>Then: ./autolabel bench rb-apply CONFIG rb_review.json --by person:NAME</small></div>
<details open class="guide"><summary><b>How to judge</b></summary>
<p>Each window: left the camera top view (made from many frames, north up), right the LiDAR map heights relative to the
path (blue = road level, red = 30 cm higher, black = no LiDAR points). White: the driven path. <span style="color:#0a0">Green</span>:
a boundary proposed from a LiDAR kerb step (numbered). Cyan: the region that is scored (between the first boundaries on
both sides).</p>
<p>A road boundary is the outline of the drivable area: the road side of a kerb, or the edge of the road surface where
there is no kerb (grass, gravel, a wall, a lowered kerb at a driveway or crossing). A lane line is not a boundary.</p>
<ol><li>Click a green line that is not on a road boundary: it turns red (wrong). Click again to undo.</li>
<li>Missing boundary inside the window (within the cyan reach): press <b>draw</b>, click points along it on either
image, press <b>finish line</b> (or Enter). <b>undo point</b> removes the last point, <b>clear drawn</b> all drawn lines.</li>
<li><b>complete</b>: every road boundary of the drivable area in this window is now green or drawn (the window is scored).
<b>not complete</b>: you cannot tell or do not want to draw it (the window is not scored).</li></ol></details>
<div id="list"></div>
<script>
const W=__DATA__;const K='__KEY__';let dec={};try{dec=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const st=()=>{try{localStorage.setItem(K,JSON.stringify(dec))}catch(e){}};let drawing=null,cur=[];
const P=(v,xy)=>[(xy[0]-v.x0)/v.px,(v.y1-xy[1])/v.px];const pts=(v,a)=>a.map(p=>P(v,p).map(x=>x.toFixed(1)).join(',')).join(' ');
function D(id){return dec[id]||(dec[id]={complete:null,lines:{},drawn:[]})}
function svg(w){const v=w.view,d=D(w.id);let s=`<svg viewBox="0 0 ${v.w} ${v.h}" preserveAspectRatio="none">`;
s+=`<polyline class="pa" points="${pts(v,w.path)}"/>`;if(w.region.length)s+=`<polygon class="rg" points="${pts(v,w.region)}"/>`;
w.lines.forEach((l,i)=>{s+=`<polyline class="ln ${d.lines[l.id]==='wrong'?'wrong':'ok'}" data-l="${l.id}" points="${pts(v,l.xy)}"/>`});
d.drawn.concat(drawing===w.id&&cur.length?[cur]:[]).forEach(a=>{s+=`<polyline class="dr" points="${pts(v,a)}"/>`});
return s+'</svg>'}
function card(w){const d=D(w.id),v=w.view;const c=d.complete===true?'complete':d.complete===false?'incomplete':'';
let h=`<div class="card ${c}" data-w="${w.id}"><div><b>${w.id}</b> ${w.split}, ${w.lines.length} proposed, LiDAR walk stops: ${JSON.stringify(w.stops)}</div>`;
w.panels.forEach(f=>{h+=`<div class="wrap" style="width:${(99/w.panels.length).toFixed(1)}%"><img src="${f}">${svg(w)}</div>`});
h+=`<div><button data-a="draw" class="${drawing===w.id?'on':''}">draw</button><button data-a="finish">finish line</button><button data-a="undo">undo point</button>
<button data-a="clear">clear drawn</button> | <button data-a="complete" class="${d.complete===true?'on':''}">complete</button>
<button data-a="incomplete" class="${d.complete===false?'on':''}">not complete</button> drawn: ${d.drawn.length}</div></div>`;return h}
function show(id){if(id){const e=document.querySelector(`[data-w="${id}"]`);e.outerHTML=card(W.find(w=>w.id===id))}
else document.getElementById('list').innerHTML=W.map(card).join('');
const n=W.filter(w=>dec[w.id]&&dec[w.id].complete!==null).length;document.getElementById('n').textContent=`${n} of ${W.length} windows decided`}
function finish(){if(drawing&&cur.length>1)D(drawing).drawn.push(cur);const w=drawing;cur=[];drawing=null;st();if(w)show(w)}
document.getElementById('list').onclick=ev=>{const cd=ev.target.closest('.card');if(!cd)return;const id=cd.dataset.w,w=W.find(x=>x.id===id),d=D(id);
const b=ev.target.closest('button');if(b){const a=b.dataset.a;if(a==='draw'){if(drawing&&drawing!==id)finish();drawing=id;cur=[]}
else if(a==='finish'){finish();return}else if(a==='undo'){if(drawing===id)cur.pop()}else if(a==='clear'){d.drawn=[];cur=[]}
else if(a==='complete'||a==='incomplete'){if(drawing===id)finish();d.complete=d.complete===(a==='complete')?null:a==='complete'}st();show(id);return}
const l=ev.target.closest('.ln');if(l&&drawing!==id){d.lines[l.dataset.l]=d.lines[l.dataset.l]==='wrong'?'ok':'wrong';st();show(id);return}
const s=ev.target.closest('svg');if(s&&drawing===id){const r=s.getBoundingClientRect(),v=w.view;
const c=(ev.clientX-r.left)/r.width*v.w,y=(ev.clientY-r.top)/r.height*v.h;cur.push([+(v.x0+c*v.px).toFixed(2),+(v.y1-y*v.px).toFixed(2)]);show(id)}};
document.addEventListener('keydown',e=>{if(e.key==='Enter')finish()});
document.getElementById('h').onchange=ev=>document.body.classList.toggle('hide',ev.target.checked);
document.getElementById('s').onclick=()=>{finish();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(
[JSON.stringify({draft:K,windows:dec},null,1)],{type:'application/json'}));a.download='rb_review.json';a.click()};show();
</script>"""


# ---------------------------------------------------------------- apply
def apply(name: str, out: str, decisions: dict, by: str) -> dict:
    """the person's decisions {window: {complete, lines: {id: ok | wrong}, drawn: [[[x, y], ..], ..]}} on the draft
    in `out` -> bench/<name>/road_boundary.json + evidence/rb_<window>.jpg, locked by that person. In a complete
    window every proposed line not marked wrong is accepted."""
    from hdlabel.agents import bench

    if not by.startswith("person:"):
        raise ValueError("only a person accepts the road boundary reference (--by person:NAME)")
    d = bench.bench_dir(name)
    if os.path.exists(f"{d}/LOCK.json") and json.load(open(f"{d}/LOCK.json"))["data"] != bench._digest(d)["data"]:
        raise RuntimeError(f"benchmark {name}: its data changed since it was locked; resolve that first")
    draft = json.load(open(f"{out}/draft.json"))
    os.makedirs(f"{d}/evidence", exist_ok=True)
    wins, stat = [], dict(windows=0, complete=0, accepted=0, rejected=0, drawn=0)  # lines: of complete windows
    for w in draft["windows"]:
        x = decisions.get(w["id"])
        if not x or x.get("complete") is None:
            continue
        lines = []
        for ln in w["lines"]:
            if (x.get("lines") or {}).get(ln["id"]) == "wrong":
                stat["rejected"] += bool(x["complete"])
                continue
            lines.append(dict(coords=ln["coords"], source=ln["source"]))
            stat["accepted"] += bool(x["complete"])
        for c in x.get("drawn") or []:
            if len(c) > 1:
                lines.append(dict(coords=[[float(p[0]), float(p[1])] for p in c], source="drawn"))
                stat["drawn"] += bool(x["complete"])
        O, N = np.array(w["rays"]["o"]), np.array(w["rays"]["n"])
        reg = region(O, N, [ln["coords"] for ln in lines], w["rays"]["reach"])
        ev = f"evidence/rb_{w['id']}.jpg"
        img = composite(out, w, [ln["coords"] for ln in lines], reg)
        cv2.imwrite(f"{d}/{ev}", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        wins.append(dict(id=w["id"], scene=w["scene"], split=w["split"], complete=bool(x["complete"]), region=reg,
                         lines=lines, evidence=ev))  # fmt: skip
        stat["windows"] += 1
        stat["complete"] += bool(x["complete"])
    rb = dict(name=name, built=datetime.date.today().isoformat(), by=by, draft=draft["built"],
              params=draft["params"], windows=wins, review=stat)  # fmt: skip
    json.dump(rb, open(f"{d}/road_boundary.json", "w"), indent=1)
    bench.lock(name, by)
    return stat
