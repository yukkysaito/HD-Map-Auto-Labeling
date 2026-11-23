"""Vector map elements of a scene from its ground label map (project_labels.py ground_map.npz), without any HD map.

Elements (3D, map frame; z from the map surface):
  lane_line       LineString  thin road markings as painted: one element per painted piece (each dash of a dashed line
                              is its own element; gaps under --paint-gap m, e.g. worn paint, are bridged).
                              props: line_id (pieces chained into one line: collinear, gap <= --max-gap m over road),
                              style of that line solid | dashed | short, width (m), length (m) of the piece,
                              painted (1.0) and segments ([[0, length]]) kept for the format
  stop_line       LineString  a thin line ACROSS the lanes (common.stop_lines: |cos| to the direction of the long lane lines
                              around < --stop-cos, paint wider than --stop-width, longer than --stop-length). props: length, width
  marking_area    Polygon     wide or compact markings (stop bars wider than --line-width, symbols, hatched zones)
  arrow           LineString  lane direction arrow, tail -> tip (from OUT_OF_project_labels/arrows.json, steps/arrows.py).
                              props: directions (straight / left / right, e.g. ["straight", "left"]), shape (S, L, R, SL,
                              SR), score, length, width, outlines (paint outline rings, map xyz). Its paint is not turned
                              into lane lines / areas; other paint around it (e.g. dashes behind its tail) is output as is
  crosswalk       Polygon     crosswalk stripes merged into one area
  curb            LineString  curb skeleton paths joined across small gaps (occlusions)
  curb_candidate  LineString  with --curb-filter (and --gt trajectories): a curb line that is not a road kerb by
                              hdlabel/curbcheck.py (not on the edge of the road reachable from the trajectories, not
                              the first edge sideways from them, small loop, wavy, short at the edge of the view);
                              props.curb_check = the failed rules. Not a curb for training / the benchmark
  marking_candidate (geometry of its kind) with --marking-gate (and --gt trajectories): a lane_line / stop_line /
                              marking_area / arrow that does not lie on the level of the road reachable from the
                              trajectories by hdlabel/markingcheck.py (ledge, rail base, raised sidewalk, plaza behind a
                              kerb step); props.candidate_of = its kind, props.marking_check = the failed rules,
                              props.marking_level = the measurements. Not a marking for training / the benchmark
  drivable_area   Polygon     road + marking + crosswalk (holes = islands), observed cells only
  road_boundary   LineString  outline of the drivable area, split into runs; props.observed = the outside is an observed
                              non-drivable class (curb, sidewalk, off-road) -> a real boundary; false -> edge of the observed
                              area (unknown, do not train it as a boundary). With --boundary-shape (hdlabel/boundary.py)
                              zig-zag stretches and short observed runs are unknown too, and the outline is smoothed
Review fixes (--fixes, review/<name>/*.json written by a reviewer, see docs/ANNOTATION.md): the element of a kind nearest
to a point (map frame, <= 1.5 m) is dropped or gets properties set, e.g. {"kind": "marking_area", "at": [x, y],
"set": {"type": "zebra"}}; "set": {"kind": "crosswalk"} changes the element kind; arrows: {"kind": "arrow", "at": ..,
"set": {"directions": ["right"], "shape": "R"}} or "drop": true.
Every marking / crosswalk / curb element gets confidence (mean vote share of its class over its cells, 0..1) and
observations (median number of projected camera views of its cells).
Writes OUT/vectors.geojson (map coordinates, metres; "crs" is the map frame of the label map), OUT/vectors.json (counts,
lengths, parameters); with --debug DIR: DIR/vectors.jpg (elements drawn on DIR/ortho.jpg, half resolution).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, median_filter
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.boundary import drop_short_runs, resample_ring, ring_zigzag, smooth_ring
from hdlabel.common import stop_lines
from hdlabel.mapproj import arrow_cls, load_arrows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("labels", help="dir with ground_map.npz, arrows.json")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--debug", help="debug dir of the scene (ortho.jpg from project_labels.py; vectors.jpg is written there)"
    )
    p.add_argument("--line-width", type=float, default=0.5, help="markings wider than this (m, median) are areas")
    p.add_argument("--min-line", type=float, default=0.8, help="min painted segment length (m)")
    p.add_argument(
        "--max-gap",
        type=float,
        default=13.0,
        help="max gap between collinear dashes chained into one line (line_id, style, stop-line direction; m)",
    )
    p.add_argument(
        "--paint-gap",
        type=float,
        default=0.5,
        help="lane lines are output per painted piece; unpainted gaps shorter than this are bridged (m)",
    )
    p.add_argument("--curb-gap", type=float, default=3.0)
    p.add_argument("--min-curb", type=float, default=1.0)
    p.add_argument("--simplify", type=float, default=0.05, help="Douglas-Peucker tolerance for lines (m)")
    p.add_argument(
        "--stop-width",
        type=float,
        default=0.35,
        help="stop line: min paint width (m; lane lines ~0.15, stop lines ~0.45)",
    )
    p.add_argument("--stop-length", type=float, default=3.0, help="stop line: min length (m)")
    p.add_argument("--frag-conf", type=float, default=0.5, help="lane lines below this confidence are dropped")
    p.add_argument("--frag-length", type=float, default=2.2, help="lane lines shorter than this (m) ...")
    p.add_argument("--frag-views", type=int, default=20, help="... and seen in fewer views are dropped")
    p.add_argument(
        "--stop-cos",
        type=float,
        default=0.35,
        help="stop line: max |cos| to the lane direction (long lane lines within 15 m)",
    )
    p.add_argument("--min-drivable", type=float, default=20.0, help="min drivable component area (m2)")
    p.add_argument(
        "--curb-filter",
        action="store_true",
        help="curb lines that are not road kerbs (hdlabel/curbcheck.py: not next to the road reachable from the "
        "trajectories, not the first edge sideways from them, loops, wavy, short at the edge of the view) become "
        "kind curb_candidate with props.curb_check = the failed rules",
    )
    p.add_argument("--gt", nargs="*", default=[], help="gt.npy trajectories of the scene's drives (--curb-filter)")
    p.add_argument(
        "--curb-adjacent", type=float, default=2.0, help="--curb-filter: max distance to the road over curb cells (m)"
    )
    p.add_argument(
        "--curb-min-adjacent", type=float, default=0.3, help="--curb-filter: min share of a curb next to the road"
    )
    p.add_argument(
        "--curb-min-first", type=float, default=0.15, help="--curb-filter: min share of a curb hit first by side rays"
    )
    p.add_argument("--curb-wavy", type=float, default=30.0, help="--curb-filter: max mean turning (deg / m)")
    p.add_argument("--curb-rules", nargs="*", help="--curb-filter: apply only these rules (default: all)")
    p.add_argument(
        "--boundary-shape",
        action="store_true",
        help="road boundary rings (hdlabel/boundary.py): stretches that zig-zag (back-and-forth turning > "
        "--boundary-zigzag deg/m within --boundary-window m) are not observed boundaries; the outline is smoothed "
        "along the ring (sigma --boundary-smooth m)",
    )
    p.add_argument("--boundary-smooth", type=float, default=0.5, help="--boundary-shape: smoothing sigma (m)")
    p.add_argument("--boundary-window", type=float, default=3.0, help="--boundary-shape: zig-zag window (m)")
    p.add_argument("--boundary-chord", type=float, default=0.5, help="--boundary-shape: chord of the headings (m)")
    p.add_argument("--boundary-zigzag", type=float, default=40.0, help="--boundary-shape: max zig-zag (deg / m)")
    p.add_argument(
        "--boundary-min-run", type=float, default=3.0, help="--boundary-shape: shorter observed runs are unknown (m)"
    )
    p.add_argument(
        "--marking-gate",
        action="store_true",
        help="markings off the road level (hdlabel/markingcheck.py: LiDAR ground height vs the local plane of the road "
        "reachable from the trajectories, height step to it) become kind marking_candidate",
    )
    p.add_argument("--mark-tol", type=float, default=0.04, help="--marking-gate: max |height - road plane| (m)")
    p.add_argument(
        "--mark-min-level", type=float, default=0.7, help="--marking-gate: min share of known cells on the road level"
    )
    p.add_argument("--mark-step", type=float, default=0.07, help="--marking-gate: height step to the road (m)")
    p.add_argument("--mark-ref", type=float, default=3.0, help="--marking-gate: road cells within this distance (m)")
    p.add_argument(
        "--mark-kerb",
        type=float,
        default=0.0,
        help="--marking-gate: off_level not for elements mostly within this distance of a curb (m; 0: off)",
    )
    p.add_argument("--mark-rules", nargs="*", help="--marking-gate: apply only these rules (default: all)")
    p.add_argument(
        "--fixes",
        nargs="*",
        default=[],
        help="review json files with 'fixes' lists (map coordinates); fixes of any file "
        "that fall inside this grid are applied, so a review also covers other scenes / drives of the same place",
    )
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    m = np.load(f"{a.labels}/ground_map.npz")
    cls = m["cls"]
    res = float(m["res"])
    lo = m["lo"]
    origin = m["origin"]
    ARROWS = load_arrows(a.labels)  # steps/arrows.py: clean arrows from sharp views
    cls = arrow_cls(cls, ARROWS, lo, res, origin)
    H = np.where(m["height_w"] > 0, m["height"], np.nan)
    sres = float(m["surf_res"])
    G = [str(g) for g in m["groups"]]
    C = {g: G.index(g) + 1 for g in G}
    observed = cls > 0
    DRIVABLE = np.isin(cls, [C["road"], C["marking"], C["crosswalk"]])
    OFFROAD = np.isin(cls, [C["curb"], C["sidewalk"], C["offroad"]])
    N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def smooth_mask(mask, sigma=1.0, min_px=30):
        s = gaussian_filter(mask.astype(np.float32), sigma) > 0.5
        n, lab, st, _ = cv2.connectedComponentsWithStats(s.astype(np.uint8), connectivity=8)
        return np.isin(lab, np.flatnonzero(st[:, cv2.CC_STAT_AREA] >= min_px)[1:]) if n > 1 else s

    def skeleton_paths(mask, min_spur_px=6):
        """skeleton of mask -> list of pixel paths (N x 2 int arrays, row/col) between ends / junctions, short spurs removed."""
        sk = skeletonize(mask)
        pts = set(zip(*np.nonzero(sk)))

        def nbrs(q):
            r, c = q
            out = []
            for dr, dc in N8:
                n = (r + dr, c + dc)
                if n not in pts:
                    continue
                if dr and dc and ((r + dr, c) in pts or (r, c + dc) in pts):
                    continue  # diagonal only without an orthogonal detour
                out.append(n)
            return out

        for _ in range(2):  # spur pruning
            adj = {q: nbrs(q) for q in pts}
            drop = []
            for q, nb in adj.items():
                if len(nb) != 1:
                    continue
                chain, prev, cur = [q], None, q
                while len(chain) <= min_spur_px:
                    nx_ = [n for n in adj[cur] if n != prev]
                    if len(nx_) != 1:
                        break
                    prev, cur = cur, nx_[0]
                    if len(adj[cur]) != 2:
                        break
                    chain.append(cur)
                if len(adj[cur]) >= 3 and len(chain) <= min_spur_px:
                    drop += chain
            if not drop:
                break
            pts -= set(drop)
        adj = {q: nbrs(q) for q in pts}
        seen = set()
        paths = []
        key = lambda u, v: (u, v) if u < v else (v, u)
        starts = [q for q, nb in adj.items() if len(nb) != 2]
        for s in starts:  # paths between break points
            for n in adj[s]:
                if key(s, n) in seen:
                    continue
                path, prev, cur = [s], s, n
                seen.add(key(s, n))
                while True:
                    path.append(cur)
                    if len(adj[cur]) != 2:
                        break
                    nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                    if key(cur, nxt) in seen:
                        break
                    seen.add(key(cur, nxt))
                    prev, cur = cur, nxt
                paths.append(np.array(path))
        for q in list(adj):  # pure cycles
            if len(adj[q]) != 2 or key(q, adj[q][0]) in seen:
                continue
            path, prev, cur = [q], q, adj[q][0]
            seen.add(key(q, cur))
            while cur != q:
                path.append(cur)
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                seen.add(key(cur, nxt))
                prev, cur = cur, nxt
            paths.append(np.array(path + [q]))
        return paths

    def px2xy(rc):
        return lo + (np.asarray(rc, float) + 0.5) * res

    def xy2px(xy):
        return np.floor((np.asarray(xy) - lo) / res).astype(int)

    def plen(xy):
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0

    def cut(xy, s0, s1):
        """the part of the polyline between arc lengths s0 and s1."""
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        t = np.r_[s0, s[(s > s0) & (s < s1)], s1]
        return np.stack([np.interp(t, s, xy[:, j]) for j in (0, 1)], 1)

    def painted_pieces(segs, gap):
        """painted intervals with gaps < gap merged (segmentation noise, worn paint)."""
        out = []
        for u, v in segs:
            if out and u - out[-1][1] < gap:
                out[-1][1] = v
            else:
                out.append([u, v])
        return out

    def smooth_line(xy, k=3):
        if len(xy) < 2 * k + 1:
            return xy
        pad = np.r_[np.repeat(xy[:1], k, 0), xy, np.repeat(xy[-1:], k, 0)]
        sm = np.stack([np.convolve(pad[:, d], np.ones(2 * k + 1) / (2 * k + 1), "valid") for d in range(2)], 1)
        sm[0], sm[-1] = xy[0], xy[-1]
        return sm

    def simplify(xy, tol):
        if len(xy) < 3:
            return xy
        c = cv2.approxPolyDP((xy / tol * 8).astype(np.float32).reshape(-1, 1, 2), 8.0, False).reshape(-1, 2) / 8 * tol
        return c

    def end_dir(xy, at_end, span=1.5):
        """outward unit direction at one end, from the last `span` m of the line."""
        q = xy[::-1] if at_end else xy
        d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)])
        j = min(int(np.searchsorted(d, span)), len(q) - 1)
        v = q[0] - q[j]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    def gap_ok(p0, p1, bad_cls):
        n = max(2, int(np.linalg.norm(p1 - p0) / res))
        s = np.linspace(p0, p1, n)[1:-1]
        ij = xy2px(s)
        ok = np.all((ij >= 0) & (ij < cls.shape), 1)
        c = np.zeros(len(s), np.uint8)
        c[ok] = cls[ij[ok, 0], ij[ok, 1]]
        return np.isin(c, bad_cls).mean() < 0.2 if len(c) else True

    def join_once(items, max_gap, bad_cls, ang, span):
        """one round: join items (xy, painted intervals) end to end where one continues the other collinearly."""
        ends = []  # (item, at_end, point, outward dir)
        for i, (xy, _) in enumerate(items):
            for at_end in (False, True):
                d = end_dir(xy, at_end, span)
                if d is not None:
                    ends.append((i, at_end, xy[-1] if at_end else xy[0], d))
        cand = []
        for u, v in cKDTree(np.array([e[2] for e in ends])).query_pairs(max_gap) if ends else []:
            (i, _, pi, di), (j, _, pj, dj) = ends[u], ends[v]
            if i == j or np.dot(di, -dj) < np.cos(np.radians(ang)):
                continue  # outward directions must be opposite
            w = pj - pi
            g = np.linalg.norm(w)
            d = di - dj
            d /= np.linalg.norm(d)
            lat = abs(d[0] * w[1] - d[1] * w[0])  # lateral offset across the mean direction
            if lat > 0.15 + 0.02 * g or np.dot(d, w) < -0.3:
                continue
            if g > 0.3 and not gap_ok(pi, pj, bad_cls):
                continue
            cand.append((lat + 0.01 * g, u, v))
        parent = list(range(len(items)))
        used = set()
        link = {}
        find = lambda x: x if parent[x] == x else find(parent[x])
        for _, u, v in sorted(cand):
            if u in used or v in used:
                continue
            i, j = ends[u][0], ends[v][0]
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            parent[ri] = rj
            used |= {u, v}
            link[(i, ends[u][1])] = (j, ends[v][1])
            link[(j, ends[v][1])] = (i, ends[u][1])
        out, done = [], set()
        for i in range(len(items)):
            if i in done:
                continue
            cur, at, vis = i, False, {i}  # walk to a free end of the chain
            while (cur, at) in link and link[(cur, at)][0] not in vis:
                j, aj = link[(cur, at)]
                vis.add(j)
                cur, at = j, not aj
            pts, segs, s0 = [], [], 0.0  # traverse from the free end (cur, at)
            while True:
                done.add(cur)
                xy, sg = items[cur]
                L = plen(xy)
                if at:
                    xy, sg = xy[::-1], [[L - b, L - a_] for a_, b in sg[::-1]]
                if pts:
                    s0 += float(np.linalg.norm(xy[0] - pts[-1]))
                segs += [[round(s0 + a_, 2), round(s0 + b, 2)] for a_, b in sg]
                s0 += L
                pts += list(xy)
                nxt = link.get((cur, not at))
                if nxt is None or nxt[0] in done:
                    break
                cur, at = nxt
            out.append((np.array(pts), segs))
        return out

    def join(lines, max_gap, bad_cls, ang=10.0):
        """join line pieces into chains [(xy, painted intervals)]: first with directions from the piece ends (1.5 m),
        then chains again with directions from 5 m (dashes are short, their direction is noisy)."""
        items = [(xy, [[0.0, round(plen(xy), 2)]]) for xy in lines]
        for span in (1.5, 5.0):
            items = join_once(items, max_gap, bad_cls, ang, span)
        return items

    def zs(xy):
        z = map_coordinates(H, ((xy - lo) / sres).T - 0.5, order=1, mode="nearest")
        if np.isnan(z).all():
            return z
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        ok = ~np.isnan(z)
        return np.interp(s, s[ok], z[ok])

    def coords(xy):
        z = zs(xy)
        return [
            [
                round(float(x + origin[0]), 3),
                round(float(y + origin[1]), 3),
                None if np.isnan(h) else round(float(h + origin[2]), 3),
            ]
            for (x, y), h in zip(xy, z)
        ]

    def polygons(mask, eps, min_area):
        cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        out = []
        if hier is None:
            return out
        for k, c in enumerate(cs):
            if hier[0][k][3] != -1 or cv2.contourArea(c) * res**2 < min_area:
                continue
            rings = [c] + [
                cs[h] for h in range(len(cs)) if hier[0][h][3] == k and cv2.contourArea(cs[h]) * res**2 >= 1.0
            ]
            out.append([px2xy(cv2.approxPolyDP(r, eps / res, True).reshape(-1, 2)[:, ::-1]) for r in rings])
        return out

    feats = []

    def add(kind, geom, xy_or_rings, **props):
        if geom == "LineString":
            cc = coords(xy_or_rings)
        else:
            cc = [coords(np.r_[r, r[:1]]) for r in xy_or_rings]
        feats.append(
            dict(type="Feature", geometry=dict(type=geom, coordinates=cc), properties=dict(kind=kind, **props))
        )

    # ---- markings: thin -> lane line pieces, wide / compact -> areas
    # arrows are elements of their own: their paint (outline + 0.3 m) is not turned into lines / areas; all other paint is
    afoot = np.zeros(cls.shape, np.uint8)
    for ar in ARROWS:
        for o in ar["outlines"]:
            ring = np.array([q[:2] for q in o]) - origin[:2]
            cv2.fillPoly(afoot, [np.round((ring - lo) / res - 0.5).astype(np.int32)[:, ::-1]], 1)
    afoot = cv2.dilate(afoot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0

    mk = smooth_mask((cls == C["marking"]) & ~afoot)
    dt = distance_transform_edt(mk)
    n, lab = cv2.connectedComponents(mk.astype(np.uint8), connectivity=8)
    pieces, area_mask, wide = [], np.zeros_like(mk), np.zeros(mk.shape, np.uint8)
    paths = skeleton_paths(mk)
    by_comp = {}
    for pth in paths:
        by_comp.setdefault(int(lab[pth[0][0], pth[0][1]]), []).append(pth)
    for k, pl in by_comp.items():
        xs = np.concatenate(pl)
        ext = np.ptp(xs, 0).max() * res
        if len(pl) > 1 and ext < 3.0:  # compact with branches: symbol / arrow head
            area_mask |= lab == k
            continue
        for q in pl:  # along each skeleton path: thin runs -> lines, wide -> area
            wq = median_filter((2 * dt[q[:, 0], q[:, 1]] - 1) * res, 9, mode="nearest") > a.line_width
            for run in np.split(np.arange(len(q)), np.flatnonzero(np.diff(wq.astype(int))) + 1):
                if wq[run[0]]:
                    for (r, c), d in zip(q[run], dt[q[run, 0], q[run, 1]]):
                        cv2.circle(wide, (int(c), int(r)), int(d) + 1, 1, -1)
                    continue
                xy = smooth_line(px2xy(q[run]))
                if plen(xy) >= a.min_line:
                    pieces.append(xy)
    area_mask |= (wide > 0) & mk
    bad = [C["crosswalk"], C["curb"], C["sidewalk"], C["offroad"]]
    lls = []
    for xy, segs in join(pieces, a.max_gap, bad):
        L = plen(xy)
        painted = sum(s1 - s0 for s0, s1 in segs) / max(L, 1e-6)
        ij = xy2px(xy)
        ok = np.all((ij >= 0) & (ij < cls.shape), 1)
        ij = ij[ok]
        on = mk[ij[:, 0], ij[:, 1]]
        w = float(np.median(2 * dt[ij[on, 0], ij[on, 1]] - 1) * res) if on.any() else None
        style = "solid" if painted >= 0.85 and L >= 6 else ("dashed" if len(segs) >= 2 else "short")
        lls.append(dict(xy=xy, segs=segs, length=L, painted=painted, width=w, style=style))
    stop = stop_lines(lls, a.stop_width, a.stop_length, a.stop_cos)
    line_id = 0
    for k, l in enumerate(lls):
        if k in stop:
            add(
                "stop_line",
                "LineString",
                simplify(l["xy"], a.simplify),
                length=round(l["length"], 2),
                width=round(l["width"], 2),
            )
        else:  # one element per painted piece; the pieces of one line share line_id and the line's style
            line_id += 1
            for s0, s1 in painted_pieces(l["segs"], a.paint_gap):
                L = s1 - s0
                if L < a.min_line:
                    continue
                add(
                    "lane_line",
                    "LineString",
                    simplify(cut(l["xy"], s0, s1), a.simplify),
                    style=l["style"],
                    line_id=line_id,
                    painted=1.0,
                    length=round(L, 2),
                    width=None if l["width"] is None else round(l["width"], 2),
                    segments=[[0.0, round(L, 2)]],
                )
    for rings in polygons(area_mask, 0.05, 0.2):
        add("marking_area", "Polygon", rings)

    for ar in ARROWS:  # tail -> tip, props from arrows.py
        feats.append(
            dict(
                type="Feature",
                geometry=dict(type="LineString", coordinates=[ar["tail"], ar["tip"]]),
                properties=dict(
                    kind="arrow",
                    arrow_id=ar["id"],
                    directions=ar["directions"],
                    shape=ar["shape"],
                    score=ar["score"],
                    length=ar["length"],
                    width=ar["width"],
                    outlines=ar["outlines"],
                ),
            )
        )

    # ---- crosswalks: stripes merged
    cw = smooth_mask(cls == C["crosswalk"])
    cw = (
        cv2.morphologyEx(cw.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        > 0
    )
    for rings in polygons(cw, 0.15, 3.0):
        add("crosswalk", "Polygon", rings[:1])

    # ---- curbs
    cb = smooth_mask(cls == C["curb"])
    cpieces = [smooth_line(px2xy(q)) for q in skeleton_paths(cb)]
    cpieces = [q for q in cpieces if plen(q) >= 0.5]
    for xy, segs in join(cpieces, a.curb_gap, [C["crosswalk"], C["marking"]], ang=20.0):
        if plen(xy) >= a.min_curb:
            add("curb", "LineString", simplify(xy, a.simplify), length=round(plen(xy), 2))

    # ---- drivable area and its boundary (observed / unknown runs)
    dr = smooth_mask(DRIVABLE, 1.5, int(a.min_drivable / res**2))
    holes = ~dr & ~smooth_mask(observed, 1.5, 0)  # unobserved holes inside the road -> road
    n, lab = cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    sz = np.bincount(lab.ravel())
    fill = sz < 4.0 / res**2
    fill[border] = False
    fill[0] = False
    dr |= fill[lab]
    off_near = cv2.dilate(OFFROAD.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for rings in polygons(dr, 0.1, a.min_drivable):
        add(
            "drivable_area",
            "Polygon",
            rings,
            area=round(float(cv2.contourArea((rings[0] / res).astype(np.float32))) * res**2, 1),
        )
        for r in rings:  # boundary runs
            if a.boundary_shape:  # hdlabel/boundary.py: ragged stretches are not a real boundary, the rest smoothed
                ring = resample_ring(r, res)
                ragged = ring_zigzag(ring, res, a.boundary_window, a.boundary_chord) > a.boundary_zigzag
                dense = smooth_ring(ring, a.boundary_smooth / res)
                ragged, dense, ring = np.r_[ragged, ragged[:1]], np.r_[dense, dense[:1]], np.r_[ring, ring[:1]]
            else:
                rr, dense = np.r_[r, r[:1]], []
                for p0, p1 in zip(rr[:-1], rr[1:]):
                    nn = max(2, int(np.linalg.norm(p1 - p0) / res))
                    dense.append(np.linspace(p0, p1, nn, endpoint=False))
                ring = dense = np.concatenate(dense + [rr[-1:]])
            ij = np.clip(xy2px(ring), 0, np.array(cls.shape) - 1)
            obs = off_near[ij[:, 0], ij[:, 1]].astype(np.int8)
            if a.boundary_shape:
                obs &= ~ragged
            obs = np.convolve(obs, np.ones(15) / 15, "same") > 0.5  # 1.5 m majority
            if a.boundary_shape:  # short observed pieces between ragged / unknown stretches: unknown
                obs = drop_short_runs(obs[:-1], int(round(a.boundary_min_run / res)), closed=True)
                obs = np.r_[obs, obs[:1]]
            cut = np.flatnonzero(np.diff(obs.astype(int))) + 1
            for seg in np.split(np.arange(len(dense)), cut):
                if len(seg) < 2:
                    continue
                xy = dense[seg[0] : seg[-1] + 2] if seg[-1] + 1 < len(dense) else dense[seg[0] :]
                if plen(xy) < 1.0:
                    continue
                add(
                    "road_boundary",
                    "LineString",
                    simplify(xy, 0.1),
                    observed=bool(obs[seg[0]]),
                    length=round(plen(xy), 2),
                )

    # ---- review fixes: the element of the kind closest to the point (within 1.5 m)
    applied = 0
    fixes = [
        fx for f in a.fixes if os.path.exists(f) for fx in json.load(open(f)).get("fixes", []) if "at" in fx
    ]  # "add" fixes: arrows.py
    gx0, gx1 = origin[:2] + lo, origin[:2] + lo + np.array(cls.shape) * res
    fixes = [fx for fx in fixes if np.all((np.array(fx["at"]) > gx0) & (np.array(fx["at"]) < gx1))]
    if fixes:
        import shapely

        geom = lambda f: shapely.geometry.shape(
            dict(
                type=f["geometry"]["type"],
                coordinates=[[q[:2] for q in r] for r in f["geometry"]["coordinates"]]
                if f["geometry"]["type"] == "Polygon"
                else [q[:2] for q in f["geometry"]["coordinates"]],
            )
        )
        from hdlabel.review_html import element_at

        drop = set()
        for fx in fixes:
            # nearest element of the kind; among those touching the point (pieces meeting at a junction) the one whose
            # reference point (review_html.element_at, what the review recorded) is the point
            cand = [
                (d > 0.05, d if d > 0.05 else float(np.hypot(*np.subtract(element_at(f), fx["at"]))), i)
                for i, f in enumerate(feats)
                if f["properties"]["kind"] == fx["kind"]
                for d in [geom(f).distance(shapely.Point(fx["at"]))]
            ]
            if not cand or min(cand)[0] and min(cand)[1] > 1.5:
                continue  # not in this scene's vectors
            best = min(cand)
            # pieces whose reference point is exactly the point are all "the element at the point" of the review
            hit = [c[2] for c in cand if not c[0] and c[1] < 0.05] if not best[0] and best[1] < 0.05 else [best[2]]
            applied += 1
            for i in hit:
                if fx.get("drop"):
                    drop.add(i)
                feats[i]["properties"].update(fx.get("set", {}), reviewed=True)
        feats = [f for i, f in enumerate(feats) if i not in drop]

    # ---- per element: confidence = mean vote share of its class over its cells, observations = median number of
    # projected views of its cells (ground_map votes / n of project_labels.py)
    CELLS, VOTES, NV = m["cells"], m["votes"].astype(np.float32), m["n"]
    KIND_CLASS = dict(lane_line="marking", stop_line="marking", marking_area="marking", arrow="marking",
                      crosswalk="crosswalk", curb="curb")  # fmt: skip

    def element_ij(f):
        """label grid cells (row, col) of an element: along a line every res, inside a polygon / arrow outline."""
        g, pr = f["geometry"], f["properties"]
        if g["type"] == "LineString" and pr["kind"] != "arrow":
            P = np.asarray(g["coordinates"], float)[:, :2] - origin[:2]
            sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            at = np.arange(0, sl[-1] + 1e-9, res)
            if pr.get("segments"):
                at = at[np.any([(at >= u) & (at <= v) for u, v in pr["segments"]], 0)]
            ij = np.floor((np.stack([np.interp(at, sl, P[:, k]) for k in (0, 1)], 1) - lo) / res).astype(int)
        else:
            rings = pr["outlines"] if pr["kind"] == "arrow" else g["coordinates"][:1]
            Q = [np.asarray(r, float)[:, :2] - origin[:2] for r in rings]
            q0 = np.floor((np.concatenate(Q).min(0) - lo) / res).astype(int)
            q1 = np.ceil((np.concatenate(Q).max(0) - lo) / res).astype(int) + 1
            win = np.zeros(q1 - q0, np.uint8)
            cv2.fillPoly(win, [np.round((q - lo) / res - q0 - 0.5).astype(np.int32)[:, ::-1] for q in Q], 1)
            ij = np.argwhere(win > 0) + q0
        return ij[np.all((ij >= 0) & (ij < cls.shape), 1)]

    def element_cells(f):
        ij = element_ij(f)
        flat = ij[:, 0] * cls.shape[1] + ij[:, 1]
        k = np.minimum(np.searchsorted(CELLS, flat), len(CELLS) - 1)
        return k[CELLS[k] == flat]

    for f in feats:
        pr = f["properties"]
        if pr["kind"] not in KIND_CLASS:
            continue
        k = element_cells(f)
        v = VOTES[k]
        tot = v.sum(1)
        ok = tot > 0
        c = C[KIND_CLASS[pr["kind"]]] - 1
        pr["confidence"] = round(float((v[ok, c] / tot[ok]).mean()), 3) if ok.any() else None
        pr["observations"] = int(np.median(NV[k])) if len(k) else 0

    # ---- lane-line fragments: low confidence, or short and seen by few views. Review of 70 lane lines: such
    # fragments were wrong 42 times out of 52 (shadows, joints, kerb edges, worn asphalt); they are 3.8 % of the lane
    # line length. Elements a review set attributes on (reviewed) are kept.
    def fragment(pr):
        if pr["kind"] != "lane_line" or pr.get("reviewed"):
            return False
        low = pr["confidence"] is not None and pr["confidence"] < a.frag_conf
        return low or (pr["length"] < a.frag_length and pr["observations"] < a.frag_views)

    n0 = len(feats)
    feats = [f for f in feats if not fragment(f["properties"])]
    dropped_fragments = n0 - len(feats)

    # ---- curb filter (--curb-filter, hdlabel/curbcheck.py): curbs that are not road kerbs -> curb_candidate
    def trajectories():
        """grid-local xy runs of the --gt trajectories on this road level (gaps > 2 m split them)."""
        trajs = []
        for g in a.gt:
            P = np.load(g)[:, 1:4] - origin
            z = map_coordinates(H, ((P[:, :2] - lo) / sres).T - 0.5, order=0, mode="constant", cval=np.nan)
            on = np.abs(P[:, 2] - z) < 1.5  # this road level (not a bridge above / road below); off the grid: nan
            for run in np.split(np.arange(len(P)), np.flatnonzero(np.diff(on.astype(int)) != 0) + 1):
                if on[run[0]] and len(run) >= 2:
                    q = P[run, :2]
                    cut = np.flatnonzero(np.linalg.norm(np.diff(q, axis=0), axis=1) > 2.0) + 1
                    trajs += [t for t in np.split(q, cut) if len(t) >= 2]
        return trajs

    curb_filter = None
    if a.curb_filter:
        from hdlabel.curbcheck import check_curbs

        trajs = trajectories()
        k = max(1, int(round(sres / res)))
        surf = np.repeat(np.repeat(m["height_w"] > 0, k, 0), k, 1)[: cls.shape[0], : cls.shape[1]]
        surf = np.pad(surf, [(0, cls.shape[0] - surf.shape[0]), (0, cls.shape[1] - surf.shape[1])])
        surf = cv2.dilate(surf.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0  # 1 m: sparse LiDAR at the edges
        idx = [
            i for i, f in enumerate(feats) if f["properties"]["kind"] == "curb" and not f["properties"].get("reviewed")
        ]
        lines = [np.asarray(feats[i]["geometry"]["coordinates"], float)[:, :2] - origin[:2] for i in idx]
        params = dict(adj_m=a.curb_adjacent, min_adjacent=a.curb_min_adjacent, min_first=a.curb_min_first,
                      wavy_deg_per_m=a.curb_wavy)  # fmt: skip
        why, st = check_curbs(lines, cls, C, lo, res, trajs, surf, params)
        rules = {}
        if a.curb_rules:  # only these rules flag a curb (the others are measured but not applied)
            why = [[r for r in w if r in a.curb_rules] for w in why]
        for i, w in zip(idx, why):
            if w:
                feats[i]["properties"].update(kind="curb_candidate", curb_check=w)
                for r in w:
                    rules[r] = rules.get(r, 0) + 1
        curb_filter = dict(st, curbs=len(idx), candidates=sum(bool(w) for w in why), rules=rules)

    # ---- marking gate (--marking-gate, hdlabel/markingcheck.py): markings off the road level -> marking_candidate
    marking_gate = None
    if a.marking_gate:
        from hdlabel.markingcheck import check_markings, upsample

        Hc = upsample(H, max(1, int(round(sres / res))), cls.shape)
        idx = [i for i, f in enumerate(feats) if f["properties"]["kind"] in ("lane_line", "stop_line", "marking_area",
               "arrow") and not f["properties"].get("reviewed")]  # fmt: skip
        params = dict(tol=a.mark_tol, min_on_level=a.mark_min_level, step_m=a.mark_step, ref_m=a.mark_ref,
                      kerb_m=a.mark_kerb)  # fmt: skip
        why, mst, st = check_markings([element_ij(feats[i]) for i in idx], cls, C, lo, res, Hc, trajectories(), params)
        if a.mark_rules:  # only these rules demote a marking (the others are measured but not applied)
            why = [[r for r in w if r in a.mark_rules] for w in why]
        rules, kinds = {}, {}
        for i, w, ms in zip(idx, why, mst):
            pr = feats[i]["properties"]
            pr["marking_level"] = ms
            if w:
                kinds[pr["kind"]] = kinds.get(pr["kind"], 0) + 1
                pr.update(candidate_of=pr["kind"], kind="marking_candidate", marking_check=w)
                for r in w:
                    rules[r] = rules.get(r, 0) + 1
        marking_gate = dict(st, markings=len(idx), candidates=sum(bool(w) for w in why), kinds=kinds, rules=rules)

    # ---- outputs
    crs = dict(type="name", properties=dict(name="map frame of the label map (Autoware map, metres)"))
    json.dump(dict(type="FeatureCollection", crs=crs, features=feats), open(f"{a.out}/vectors.geojson", "w"))
    summ = {}
    for f in feats:
        pr = f["properties"]
        k = (
            pr["kind"]
            + (f"/{pr['style']}" if "style" in pr else "")
            + ("" if "observed" not in pr else ("/observed" if pr["observed"] else "/unknown"))
        )
        s = summ.setdefault(k, dict(n=0, length_m=0.0))
        s["n"] += 1
        s["length_m"] = round(s["length_m"] + pr.get("length", 0.0), 1)
    hide = (
        set() if a.curb_filter else {"curb_filter", "curb_adjacent", "curb_min_adjacent", "curb_min_first", "curb_wavy"}
    )
    if not a.marking_gate:
        hide |= {"marking_gate", "mark_tol", "mark_min_level", "mark_step", "mark_ref", "mark_kerb", "mark_rules"}
    if not (a.curb_filter or a.marking_gate):
        hide.add("gt")
    if not a.boundary_shape:
        hide |= {k for k in vars(a) if k.startswith("boundary_")}
    params = {k: v for k, v in vars(a).items() if k not in hide}  # default: as before
    info = dict(
        source=os.path.abspath(a.labels),
        params=params,
        fixes_in_grid=len(fixes),
        fixes_applied=applied,
        dropped_fragments=dropped_fragments,
        elements=summ,
    )
    if curb_filter is not None:
        info["curb_filter"] = curb_filter
    if marking_gate is not None:
        info["marking_gate"] = marking_gate
    json.dump(info, open(f"{a.out}/vectors.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("elements: " + ", ".join(f"{k} {v['n'] if isinstance(v, dict) and 'n' in v else v}" for k, v in summ.items()))

    ortho = cv2.imread(f"{a.debug}/ortho.jpg") if a.debug else None
    if ortho is not None:
        sc = 0.5
        img = cv2.resize(ortho, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        img = (img * 0.6).astype(np.uint8)
        Wc = cls.shape[1]

        def ipx(cc):  # map coords -> preview pixel (ortho = rot90 of the [x, y] grid: col = x, row = W - 1 - y)
            q = np.array([c[:2] for c in cc]) - origin[:2]
            ij = (q - lo) / res
            return np.round(np.c_[ij[:, 0], Wc - ij[:, 1]] * sc).astype(np.int32)

        COL = {
            "lane_line/solid": (255, 255, 255),
            "lane_line/dashed": (0, 255, 255),
            "lane_line/short": (0, 160, 255),
            "curb": (255, 80, 0),
            "curb_candidate": (160, 0, 255),
            "marking_candidate": (160, 0, 255),
            "road_boundary/observed": (0, 0, 255),
            "road_boundary/unknown": (120, 120, 120),
            "arrow": (0, 255, 0),
            "stop_line": (255, 255, 0),
        }
        for f in feats:
            pr, g = f["properties"], f["geometry"]
            if g["type"] == "Polygon":
                col = {"drivable_area": (0, 120, 0), "crosswalk": (255, 0, 255), "marking_area": (0, 200, 255),
                       "marking_candidate": (160, 0, 255)}[pr["kind"]]  # fmt: skip
                if pr["kind"] == "drivable_area":
                    continue
                cv2.polylines(img, [ipx(r) for r in g["coordinates"]], True, col, 1, cv2.LINE_AA)
            else:
                k = (
                    pr["kind"]
                    + (f"/{pr['style']}" if "style" in pr else "")
                    + ("" if "observed" not in pr else ("/observed" if pr["observed"] else "/unknown"))
                )
                if k.startswith("marking_candidate"):
                    k = "marking_candidate"
                cv2.polylines(img, [ipx(g["coordinates"])], False, COL[k], 1, cv2.LINE_AA)
                if k == "arrow" or "outlines" in pr:
                    cv2.polylines(img, [ipx(o) for o in pr["outlines"]], True, COL[k], 1, cv2.LINE_AA)
        cv2.imwrite(f"{a.debug}/vectors.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
