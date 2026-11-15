"""Evaluation: agreement of two vector maps (vectorize.py vectors.geojson), without any HD map.

Typical use: repeatability, vectors from drive A's labels vs. drive B's labels (independent images). Only the area
observed in both label maps is compared (cells observed in both ground_map.npz, optionally within --corridor m of a
trajectory). Per kind (lane_line, curb, road_boundary observed, crosswalk / marking_area outlines):
  point level     lines densified to 0.1 m; precision = share of A's points within tol of B's, recall vice versa,
                  F at --tols (m), and mean distance (each point capped at 1 m)
  instance level  (MapTR-like) A's elements clipped to the common area (>= --min-len m); an element matches a B element
                  if their symmetric chamfer distance <= thr; F at --thrs (m). For matched lane lines: style agreement.
                  Over the whole map this is strict: a long line split / extended differently in A and B does not match.
  per sample      with --samples (export.py scene dataset dir): the same instance F within each sample's BEV window (every
                  --sample-every th sample), both maps clipped to the window as a training sample would be.
Writes OUT.json.
"""

import argparse
import json

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("a", help="vectors.geojson A")
    p.add_argument("b", help="vectors.geojson B")
    p.add_argument("--labels-a", required=True)
    p.add_argument("--labels-b", required=True, help="ground_map.npz of A / B (observed area)")
    p.add_argument("--gt", help="restrict to --corridor m around this trajectory")
    p.add_argument("--corridor", type=float, default=15.0)
    p.add_argument("--tols", type=float, nargs="+", default=[0.2, 0.5])
    p.add_argument("--thrs", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--min-len", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", help="export.py dataset dir: also per-sample-window instance F")
    p.add_argument("--sample-every", type=int, default=5)
    a = p.parse_args(argv)

    A, B = np.load(a.labels_a), np.load(a.labels_b)
    res = float(A["res"])
    lo, origin = A["lo"], A["origin"]
    shift = np.rint((B["lo"] + B["origin"][:2] - lo - origin[:2]) / res).astype(int)
    ca, cb = A["cls"], B["cls"]
    cb2 = np.zeros_like(ca)
    r0, c0 = max(0, shift[0]), max(0, shift[1])
    r1, c1 = min(ca.shape[0], shift[0] + cb.shape[0]), min(ca.shape[1], shift[1] + cb.shape[1])
    cb2[r0:r1, c0:c1] = cb[r0 - shift[0] : r1 - shift[0], c0 - shift[1] : c1 - shift[1]]
    common = (ca > 0) & (cb2 > 0)
    if a.gt:
        g = np.load(a.gt)[:, 1:3] - origin[:2]
        ij = ((g - lo) / res).astype(int)
        m = np.zeros(ca.shape, bool)
        ok = np.all((ij >= 0) & (ij < ca.shape), 1)
        m[ij[ok, 0], ij[ok, 1]] = True
        common &= distance_transform_edt(~m) * res < a.corridor

    def in_common(P):
        ij = np.floor((P - lo) / res).astype(int)
        ok = np.all((ij >= 0) & (ij < common.shape), 1)
        r = np.zeros(len(P), bool)
        r[ok] = common[ij[ok, 0], ij[ok, 1]]
        return r

    def dens(c):
        c = np.array([q[:2] for q in c], float) - origin[:2]
        s0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        s = np.arange(0, s0[-1] + 1e-9, 0.1)
        return np.stack([np.interp(s, s0, c[:, d]) for d in range(2)], 1)

    def elements(f, kind):
        """densified elements of a kind, clipped to the common area -> list of (points, props)."""
        out = []
        for ft in json.load(open(f))["features"]:
            pr, g = ft["properties"], ft["geometry"]
            k = pr["kind"]
            if k == "road_boundary":
                k = "road_boundary" if pr["observed"] else None
            if k != kind:
                continue
            for c in g["coordinates"] if g["type"] == "Polygon" else [g["coordinates"]]:
                P = dens(c)
                inside = in_common(P)
                for run in np.split(np.arange(len(P)), np.flatnonzero(np.diff(inside.astype(int))) + 1):
                    if inside[run[0]] and len(run) >= 2:
                        out.append((P[run], pr))
        return out

    def match(ia, ib):
        """symmetric chamfer distance matrix between element lists (points, props)."""
        D = np.full((len(ia), len(ib)), np.inf)
        if not ia or not ib:
            return D
        tb = [cKDTree(e[0]) for e in ib]
        cb_ = np.array([e[0].mean(0) for e in ib])
        rb = np.array([np.linalg.norm(e[0] - e[0].mean(0), axis=1).max() for e in ib])
        for i, (P, _) in enumerate(ia):
            tp = cKDTree(P)
            c, rr = P.mean(0), np.linalg.norm(P - P.mean(0), axis=1).max()
            for j in np.flatnonzero(np.linalg.norm(cb_ - c, axis=1) < rr + rb + 2.0):
                D[i, j] = (tb[j].query(P)[0].mean() + tp.query(ib[j][0])[0].mean()) / 2
        return D

    def greedy(D, thr):
        used_a, used_b, pairs = set(), set(), []
        for i, j in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
            if D[i, j] > thr:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
        return pairs

    def prf(m, na, nb):
        pr_, rc = m / max(na, 1), m / max(nb, 1)
        return dict(
            n_a=na,
            n_b=nb,
            matched=m,
            precision=round(pr_, 4),
            recall=round(rc, 4),
            f=round(2 * pr_ * rc / max(pr_ + rc, 1e-9), 4),
        )

    def instance(ea, eb, kind):
        ia = [e for e in ea if len(e[0]) * 0.1 >= a.min_len]
        ib = [e for e in eb if len(e[0]) * 0.1 >= a.min_len]
        D = match(ia, ib)
        out = {}
        for thr in a.thrs:
            pairs = greedy(D, thr)
            e = prf(len(pairs), len(ia), len(ib))
            if kind == "lane_line" and pairs:
                e["style_agreement"] = round(
                    float(np.mean([ia[i][1]["style"] == ib[j][1]["style"] for i, j in pairs])), 4
                )
            out[f"{thr}m"] = e
        return out

    def per_sample(ea, eb, kind):
        """instance F summed over sample BEV windows (elements clipped to each window)."""
        bev = json.load(open(f"{a.samples}/scene.json"))["bev"]
        (x0, x1), (y0, y1) = bev["x"], bev["y"]
        rows = [json.loads(l) for l in open(f"{a.samples}/samples.jsonl")][:: a.sample_every]
        tot = {t: [0, 0, 0] for t in a.thrs}

        def clip(els, T):
            R, t = T[:2, :2], T[:2, 3] - origin[:2]
            out = []
            for P, pr in els:
                Q = (P - t) @ R
                ins = (Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 1] >= y0) & (Q[:, 1] <= y1)
                for run in np.split(np.arange(len(Q)), np.flatnonzero(np.diff(ins.astype(int))) + 1):
                    if ins[run[0]] and len(run) * 0.1 >= a.min_len:
                        out.append((Q[run], pr))
            return out

        for row in rows:
            T = np.array(row["T_map_base"])
            wa, wb = clip(ea, T), clip(eb, T)
            if not wa and not wb:
                continue
            D = match(wa, wb)
            for t in a.thrs:
                tot[t][0] += len(greedy(D, t))
                tot[t][1] += len(wa)
                tot[t][2] += len(wb)
        return {f"{t}m": prf(*tot[t]) for t in a.thrs} | dict(samples=len(rows))

    res_out = dict(a=a.a, b=a.b, common_km2=round(float(common.sum()) * res**2 / 1e6, 4), kinds={})
    for kind in ("lane_line", "curb", "road_boundary", "crosswalk", "marking_area"):
        ea, eb = elements(a.a, kind), elements(a.b, kind)
        if not ea or not eb:
            continue
        Pa, Pb = np.concatenate([e[0] for e in ea]), np.concatenate([e[0] for e in eb])
        da, db = cKDTree(Pb).query(Pa)[0], cKDTree(Pa).query(Pb)[0]
        r = dict(
            length_a_m=round(len(Pa) * 0.1),
            length_b_m=round(len(Pb) * 0.1),
            mean_dist_cm=round(float((np.minimum(da, 1).mean() + np.minimum(db, 1).mean()) / 2 * 100), 1),
            point={},
            instance={},
        )
        for t in a.tols:
            pr_, rc = float((da <= t).mean()), float((db <= t).mean())
            r["point"][f"{t}m"] = dict(
                precision=round(pr_, 4), recall=round(rc, 4), f=round(2 * pr_ * rc / max(pr_ + rc, 1e-9), 4)
            )
        r["instance"] = instance(ea, eb, kind)
        if a.samples:
            r["per_sample"] = per_sample(ea, eb, kind)
        res_out["kinds"][kind] = r
    json.dump(res_out, open(f"{a.out}.json", "w"), indent=1)
    for k, r in res_out["kinds"].items():
        print(
            f"{k:14s} pointF@{a.tols[0]} {r['point'][f'{a.tols[0]}m']['f']:.3f}  mean {r['mean_dist_cm']:5.1f} cm  "
            + "  ".join(f"instF@{t} {r['instance'][f'{t}m']['f']:.3f}" for t in a.thrs)
            + (f"  style {r['instance'][f'{a.thrs[-1]}m'].get('style_agreement')}" if k == "lane_line" else "")
            + (
                ""
                if "per_sample" not in r
                else "  | per sample " + "  ".join(f"F@{t} {r['per_sample'][f'{t}m']['f']:.3f}" for t in a.thrs)
            )
        )


if __name__ == "__main__":
    main()
