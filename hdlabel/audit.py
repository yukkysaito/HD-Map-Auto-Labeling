"""Acceptance audit of a labelled output: per scene risk from the automatic checks, ranked, plus a reproducible random
sample for human inspection, so reviewers look at the riskiest scenes first and accept a batch on a fixed rule.

Risk terms (0 = fine, 1 = bad), weighted mean -> risk:
  curb        1 - share of curb profiles with a LiDAR height step (geom_check), below 0.8 counts
  height      vector z vs LiDAR ground p95 above 10 cm (geom_check)
  qa          share of samples failing QA (export)
  calib       time offsets: coarse / fallback 0.5, failed 1 (scene.json)
  boundary    share of the road boundary length that is only the edge of the observed area
  confidence  share of marking / crosswalk / curb elements with confidence < 0.5 (vectorize)
  repeat      1 - agreement of the labels of two drives (./autolabel qa), when measured
Writes debug/audit.json and debug/audit.html (links to each scene's review sheets).
"""

from __future__ import annotations

import html
import json
import os
import random

import numpy as np

WEIGHTS = dict(curb=1.0, height=0.5, qa=1.0, calib=1.0, boundary=0.5, confidence=1.0, repeat=1.0)


def _len(c):
    P = np.asarray(c, float)[:, :2]
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def scene_terms(dataset_dir: str, debug_dir: str, vectors: str, qa_dir: str | None) -> dict:
    """risk terms of one scene (missing inputs -> term absent)."""
    t = {}
    gc = f"{debug_dir}/geom_check.json"
    if os.path.exists(gc):
        g = json.load(open(gc))
        if g.get("curb", {}).get("profiles"):
            t["curb"] = max(0.0, 0.8 - g["curb"]["step_within_0_3m"]) / 0.8
        if g.get("height", {}).get("points"):
            t["height"] = min(1.0, max(0.0, g["height"]["p95_cm"] - 10.0) / 20.0)
    sc = json.load(open(f"{dataset_dir}/scene.json"))
    t["qa"] = 1.0 - sc["qa_ok"] / max(sc["samples"], 1)
    st = [c.get("time_offset_status") for c in sc["cameras"].values()]
    t["calib"] = 1.0 if "failed" in st else (0.5 if any(s in ("coarse", "fallback") for s in st) else 0.0)
    feats = json.load(open(vectors))["features"]
    bl = [
        (_len(f["geometry"]["coordinates"]), f["properties"].get("observed"))
        for f in feats
        if f["properties"]["kind"] == "road_boundary"
    ]
    if bl and sum(x for x, _ in bl) > 0:
        t["boundary"] = sum(x for x, o in bl if not o) / sum(x for x, _ in bl)
    conf = [f["properties"]["confidence"] for f in feats if f["properties"].get("confidence") is not None]
    if conf:
        t["confidence"] = float(np.mean(np.array(conf) < 0.5))
    if qa_dir and os.path.exists(f"{qa_dir}/labels.json"):
        t["repeat"] = 1.0 - float(json.load(open(f"{qa_dir}/labels.json"))["agreement"])
    return {k: round(float(v), 3) for k, v in t.items()}


def risk(terms: dict, weights: dict = WEIGHTS) -> float:
    w = {k: weights[k] for k in terms if k in weights}
    return round(sum(terms[k] * w[k] for k in w) / max(sum(w.values()), 1e-9), 3)


def audit(scenes: list[dict], sample: int = 5, seed: int = 0, max_risk: float = 0.3) -> dict:
    """scenes: dicts with scene, terms -> ranking, random sample, batch verdict (accept when no scene exceeds max_risk)."""
    for s in scenes:
        s["risk"] = risk(s["terms"])
    ranked = sorted(scenes, key=lambda s: -s["risk"])
    rnd = random.Random(seed)
    pick = sorted(rnd.sample([s["scene"] for s in scenes], min(sample, len(scenes))))
    over = [s["scene"] for s in ranked if s["risk"] > max_risk]
    return dict(rule=dict(max_risk=max_risk, sample=sample, seed=seed, weights=WEIGHTS), accept=not over, over_max_risk=over,
                random_sample=pick, scenes=ranked)  # fmt: skip


def write_html(res: dict, path: str, link) -> None:
    """ranking table; link(scene) -> relative path of the scene's debug directory."""
    keys = list(WEIGHTS)
    rows = []
    for s in res["scenes"]:
        cells = "".join(f"<td>{s['terms'].get(k, '')}</td>" for k in keys)
        mark = " *" if s["scene"] in res["random_sample"] else ""
        d = link(s["scene"])
        a = f"<a href='{d}/review.html'>review</a> <a href='{d}/map.jpg'>map</a> <a href='{d}/arrows.jpg'>arrows</a>"
        bad = " class='bad'" if s["scene"] in res["over_max_risk"] else ""
        rows.append(
            f"<tr{bad}><td>{html.escape(s['scene'])}{mark}</td><td><b>{s['risk']}</b></td>{cells}<td>{a}</td></tr>"
        )
    verdict = (
        "ACCEPT"
        if res["accept"]
        else f"REVIEW: {len(res['over_max_risk'])} scenes over max risk {res['rule']['max_risk']}"
    )
    open(path, "w").write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Audit</title><style>
body{{font:14px sans-serif;margin:16px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:3px 6px;text-align:right}}
td:first-child{{text-align:left}} tr.bad{{background:#fdd}}</style></head><body>
<h2>Audit: {verdict}</h2><p>risk = weighted mean of the terms (0 fine .. 1 bad, see hdlabel/audit.py); * = random sample
(seed {res["rule"]["seed"]}) to inspect by hand; red = over max risk.</p>
<table><tr><th>scene</th><th>risk</th>{"".join(f"<th>{k}</th>" for k in keys)}<th></th></tr>{"".join(rows)}</table></body></html>""")
