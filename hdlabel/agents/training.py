"""Model loop: train the small multi-camera BEV network (hdlabel/eval/train_smoke.py) on the accepted data, evaluate it,
and turn its failures into work for the data and tool loops.

Failure analysis (debug/eval/analysis.json, knowledge kind "training"):
  classes      val IoU per class vs the prior baseline (a class the model cannot learn above the prior points to too
               little data or inconsistent labels)
  scenes       val scenes whose IoU is far below the others, with their verification issue counts
  suspects     places where the model confidently disagrees with the labels (clustered over samples, map frame):
               written to debug/model_suspects.json; the next debug step puts them into the evidence packs (label-only
               -> the element is checked, model-only -> a map tile for missed paint), so the VERIFIER decides from the
               sensor evidence. The model's output never becomes a label.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from hdlabel.common import ROOT


def accepted(area) -> list[str]:
    out = []
    for s in area.scenes():
        f = f"{area.dbg_dir(s['id'])}/verification.json"
        if os.path.exists(f) and json.load(open(f))["state"] == "accepted":
            out.append(s["id"])
    return out


def cluster(points: list[dict], radius=3.0) -> list[dict]:
    """disagreements of the same class / type within radius m -> one suspect with the number of samples."""
    out = []
    for p in sorted(points, key=lambda p: -p["area_m2"]):
        for c in out:
            if (
                c["cls"] == p["cls"]
                and c["type"] == p["type"]
                and np.hypot(*(np.subtract(c["map_xy"], p["map_xy"]))) < radius
            ):
                c["samples"] += 1
                c["area_m2"] = max(c["area_m2"], p["area_m2"])
                break
        else:
            out.append(dict(p, samples=1))
    return sorted(out, key=lambda c: -c["samples"])


def analyse(area, res: dict) -> dict:
    iou = res["val_iou"]
    weak = {c: v for c, v in iou.items() if v["model"] - v["prior"] < 0.05}
    scenes = res.get("val_scene_iou", {})
    mean = {sid: float(np.mean(list(v.values()))) for sid, v in scenes.items() if v}
    med = float(np.median(list(mean.values()))) if mean else 0
    issues = {}
    for sid in mean:
        f = f"{area.dbg_dir(sid)}/verification.json"
        issues[sid] = len(json.load(open(f)).get("issues", [])) if os.path.exists(f) else None
    low = [dict(scene=s, mean_iou=round(m, 3), issues=issues.get(s)) for s, m in mean.items() if m < 0.7 * med]
    sus = [c for c in cluster(res.get("disagreements", [])) if c["samples"] >= 2]
    return dict(weak_classes=weak, low_scenes=low, suspects=sus[:40], median_scene_iou=round(med, 3))


def loop(area, kb, steps=600) -> dict:
    scenes = accepted(area)
    if not scenes:
        return dict(result="no accepted scenes")
    ds = f"{area.dataset}/dataset.json"
    log = f"{area.dbg}/logs/train_agent.log"
    cmd = [
        sys.executable,
        f"{ROOT}/hdlabel/eval/train_smoke.py",
        ds,
        "--steps",
        str(steps),
        "--tag",
        "_agent",
        "--scenes",
        *scenes,
    ]
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode:
        kb.add("training", "training failed", open(log).read()[-800:], ["training"], status="failed", area=area.name)
        return dict(result=f"training failed (log {log})")
    res = json.load(open(f"{area.dbg}/eval/train_smoke_agent.json"))
    an = analyse(area, res)
    json.dump(an, open(f"{area.dbg}/eval/analysis.json", "w"), indent=1)
    json.dump(an["suspects"], open(f"{area.dbg}/model_suspects.json", "w"), indent=1)
    idx = json.load(open(ds))
    ious = ", ".join(f"{c} {v['model']} (prior {v['prior']})" for c, v in res["val_iou"].items())
    text = (f"val IoU {ious}; "
            f"weak: {list(an['weak_classes'])}; low scenes: {[x['scene'] for x in an['low_scenes']]}; "
            f"{len(an['suspects'])} suspects to verify")  # fmt: skip
    kb.add("training", f"train_smoke on dataset {idx.get('version')} ({len(scenes)} accepted scenes)", text,
           ["training", *an["weak_classes"]], data=dict(dataset_version=idx.get("version"), dataset_mtime=os.path.getmtime(ds),
                                                        val_iou=res["val_iou"], analysis={k: v for k, v in an.items() if k != "suspects"}),
           area=area.name)  # fmt: skip
    return dict(result="trained", val_iou=res["val_iou"], weak=list(an["weak_classes"]), suspects=len(an["suspects"]),
                minutes=res.get("minutes"))  # fmt: skip
