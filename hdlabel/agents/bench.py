"""Protected benchmark: a fixed reference to compare tool versions and to measure the verifier, which the agents
cannot change.

bench/<name>/ (local: holds map coordinates)
  reference.json   reviewed elements {key, kind, group, at, size_m, scene, judges: {judge: decision}, ref: ok | wrong}
                   and the scene split dev / holdout (agents see dev details, holdout only as pass / fail)
  evidence/        the evidence image of every element (frozen copies)
  criteria.yaml    the adoption gate (copied from hdlabel/agents/criteria.yaml when built)
  LOCK.json        sha256 of all of the above + the evaluator code, who locked it and when
Every evaluation checks the lock first; a changed reference / criteria / evaluator stops it. Only a person replaces the
reference (`./autolabel bench build --by NAME`), and the evolve loop rejects candidates that touch protected paths.

Reference = majority of independent judges per element (default: the original reviewer + two other models, of two
vendors); elements without a majority are left out. Its quality is limited by the judges: the person's acceptance page
samples it.

Metrics of a label run (raw vectors, i.e. WITHOUT review fixes, of the bench scenes), per group (marking, curb):
  recall    reference-ok elements (length weighted) that still have an element of the group within tol
  fp        reference-wrong elements (drop / off) that still have one (lower is better)
  lidar     curbs only: share of curb profiles with a LiDAR height step within 0.3 m (geom_check.curb_offsets), over
            all curbs of the run - needs no reference
  length    total length (m) per group
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import shutil

import numpy as np
import yaml

from hdlabel.common import ROOT

GROUP = dict(lane_line="marking", stop_line="marking", arrow="marking", marking_area="marking", crosswalk="marking",
             curb="curb")  # fmt: skip
EVALUATOR = [os.path.join(ROOT, "hdlabel", "agents", f) for f in ("bench.py",)]
PROTECTED = ("bench/", "hdlabel/agents/bench.py", "hdlabel/agents/criteria.yaml", "hdlabel/agents/prompts/verifier.md")
CRITERIA = os.path.join(ROOT, "hdlabel", "agents", "criteria.yaml")


def bench_dir(name: str) -> str:
    return os.path.join(ROOT, "bench", name)


# ---------------------------------------------------------------- lock
def _digest(d: str) -> dict:
    files = [f"{d}/reference.json", f"{d}/criteria.yaml"] + sorted(glob.glob(f"{d}/evidence/*"))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    code = hashlib.sha256(b"".join(open(f, "rb").read() for f in EVALUATOR)).hexdigest()
    return dict(data=h.hexdigest(), evaluator=code, files=len(files))


def lock(name: str, by: str):
    d = bench_dir(name)
    json.dump(dict(_digest(d), locked_by=by, date=datetime.datetime.now().isoformat(timespec="seconds")),
              open(f"{d}/LOCK.json", "w"), indent=1)  # fmt: skip


def check_lock(name: str):
    d = bench_dir(name)
    if not os.path.exists(f"{d}/LOCK.json"):
        raise RuntimeError(f"benchmark {name} is not locked (./autolabel bench lock --by NAME)")
    lk, now = json.load(open(f"{d}/LOCK.json")), _digest(d)
    for k in ("data", "evaluator"):
        if lk[k] != now[k]:
            raise RuntimeError(f"benchmark {name}: {k} changed since it was locked by {lk['locked_by']} on {lk['date']}"
                               " - only a person may re-lock it")  # fmt: skip
    return lk


# ---------------------------------------------------------------- reference
def collect(review_files: list[str], dbg_dir) -> list[dict]:
    """decided elements of the review files (dedup by kind + location): the candidates of the reference."""
    out = []
    for f in review_files:
        scene = os.path.splitext(os.path.basename(f))[0]
        r = json.load(open(f))
        for e in r.get("checked", []) + [x for x in r.get("fixes", []) if x.get("drop")]:
            if not e.get("decision") or not e.get("evidence") or e["kind"] not in GROUP:
                continue
            img = f"{dbg_dir(scene)}/{e['evidence']}"
            if not os.path.exists(img) or any(o["kind"] == e["kind"] and _near(o["at"], e["at"], 0.05) for o in out):
                continue
            out.append(dict(kind=e["kind"], group=GROUP[e["kind"]], at=e["at"], size_m=e.get("size_m") or 0.0,
                            scene=scene, why=e.get("why"), image=img,
                            judges={r.get("reviewer", "reviewer"): e["decision"]}))  # fmt: skip
    for i, e in enumerate(out):
        e["key"] = str(i)
    return out


def _near(a, b, tol):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) < tol


def majority(judges: dict) -> str | None:
    """ok / wrong by majority of the judges' decisions (drop and off = wrong; unsure counts for neither)."""
    v = ["wrong" if d in ("drop", "off") else d for d in judges.values()]
    for k in ("ok", "wrong"):
        if v.count(k) * 2 > len(v):
            return k
    return None


def build(name: str, items: list[dict], judge_fns: dict, by: str, holdout_share=0.35) -> dict:
    """items (collect) + more judges {judge_name: fn(rows) -> {key: {decision}}} -> bench/<name>/ (not locked)."""
    d = bench_dir(name)
    if os.path.exists(f"{d}/LOCK.json") and not by.startswith("person:"):
        raise RuntimeError(f"benchmark {name} is locked; only a person replaces it (--by person:NAME)")
    os.makedirs(f"{d}/evidence", exist_ok=True)
    for j, fn in judge_fns.items():
        res = fn([dict(key=e["key"], image=e["image"], kind=e["kind"], reason=e["why"], size_m=e["size_m"],
                       confidence=None) for e in items])  # fmt: skip
        for e in items:
            r = res.get(e["key"])
            if r and not r.get("error"):
                e["judges"][j] = r["decision"]
    scenes = sorted({e["scene"] for e in items})
    hold = [s for s in scenes if int(hashlib.sha1(s.encode()).hexdigest(), 16) % 100 < holdout_share * 100]
    ref = []
    for e in items:
        m = majority(e["judges"])
        ev = f"evidence/{e['kind']}_{e['at'][0]:.2f}_{e['at'][1]:.2f}.jpg"
        shutil.copyfile(e["image"], f"{d}/{ev}")
        ref.append(dict({k: v for k, v in e.items() if k != "image"}, evidence=ev, ref=m,
                        split="holdout" if e["scene"] in hold else "dev"))  # fmt: skip
    out = dict(name=name, built=datetime.date.today().isoformat(), by=by, judges=sorted({j for e in items for j in e["judges"]}),
               items=ref, holdout_scenes=hold)  # fmt: skip
    json.dump(out, open(f"{d}/reference.json", "w"), indent=1)
    shutil.copyfile(CRITERIA, f"{d}/criteria.yaml")
    return out


# ---------------------------------------------------------------- evaluation
def _dist(f: dict, p) -> float:
    from shapely.geometry import Point, shape

    g = shape({"type": f["geometry"]["type"], "coordinates": f["geometry"]["coordinates"]})
    return float(g.distance(Point(p)))


def evaluate(name: str, features: list[dict], curb_profiles: list[float] | None = None, tol=0.5,
             split: str | None = None) -> dict:  # fmt: skip
    """metrics (module doc) of a run's raw vectors (all bench scenes merged, map frame) against the reference."""
    check_lock(name)
    ref = json.load(open(f"{bench_dir(name)}/reference.json"))
    by_group = {}
    for f in features:
        g = GROUP.get(f["properties"]["kind"])
        if g:
            by_group.setdefault(g, []).append(f)
    out = {}
    for g in ("marking", "curb"):
        fs = by_group.get(g, [])
        res = dict(ok=0, ok_kept=0, ok_m=0.0, ok_kept_m=0.0, wrong=0, wrong_kept=0, wrong_m=0.0, wrong_kept_m=0.0)
        for e in ref["items"]:
            if e["group"] != g or e["ref"] is None or (split and e["split"] != split):
                continue
            kept = any(_dist(f, e["at"]) <= tol for f in fs)
            k = e["ref"]
            res[k] += 1
            res[f"{k}_m"] += e["size_m"]
            res[f"{k}_kept"] += kept
            res[f"{k}_kept_m"] += e["size_m"] * kept
        res["recall"] = round(res["ok_kept_m"] / res["ok_m"], 4) if res["ok_m"] else None
        res["fp"] = round(res["wrong_kept_m"] / res["wrong_m"], 4) if res["wrong_m"] else None
        res["length"] = round(sum(_length(f) for f in fs), 1)
        res["elements"] = len(fs)
        out[g] = {k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}
    if curb_profiles is not None:
        cp = np.array(curb_profiles)
        out["curb"]["lidar"] = round(float((cp <= 0.3).mean()), 4) if len(cp) else None
        out["curb"]["lidar_profiles"] = int(len(cp))
    return out


def _length(f) -> float:
    c = np.array(
        f["geometry"]["coordinates"][0] if f["geometry"]["type"] == "Polygon" else f["geometry"]["coordinates"]
    )
    c = c.reshape(-1, c.shape[-1])[:, :2]
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0


def metric(m: dict, path: str):
    g, k = path.split(".")
    return (m.get(g) or {}).get(k)


def gate(name: str, base: dict, cand: dict, target: str) -> dict:
    """adoption decision from criteria.yaml: no metric regresses beyond its tolerance (on all reference items and on
    the holdout split), and the target metric improves by at least min_improvement. base / cand: {all, holdout}."""
    check_lock(name)
    cr = yaml.safe_load(open(f"{bench_dir(name)}/criteria.yaml"))
    fails, rows = [], []
    for path, tol in cr["no_regression"].items():
        higher = not path.endswith(".fp")
        for sp in ("all", "holdout"):
            b, c = metric(base[sp], path), metric(cand[sp], path)
            if b is None or c is None:
                continue
            delta = (c - b) if higher else (b - c)
            rows.append(dict(metric=path, split=sp, base=b, cand=c, better=round(delta, 4)))
            if delta < -tol:
                fails.append(f"{path} ({sp}) regressed: {b} -> {c} (tolerance {tol})")
    higher = not target.endswith(".fp")
    b, c = metric(base["all"], target), metric(cand["all"], target)
    gain = None if b is None or c is None else round((c - b) if higher else (b - c), 4)
    if gain is None or gain < cr["min_improvement"]:
        fails.append(f"target {target} improved by {gain} < {cr['min_improvement']}")
    return dict(adopt=not fails, fails=fails, target=target, gain=gain, rows=rows)


def verifier_accuracy(name: str, results: dict, judge: str) -> dict:
    """a judge's decisions {key: {decision}} on the reference elements vs the reference. When the judge is one of the
    reference's own judges, it is compared with the majority of the OTHER judges (leave-one-out)."""
    check_lock(name)
    ref = json.load(open(f"{bench_dir(name)}/reference.json"))
    conf, per = {}, {}
    for e in ref["items"]:
        others = {k: v for k, v in e["judges"].items() if k != judge}
        truth = majority(others) if judge in e["judges"] else e["ref"]
        r = results.get(e["key"])
        if truth is None or r is None:
            continue
        d = r["decision"]
        got = "wrong" if d in ("drop", "off") else d
        conf[f"{truth}->{got}"] = conf.get(f"{truth}->{got}", 0) + 1
        s = per.setdefault(e["group"], dict(n=0, correct=0, unsure=0))
        s["n"] += 1
        s["correct"] += got == truth
        s["unsure"] += got == "unsure"
    n = sum(s["n"] for s in per.values())
    tp, fn = conf.get("wrong->wrong", 0), conf.get("wrong->ok", 0) + conf.get("wrong->unsure", 0)
    fp = conf.get("ok->wrong", 0)
    return dict(judge=judge, n=n, accuracy=round(sum(s["correct"] for s in per.values()) / n, 4) if n else None,
                wrong_recall=round(tp / (tp + fn), 4) if tp + fn else None,
                wrong_precision=round(tp / (tp + fp), 4) if tp + fp else None,
                per_group={g: dict(s, accuracy=round(s["correct"] / s["n"], 4)) for g, s in per.items()},
                confusion=conf)  # fmt: skip
