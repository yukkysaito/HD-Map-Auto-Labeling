"""Experiments: run a candidate tool version in isolation and measure it on the protected benchmark.

A candidate is a config overlay (a registry variant: other model / parameters) and / or other code (a git worktree with
the change). It runs into output/<name>/experiments/<id>/:
  work/drives/<d>/   symlinks to the main run's drive data (read only); paths a changed drive-level tool writes
                     (registry `writes`) are fresh directories, so the main data is never touched
  work/scenes/<s>/   copies of the main scene data + stamps: steps whose code and arguments did not change are skipped
  review/            empty: the benchmark measures raw tool output, without review fixes
The main repository's evaluator (hdlabel/agents/bench.py, locked) measures the result, never the candidate's code.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from hdlabel.common import ROOT

PY = sys.executable


def exp_root(area, exp_id: str) -> str:
    return f"{area.out}/experiments/{exp_id}"


def bench_scenes(area) -> list[str]:
    """the scenes with their own labels (owners): their vectors cover every scene of the area."""
    return sorted({area.owners.get(s["id"], s["id"]) for s in area.scenes()})


def prepare(area, exp_id: str, overlay: dict, scenes: list[str], drive_writes=()) -> str:
    from hdlabel.run import merge

    E = exp_root(area, exp_id)
    if os.path.exists(E):
        shutil.rmtree(E)
    os.makedirs(f"{E}/work/drives")
    os.makedirs(f"{E}/work/scenes")
    os.makedirs(f"{E}/review")
    for d in area.drives:
        src, dst = area.data(d), f"{E}/work/drives/{d}"
        if not os.path.isdir(src):
            continue
        os.makedirs(dst)
        for x in os.listdir(src):
            if any(fnmatch.fnmatch(x, w) for w in drive_writes):
                if os.path.isdir(f"{src}/{x}"):
                    os.makedirs(f"{dst}/{x}")
                continue
            os.symlink(f"{src}/{x}", f"{dst}/{x}")
    for s in scenes:
        if os.path.isdir(area.scene_dir(s)):
            shutil.copytree(area.scene_dir(s), f"{E}/work/scenes/{s}", symlinks=True)
    shutil.copyfile(f"{area.work}/scenes.json", f"{E}/work/scenes.json")
    shutil.copytree(f"{area.work}/stamps", f"{E}/work/stamps")
    for w in drive_writes:  # their stamps must not say "done"
        for st in glob.glob(f"{E}/work/stamps/*.json"):
            if w.rstrip("*").startswith("seg") and os.path.basename(st).startswith("segment_"):
                os.remove(st)
    cfg = merge(area.cfg, overlay)
    absp = lambda p: p and os.path.join(ROOT, os.path.expanduser(str(p)))  # noqa: E731
    # debug on: the steps get the same arguments as in the main run (their stamps match, unchanged steps are skipped)
    cfg.update(name=exp_id, output=f"{area.out}/experiments", review_dir=f"{E}/review", debug=True,
               pointcloud_map=absp(cfg.get("pointcloud_map")), tf_static=absp(cfg.get("tf_static")),
               logs=[absp(x) for x in cfg.get("logs") or []],
               drives={k: dict(v, bags=absp(v["bags"])) for k, v in (cfg.get("drives") or {}).items()})  # fmt: skip
    cfg.setdefault("verification", {})["enabled"] = False
    yaml.safe_dump(cfg, open(f"{E}/experiment.yaml", "w"), sort_keys=False)
    json.dump(dict(id=exp_id, overlay=overlay, scenes=scenes, created=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(f"{E}/experiment.json", "w"), indent=1)  # fmt: skip
    return E


def run(area, exp_id: str, code_dir: str = ROOT, drive_level=False, timeout=6 * 3600) -> dict:
    """label (and ingest / segment when a drive-level tool changed) the experiment's scenes with code_dir's code."""
    E = exp_root(area, exp_id)
    meta = json.load(open(f"{E}/experiment.json"))
    t = time.time()
    log = open(f"{E}/run.log", "w")
    for cmd in (["ingest"] if drive_level else []) + ["label"]:
        r = subprocess.run([PY, f"{code_dir}/hdlabel/run.py", cmd, f"{E}/experiment.yaml", "--scene",
                            *meta["scenes"]], stdout=log, stderr=subprocess.STDOUT, cwd=code_dir, timeout=timeout,
                           env=dict(os.environ, PYTHONPATH=""))  # fmt: skip
        if r.returncode:
            raise RuntimeError(f"experiment {exp_id}: {cmd} failed (see {E}/run.log)")
    meta.update(seconds=round(time.time() - t, 1), code_dir=code_dir)
    json.dump(meta, open(f"{E}/experiment.json", "w"), indent=1)
    return meta


def features(work: str, scenes: list[str]) -> list[dict]:
    out = []
    for s in scenes:
        f = f"{work}/scenes/{s}/vectors/vectors.geojson"
        if os.path.exists(f):
            out += json.load(open(f))["features"]
    return out


def curb_profiles(map_path: str, work: str, scenes: list[str]) -> list[float]:
    """LiDAR step offset of every curb profile of the scenes' vectors (geom_check.curb_offsets), no reference needed."""
    from scipy.spatial import cKDTree

    from hdlabel.steps.geom_check import curb_offsets
    from hdlabel.surface import map_files, read_pcd

    out = []
    for s in scenes:
        f = f"{work}/scenes/{s}/vectors/vectors.geojson"
        g = f"{work}/scenes/{s}/labels/ground_map.npz"
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        m = np.load(g)
        H, lo, sres, origin = m["height"], m["lo"], float(m["surf_res"]), m["origin"]
        c = origin[:2] + lo + np.array(H.shape) * sres / 2
        Q = np.concatenate([read_pcd(x) for x in map_files(map_path, np.array([c]),
                                                             float(np.linalg.norm(np.array(H.shape) * sres) / 2))]).astype(np.float64)  # fmt: skip
        bb0, bb1 = origin[:2] + lo, origin[:2] + lo + np.array(H.shape) * sres
        Q = Q[np.all((Q[:, :2] > bb0) & (Q[:, :2] < bb1), 1)]
        tree = cKDTree(Q[:, :2])
        for ft in json.load(open(f))["features"]:
            if ft["properties"]["kind"] == "curb":
                out += [o for _, o in curb_offsets(ft["geometry"]["coordinates"], Q, tree)]
    return out


def spec_gate(g: dict, feats: list[dict]) -> dict:
    """the benchmark gate plus spec conformance: a candidate whose vectors do not follow spec/labels.yaml (a kind or
    geometry the spec does not define, a missing attribute) is not adopted - the spec is the person's, and data that
    breaks it fails `./autolabel validate`. Attributes the spec does not name are warnings and do not block."""
    from hdlabel import spec

    errs, _ = spec.conform(feats)
    if not errs:
        return g
    more = f" (+{len(errs) - 3} more)" if len(errs) > 3 else ""
    return dict(g, adopt=False, fails=[*g["fails"], f"output breaks spec/labels.yaml: {'; '.join(errs[:3])}{more}"],
                spec_errors=len(errs))  # fmt: skip


def measure(area, exp_id: str) -> dict:
    """benchmark metrics of an experiment (all reference elements and the holdout split) -> experiment dir."""
    from hdlabel.agents import bench

    E = exp_root(area, exp_id)
    meta = json.load(open(f"{E}/experiment.json"))
    feats = features(f"{E}/work", meta["scenes"])
    prof = curb_profiles(area.map, f"{E}/work", meta["scenes"])
    res = dict(all=bench.evaluate(area.name, feats, prof), holdout=bench.evaluate(area.name, feats, prof, split="holdout"),
               seconds=meta.get("seconds"), lock=bench.check_lock(area.name)["data"][:12])  # fmt: skip
    json.dump(res, open(f"{E}/metrics.json", "w"), indent=1)
    return res
