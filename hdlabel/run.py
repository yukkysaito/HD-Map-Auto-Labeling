"""hdlabel: HD-map training data (road markings incl. turn arrows, lane lines, crosswalks, curbs, drivable area) from
rosbags (cameras, localization, official calibration) and a LiDAR point cloud map. No HD / vector map is used.

usage: ./autolabel COMMAND [CONFIG.yaml] [--map PCD] [--logs GLOB ...] [--tf-static JSON] [--name NAME] [--output DIR]
                   [--drive NAME ...] [--scene ID ...] [--force] [--no-debug] [--jobs N]
  e.g. ./autolabel run --map /data/pointcloud_map.pcd --logs "/data/logs/*" --name my_area
       ./autolabel run my_area.yaml                (settings: configs/default.yaml, overridden by the given config)

  run      everything below in order; finished work is skipped, failed items are retried, one failing bag / drive /
           scene does not stop the others (output/<name>/status.json lists what failed and its log)
  ingest   bags -> work/drives/<drive>: undistorted images, poses, calibration, keyframes, data checks, segmentation
  scenes   split every drive into scenes (scenes.length_m of driving; labels cover the route +- scenes.context_m)
  calib    per drive: official extrinsics + image time offsets (on the map surface of its most-turning scenes)
  label    per scene: road surface (multi-level safe), labels from the cameras of all drives on that level, turn
           arrows, vectors (+ review fixes)
  export   per scene: training data -> dataset/<scene>, then dataset/dataset.json (train / val by location)
  debug    per scene: debug images (review sheets, previews), PCDs and the LiDAR geometry check -> debug/<scene>
           (part of run unless debug: false in the config or --no-debug)
  qa       repeatability: scenes seen by two drives labelled from each drive alone, compared (no HD map)
  status   what is done / failed
  review   scenes whose evidence pack (debug/<scene>/check) awaits a reviewer; with --scene ID --decisions FILE:
           apply the reviewer's decisions to review/<scene>.json (docs/AGENT_REVIEW.md)
  signoff  --signoff FILE: record the person's acceptance saved by debug/audit.html (rejected scenes leave the data)
  prune    delete undistorted frames neither the dataset nor labelling uses (automatic in run with prune_frames: true)
Outputs (one root, config `output`, default output/<name>/):
  dataset/   training data only (docs/DATASET.md)      debug/   images, PCDs, logs, QA (can be deleted / disabled)
  work/      intermediate data for resuming (drives: images, poses, segmentation, calibration; scenes: surface, labels)
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdlabel.common import ROOT, load_tf_static, load_yaml

PY = sys.executable
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.yaml")
STEPS = os.path.join(ROOT, "hdlabel", "steps")
EVAL = os.path.join(ROOT, "hdlabel", "eval")


AGENT_COMMANDS = ("kb", "tools", "bench", "experiment", "impact", "evolve", "train", "spec")

# config keys a project config replaces as a whole (a camera list is not merged with the default one)
REPLACE = ("cameras",)


def merge(base: dict, over: dict) -> dict:
    """deep merge: keys of over replace those of base (dicts are merged key by key, except REPLACE keys)."""
    out = dict(base)
    for k, v in (over or {}).items():
        deep = isinstance(v, dict) and isinstance(base.get(k), dict) and k not in REPLACE
        out[k] = merge(base[k], v) if deep else v
    return out


def load_config(path=None, registry=None, **override) -> dict:
    """configs/default.yaml <- active tool variants (tools/registry.yaml, or the given registry dict) <- the project
    config (path) <- command line values (not None)."""
    from hdlabel.agents import registry as reg

    c = merge(load_yaml(DEFAULT_CONFIG), reg.overlay(registry))
    c = merge(c, load_yaml(path) if path else {})
    return merge(c, {k: v for k, v in override.items() if v is not None})


class Area:
    """one output root: a point cloud map + the drives recorded in it (config: configs/default.yaml)."""

    def __init__(self, cfg, debug=None):
        self.cfg = c = cfg if isinstance(cfg, dict) else load_config(cfg)
        self.name = c["name"]
        path = lambda p: os.path.join(ROOT, os.path.expanduser(str(p)))  # relative: to the repository
        self.map = c.get("pointcloud_map") and path(c["pointcloud_map"])
        assert "lanelet" not in (self.map or ""), "pointcloud_map must be the point cloud map, never the lanelet2 map"
        self.out = path(os.path.join(c.get("output", "output"), self.name))
        self.work, self.dataset, self.dbg = f"{self.out}/work", f"{self.out}/dataset", f"{self.out}/debug"
        self.debug = c.get("debug", True) if debug is None else debug
        self.review = path(c.get("review_dir") or f"review/{self.name}")
        self.tf_static = c.get("tf_static") and path(c["tf_static"])
        os.makedirs(f"{self.dbg}/logs", exist_ok=True)
        os.makedirs(f"{self.work}/stamps", exist_ok=True)
        self.sensors = json.dumps(
            {k: c.get(k) for k in ("topics", "base_frame", "cameras", "image_scale", "jpeg_quality")}, sort_keys=True
        )  # ingest's settings: part of its arguments (changed -> ingested again)
        self.drives = {}
        for name, d in (c.get("drives") or {}).items():
            self.drives[name] = dict(bags=sorted(glob.glob(path(d["bags"]))), tf_from=d.get("tf_from"))
        stamp_name = lambda p: (lambda st: "".join(st.groups()[:3]) + "-" + "".join(st.groups()[3:]) if st else None)(
            re.search(r"(\d{4})-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)", os.path.basename(p))
        )
        for pat in c.get("logs") or []:  # a directory or an .mcap file = one drive
            for x in sorted(glob.glob(path(pat))):
                bags = sorted(glob.glob(f"{x}/*.mcap")) if os.path.isdir(x) else [x] if x.endswith(".mcap") else []
                if bags:
                    self.drives.setdefault(
                        stamp_name(x) or os.path.splitext(os.path.basename(x))[0], dict(bags=bags, tf_from=None)
                    )
        self.owners = {}
        self.lock = threading.Lock()
        self.gpu = threading.Semaphore(c.get("jobs", {}).get("gpu", 1))
        self.status = json.load(open(f"{self.out}/status.json")) if os.path.exists(f"{self.out}/status.json") else {}
        from hdlabel.agents.llm import Agents

        self.agents = Agents(c.get("agents", {}), f"{self.out}/agent")  # models of the agent roles, cache, ledger

    def data(self, d):
        return f"{self.work}/drives/{d}"

    def calib(self, d):
        return f"{self.work}/drives/{d}/calib"

    def scene_dir(self, s):
        return f"{self.work}/scenes/{s}"

    def ds_dir(self, s):
        return f"{self.dataset}/{s}"

    def dbg_dir(self, s):
        return f"{self.dbg}/scenes/{s}"

    def review_file(self, s):
        return f"{self.review}/{s}.json"

    def reviews(self):
        return sorted(glob.glob(f"{self.review}/*.json"))  # fixes in map coordinates: all apply

    def scenes(self):
        return (
            json.load(open(f"{self.work}/scenes.json"))["scenes"] if os.path.exists(f"{self.work}/scenes.json") else []
        )

    def record(self, stage, item, ok, log=None, note=None):
        with self.lock:
            self.status.setdefault(stage, {})[item] = dict(
                ok=bool(ok), time=time.strftime("%Y-%m-%d %H:%M:%S"), log=log, note=note
            )
            json.dump(self.status, open(f"{self.out}/status.json", "w"), indent=1)


def mtime(f):
    return os.path.getmtime(f) if f and os.path.exists(f) else 0


_CODE = {}


def _hdlabel_imports(path: str) -> list[str]:
    """files of the hdlabel modules a file imports (from hdlabel.x / from .x / import hdlabel.x)."""
    import ast

    pkg, out = os.path.join(ROOT, "hdlabel"), []
    for node in ast.walk(ast.parse(open(path).read())):
        names = []
        if isinstance(node, ast.ImportFrom):
            mod = node.module if node.level == 0 else f"hdlabel.{node.module}" if node.module else "hdlabel"
            names = [mod] + [f"{mod}.{n.name}" for n in node.names]  # from hdlabel import x: x may be a module
        elif isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        for n in names:
            if n.startswith("hdlabel."):
                f = os.path.join(pkg, *n.split(".")[1:]) + ".py"
                if os.path.exists(f):
                    out.append(f)
    return out


def code_version(script: str) -> str:
    """hash of a step's code: the script, the hdlabel modules it imports (transitively) and the assets. A change of
    any of them redoes the step (the stamp key includes it); unrelated modules do not."""
    if script not in _CODE:
        files, todo = set(), [script]
        while todo:
            f = todo.pop()
            if f not in files:
                files.add(f)
                todo += _hdlabel_imports(f)
        h = hashlib.sha1()
        for f in sorted(files) + sorted(glob.glob(os.path.join(ROOT, "hdlabel", "assets", "*"))):
            h.update(open(f, "rb").read())
        _CODE[script] = h.hexdigest()
    return _CODE[script]


def class_map(area) -> str:
    """config segmentation.class_map as JSON ('' = the default Vistas ids of common.GROUPS)."""
    cm = area.cfg.get("segmentation", {}).get("class_map")
    return json.dumps(cm, sort_keys=True) if cm else ""


DRY_RUN = None  # a list: step() only records (tag, script) of the steps that would run
DRY_RUN_OUT = set()  # their outputs (steps reading them would run too)


def step(area, tag, script, args, outputs, inputs=(), gpu=False, force=False):
    """run hdlabel/steps|eval/<script> with args unless done: every output exists, newer than every input, and the
    arguments and code are those of the last successful run (work/stamps/<tag>.json). Logs to debug/logs/<tag>.log."""
    args = [str(x) for x in args]
    stamp = f"{area.work}/stamps/{tag}.json"
    rel = [a.replace(area.out, "<OUT>") for a in args]  # keys do not depend on where the output root is (experiments)
    key = [os.path.relpath(script, ROOT), rel, code_version(script)] + ([class_map(area)] if class_map(area) else [])
    key = hashlib.sha1(json.dumps(key).encode()).hexdigest()
    old = [script, args, code_version(script)] + ([class_map(area)] if class_map(area) else [])  # stamps before 0.4
    old = hashlib.sha1(json.dumps(old).encode()).hexdigest()
    done = (
        not force
        and os.path.exists(stamp)
        and json.load(open(stamp)).get("key") in (key, old)
        and all(map(os.path.exists, outputs))
        and min(map(mtime, outputs)) >= max([mtime(i) for i in inputs] + [0])
    )
    if DRY_RUN is not None:  # impact analysis: record what would run (and what follows from its outputs)
        if not done or any(i in DRY_RUN_OUT for i in inputs):
            DRY_RUN.append((tag, os.path.relpath(script, ROOT)))
            DRY_RUN_OUT.update(outputs)
        return True
    if done:
        if json.load(open(stamp)).get("key") != key:
            json.dump(dict(json.load(open(stamp)), key=key), open(stamp, "w"))
        return True
    log = f"{area.dbg}/logs/{tag}.log"
    with area.gpu if gpu else _null(), open(log, "w") as f:
        t = time.time()
        f.write("$ " + " ".join([script, *args]) + "\n")
        f.flush()
        env = dict(os.environ, HDLABEL_DEVICE=str(area.cfg.get("device") or "auto"), HDLABEL_CLASS_MAP=class_map(area))
        r = subprocess.run([PY, script, *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    lines = open(log).read().strip().splitlines()
    if r.returncode == 0:
        json.dump(dict(key=key, args=args, done=time.time()), open(stamp, "w"))
    elif os.path.exists(stamp):
        os.remove(stamp)
    print(
        f"  {tag}: {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t:.0f} s) "
        f"{' | '.join(lines[-1:] if r.returncode == 0 else lines[-4:])[:200]}",
        flush=True,
    )
    if r.returncode:
        raise StepError(f"{tag} failed", log)
    return True


class StepError(Exception):
    def __init__(self, msg, log):
        super().__init__(msg)
        self.log = log


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------- per drive
def ingest(area, d, force):
    """bags -> work/drives/<d> (images, poses, calibration, keyframes, checks), then segmentation of the keyframes."""
    D = area.data(d)
    info = area.drives[d]
    if not info["bags"]:
        raise StepError(f"{d}: no bags found", None)
    pr = f"{D}/pruned.json"  # frames pruned for another export: extract again
    redo = os.path.exists(pr) and json.load(open(pr)).get("export") != export_key(area)
    step(
        area,
        f"ingest_{d}",
        f"{STEPS}/ingest.py",
        ["--out", D, "--sensors", area.sensors, *info["bags"]],
        [f"{D}/ingest.json"],
        info["bags"],
        force=force or redo,
    )
    ing = json.load(open(f"{D}/ingest.json"))
    if not ing.get("cameras"):
        return "no cameras in the bags (not labelled)"
    if not os.path.exists(f"{D}/tf_static.json"):  # filtered bags: drive's tf_from, the config's file, another drive
        src = (
            info["tf_from"]
            or area.tf_static
            or next((o for o in area.drives if o != d and os.path.exists(f"{area.data(o)}/tf_static.json")), None)
        )
        is_file = src and os.path.splitext(src)[1] in (".json", ".yaml", ".yml", ".mcap")
        f = src and (os.path.join(ROOT, os.path.expanduser(src)) if is_file else f"{area.data(src)}/tf_static.json")
        if not f or not os.path.exists(f):
            raise StepError(f"{d}: no /tf_static in the bags and none configured (config tf_static)", None)
        tfb = load_tf_static(
            f, area.cfg.get("base_frame", "base_link"), area.cfg["topics"].get("tf_static", "/tf_static")
        )
        if not tfb:
            raise StepError(f"{d}: {f} has no frame chained to {area.cfg.get('base_frame', 'base_link')}", None)
        json.dump(tfb, open(f"{D}/tf_static.json", "w"), indent=1)
        print(f"  {d}: tf_static from {os.path.relpath(f, ROOT)}")
    step(
        area,
        f"segment_{d}",
        f"{STEPS}/segment.py",
        ["--data", D, "--model", area.cfg["segmentation"]["model"]],
        [f"{D}/seg_done.json"],
        [f"{D}/ingest.json"],
        gpu=True,
        force=force,
    )
    issues = ing.get("issues") or {}
    return (
        ", ".join(
            f"{k}: {v if not isinstance(v, list) or len(v) < 8 else f'{len(v)} items'}" for k, v in issues.items() if v
        )
        or None
    )


def tile_runs(xy, s, L, min_m=30.0):
    """pose indices of a drive cut where it crosses the lines of a fixed L x L map grid (runs inside one grid cell);
    pieces shorter than min_m driven are merged into the previous run. Drives over the same road get the same cuts,
    whatever their start, so their scenes coincide and share labels (label_owners)."""
    cell = np.floor(xy / L).astype(np.int64)
    cut = np.flatnonzero(np.any(np.diff(cell, axis=0) != 0, 1)) + 1
    runs = np.split(np.arange(len(xy)), cut)
    out = []
    for r in runs:
        if out and (s[r[-1]] - s[r[0]] < min_m or s[out[-1][-1]] - s[out[-1][0]] < min_m):
            out[-1] = np.r_[out[-1], r]
        else:
            out.append(r)
    return out


def split_scenes(area):
    """scenes = runs of a drive inside one cell of a fixed length_m grid (tile_runs); t_range = the samples' time range, bbox = the route +- context_m around it
    (so every sample's BEV ahead / behind is labelled) + corridor_m. Optional scenes.region (map box): only scenes whose
    path centre is inside."""
    sc = area.cfg["scenes"]
    L, ctx, cor = sc["length_m"], sc.get("context_m", 100), sc.get("corridor_m", 40)
    reg = (
        np.array(sc["region"], float) if sc.get("region") else None
    )  # [xmin, ymin, xmax, ymax]: only scenes centred inside
    scenes = []
    for d in area.drives:
        ing = f"{area.data(d)}/ingest.json"
        if (
            not os.path.exists(ing)
            or not json.load(open(ing)).get("cameras")
            or not os.path.exists(f"{area.data(d)}/gt.npy")
        ):
            continue
        g = np.load(f"{area.data(d)}/gt.npy")
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g[:, 1:3], axis=0), axis=1))]
        yaw = np.unwrap(2 * np.arctan2(g[:, 6], g[:, 7]))
        for k, idx in enumerate(tile_runs(g[:, 1:3], s, L)):
            if s[idx[-1]] - s[idx[0]] < 20:
                continue  # standing still
            if reg is not None and not np.all((g[idx, 1:3].mean(0) >= reg[:2]) & (g[idx, 1:3].mean(0) <= reg[2:])):
                continue
            c = np.flatnonzero((s >= s[idx[0]] - ctx) & (s <= s[idx[-1]] + ctx))
            xy = g[c, 1:3]
            scenes.append(
                dict(
                    id=f"{d}_{k:03d}",
                    drive=d,
                    t_range=[int(g[idx[0], 0]), int(g[idx[-1], 0])],
                    route_t=[int(g[c[0], 0]), int(g[c[-1], 0])],
                    path_m=round(float(s[idx[-1]] - s[idx[0]]), 1),
                    turn_deg=round(float(np.degrees(np.abs(np.diff(yaw[idx])).sum())), 1),
                    bbox=np.round(np.r_[xy.min(0) - cor, xy.max(0) + cor], 1).tolist(),
                )
            )
    new = dict(length_m=L, context_m=ctx, corridor_m=cor, scenes=scenes)
    f = f"{area.work}/scenes.json"
    if not os.path.exists(f) or json.load(open(f)) != new:
        json.dump(new, open(f, "w"), indent=1)  # unchanged: keep its time
    ids = {s["id"] for s in scenes}
    gone = 0  # outputs of scenes that no longer exist
    for d in glob.glob(f"{area.work}/scenes/*") + glob.glob(f"{area.dataset}/*/") + glob.glob(f"{area.dbg}/scenes/*"):
        if os.path.isdir(d) and os.path.basename(d.rstrip("/")) not in ids:
            shutil.rmtree(d)
            gone += 1
    print(
        f"  {len(scenes)} scenes from {len(area.drives)} drives"
        + (f" ({gone} outputs of former scenes removed)" if gone else "")
    )


def calibrate(area, d, force):
    C = area.calib(d)
    D = area.data(d)
    sc = [s for s in area.scenes() if s["drive"] == d]
    if not sc:
        return "no scenes (no cameras or did not drive 20 m)"
    fixed = area.cfg.get("calib", {}).get("time_offset", "estimate")
    top = (
        []
        if fixed != "estimate"
        else sorted(sc, key=lambda s: -s["turn_deg"])[: area.cfg.get("calib", {}).get("scenes", 3)]
    )
    win = [] if fixed == "estimate" else ["--fixed", float(fixed)]
    for s in top:
        G = f"{area.work}/drives/{d}/offset_surface/{s['id']}"
        step(
            area,
            f"calib_surface_{s['id']}",
            f"{STEPS}/map_surface.py",
            ["--map", area.map, "--gt", f"{D}/gt.npy", "--out", G, "--bbox", *s["bbox"]],
            [f"{G}/map_ground.npz"],
            [f"{D}/gt.npy"],
            force=force,
        )
        win += [
            "--window",
            f"{G}/map_ground.npz",
            *s["t_range"],
        ]  # the turning part itself: context frames blunt the curves
    step(
        area,
        f"calib_{d}",
        f"{STEPS}/time_offset.py",
        [*win, "--data", D, "--calib", C],
        [f"{C}/time_offset.json"],
        [f"{D}/ingest.json", f"{D}/tf_static.json"],
        gpu=True,
        force=force,
    )
    st = json.load(open(f"{C}/time_offset.json"))["status"]
    weak = {c: v for c, v in st.items() if v not in ("fine", "reference", "fixed")}
    if not weak:
        return None
    return f"time offsets not fine: {weak}" + (
        " - a camera FAILED: this drive's samples fail QA" if "failed" in weak.values() else ""
    )


# ---------------------------------------------------------------- per scene
def scene_drives(area, s):
    """calibrated drives that drove inside the scene's box ON THE SAME ROAD LEVEL as the scene's route (points within
    30 m of the route and > 2.5 m above / below it are another level: bridge, road under a deck): its own drive first,
    then the others by number of same-level points, at most fusion.max_drives. With fusion.max_days_apart, drives
    recorded more days apart than that from the scene's drive are not fused (repainted / rebuilt roads)."""
    bb = np.array(s["bbox"])
    g0 = np.load(f"{area.data(s['drive'])}/gt.npy")
    days = area.cfg.get("fusion", {}).get("max_days_apart")
    ref = g0[(g0[:, 0] >= s["route_t"][0]) & (g0[:, 0] <= s["route_t"][1]), 1:4]
    from scipy.spatial import cKDTree

    tree = cKDTree(ref[:, :2])
    count = {}
    for d in area.drives:
        if not os.path.exists(f"{area.calib(d)}/time_offset.json"):
            continue
        g = np.load(f"{area.data(d)}/gt.npy")
        if days is not None and abs(g[0, 0] - g0[0, 0]) > days * 86400e9:
            continue
        g = g[:, 1:4]
        g = g[np.all((g[:, :2] > bb[:2]) & (g[:, :2] < bb[2:]), 1)]
        if not len(g):
            continue
        dist, i = tree.query(g[:, :2])
        n = int(((dist < 30) & (np.abs(g[:, 2] - ref[i, 2]) < 2.5)).sum())
        if n:
            count[d] = n
    order = sorted(count, key=lambda d: (d != s["drive"], -count[d]))
    return order[: area.cfg.get("fusion", {}).get("max_drives", 5)]


def label_owners(area, scenes):
    """scene id -> id of the scene whose labels it uses. A scene shares the labels of an earlier scene (of another drive)
    that fuses the same drives and whose box contains its box (scenes.share_tolerance_m, default 20 m: the edge of the
    context, 150+ m ahead): repeated drives of a road are labelled once, every drive still gets its own samples."""
    own, done = {}, []
    for s in scenes:
        dr = set(scene_drives(area, s))
        bb = np.array(s["bbox"])
        tol = area.cfg["scenes"].get("share_tolerance_m", 20)
        o = next(
            (
                t
                for t, tdr, tbb in done
                if tdr == dr and np.all(tbb[:2] <= bb[:2] + tol) and np.all(tbb[2:] >= bb[2:] - tol)
            ),
            None,
        )
        own[s["id"]] = o or s["id"]
        if o is None:
            done.append((s["id"], dr, bb))
    return own


def marking_args(area) -> list:
    """vectorize.py options from the config `markings` (regional conventions)."""
    mk, st, fr = area.cfg["markings"], area.cfg["markings"]["stop_line"], area.cfg["markings"]["fragments"]
    return ["--line-width", mk["line_width"], "--stop-width", st["min_width"], "--stop-length", st["min_length"],
            "--stop-cos", st["max_cos"], "--frag-conf", fr["min_confidence"], "--frag-length", fr["short_length"],
            "--frag-views", fr["short_min_views"]]  # fmt: skip


def projection_args(area) -> list:
    """project_labels.py options from the config `projection` and `fusion`."""
    pr = area.cfg.get("projection") or {}
    args = ["--mode", pr.get("mode", "splat")]
    if pr.get("mode") == "gather":
        args += ["--power", pr.get("power", 2.0)]
        args += ["--rgb-power", pr["rgb_power"]] if pr.get("rgb_power") else []
    shift = (area.cfg.get("fusion") or {}).get("max_shift_m")
    pt = pr.get("paint") or {}
    if pt.get("enabled"):
        from hdlabel.paint import DEFAULTS

        args += ["--paint", *[pt.get(k, v) for k, v in DEFAULTS.items()]]
    cw = pt.get("crosswalk") or {}
    if cw.get("enabled"):
        from hdlabel.paint import XWALK_DEFAULTS

        args += ["--paint-crosswalk", *[cw.get(k, v) for k, v in XWALK_DEFAULTS.items()]]
    return args + (["--max-shift", shift] if shift else [])


def curb_args(area, gts) -> list:
    """vectorize.py curb filter options from the config `markings.curb_filter` (none when it is off: unchanged)."""
    cf = area.cfg["markings"].get("curb_filter") or {}
    if not cf.get("enabled"):
        return []
    return ["--curb-filter", "--gt", *gts, "--curb-adjacent", cf["adjacent_m"], "--curb-min-adjacent",
            cf["min_adjacent"], "--curb-min-first", cf["min_first_edge"], "--curb-wavy", cf["max_wavy_deg_per_m"]] \
        + (["--curb-rules", *cf["rules"]] if cf.get("rules") else [])  # fmt: skip


def marking_gate_args(area, gts) -> list:
    """vectorize.py marking gate options from the config `markings.marking_gate` (none when it is off: unchanged);
    the trajectories (--gt) only when the curb filter does not pass them already."""
    mg = area.cfg["markings"].get("marking_gate") or {}
    if not mg.get("enabled"):
        return []
    gt = [] if (area.cfg["markings"].get("curb_filter") or {}).get("enabled") else ["--gt", *gts]
    return ["--marking-gate", *gt, "--mark-tol", mg["tol_m"], "--mark-min-level", mg["min_on_level"], "--mark-step",
            mg["step_m"], "--mark-ref", mg["ref_m"]] + (["--mark-rules", *mg["rules"]] if mg.get("rules") else []) + (
            ["--mark-kerb", mg["kerb_m"]] if mg.get("kerb_m") else [])  # fmt: skip


def fixes_file(area) -> str:
    """work/fixes.json: the fixes (deletions / changes) of all review files in one file, rewritten only when they
    change - the input of vectorize, so a new 'ok' decision does not redo every scene's vectors."""
    from hdlabel.agents.verify import save_json

    f = f"{area.work}/fixes.json"
    with area.lock:
        fx = [x for r in area.reviews() for x in json.load(open(r)).get("fixes", [])]
        save_json(f, dict(fixes=fx))
    return f


def label(area, s, force):
    sid = s["id"]
    owner = area.owners.get(sid, sid)
    if owner != sid:  # same drives, inside the owner's box: shared labels
        shutil.rmtree(area.scene_dir(sid), ignore_errors=True)  # own labels of an earlier run would be stale
        return f"labels shared with {owner}"
    S = area.scene_dir(sid)
    Dg = area.dbg_dir(sid) if area.debug else None
    drives = scene_drives(area, s)
    gts = [f"{area.data(d)}/gt.npy" for d in drives]
    dbg = lambda *x: ["--debug", Dg, *x] if Dg else []
    step(
        area,
        f"surface_{sid}",
        f"{STEPS}/map_surface.py",
        [
            "--map",
            area.map,
            "--out",
            f"{S}/surface",
            "--bbox",
            *s["bbox"],
            "--gt",
            *gts,
            "--t-range",
            *s["route_t"],
            *dbg(),
        ],
        [f"{S}/surface/map_ground.npz"],
        gts,
        force=force,
    )
    # annotation plan: the tool variants for this scene (registry `when` conditions on the scene's facts)
    from hdlabel.agents import annotate, registry

    reg = registry.load()
    fx = annotate.facts(area, s, drives)
    chosen = annotate.plan(reg, fx)
    json.dump(dict(facts=fx, tools=chosen), open(f"{S}/plan.json", "w"), indent=1)
    ov = annotate.overlay(reg, chosen)
    if ov:  # another variant than the active one for this scene: its settings
        area = copy.copy(area)
        area.cfg = merge(area.cfg, ov)
    drv = sum((["--drive", area.data(d), area.calib(d)] for d in drives), [])
    calibs = [f"{area.calib(d)}/time_offset.json" for d in drives]
    step(
        area,
        f"labels_{sid}",
        f"{STEPS}/project_labels.py",
        ["--ground", f"{S}/surface/map_ground.npz", "--out", f"{S}/labels", *drv, *projection_args(area), *dbg()],
        [f"{S}/labels/ground_map.npz"],
        [f"{S}/surface/map_ground.npz", *calibs],
        gpu=True,
        force=force,
    )
    adds = [f for f in area.reviews() if '"add"' in open(f).read()]  # review files adding missed arrows
    step(
        area,
        f"arrows_{sid}",
        f"{STEPS}/arrows.py",
        [
            "--labels",
            f"{S}/labels",
            *drv,
            *(["--fixes", *adds] if adds else []),
            "--templates",
            area.cfg["markings"]["arrow_templates"],
            *dbg(),
        ],
        [f"{S}/labels/arrows.json"],
        [f"{S}/labels/ground_map.npz", *adds],
        force=force,
    )  # CPU-bound (image warps, matching): not GPU-limited
    step(
        area,
        f"vectors_{sid}",
        f"{STEPS}/vectorize.py",
        [
            f"{S}/labels",
            "--out",
            f"{S}/vectors",
            "--fixes",
            fixes_file(area),
            *marking_args(area),
            *curb_args(area, gts),
            *marking_gate_args(area, gts),
            *dbg(),
        ],
        [f"{S}/vectors/vectors.geojson"],
        [f"{S}/labels/ground_map.npz", f"{S}/labels/arrows.json", fixes_file(area)],
        force=force,
    )
    return f"drives {drives}"


def export_key(area):
    ds = area.cfg["dataset"]
    return [ds["every_s"], ds.get("val_every_s"), ds["history_s"], ds.get("history_dt_s", 0.5)]


def export(area, s, force):
    S = area.scene_dir(area.owners.get(s["id"], s["id"]))
    ds = area.cfg["dataset"]
    rv = area.review_file(s["id"])
    bev = ds["bev"]
    every = (
        (ds.get("val_every_s") or ds["every_s"]) if scene_split(area, s)[0] != "train" else ds["every_s"]
    )  # val / test: model cycle
    step(
        area,
        f"export_{s['id']}",
        f"{STEPS}/export.py",
        [
            "--labels",
            f"{S}/labels",
            "--vectors",
            f"{S}/vectors/vectors.geojson",
            "--data",
            area.data(s["drive"]),
            "--calib",
            area.calib(s["drive"]),
            "--out",
            area.ds_dir(s["id"]),
            "--t-range",
            *s["t_range"],
            "--ref",
            area.cfg.get("reference_camera", list(area.cfg["cameras"])[0]),
            "--every",
            every,
            "--history",
            ds["history_s"],
            "--history-dt",
            ds.get("history_dt_s", 0.5),
            "--x",
            *bev["x"],
            "--y",
            *bev["y"],
            "--res",
            bev["res"],
            "--review",
            rv,
            *(["--debug", area.dbg_dir(s["id"])] if area.debug else []),
        ],
        [f"{area.ds_dir(s['id'])}/scene.json"],
        [f"{S}/vectors/vectors.geojson", f"{S}/labels/arrows.json", rv],
        force=force,
    )
    sc = json.load(open(f"{area.ds_dir(s['id'])}/scene.json"))
    return f"{sc['qa_ok']} / {sc['samples']} samples ok"


def debug(area, s, force):
    """debug images / PCDs of a scene (review sheets, geometric check against the LiDAR map)."""
    o = area.owners.get(s["id"], s["id"])
    D, S, Do = area.dbg_dir(s["id"]), area.scene_dir(o), area.dbg_dir(o)
    ds = area.ds_dir(s["id"])
    step(
        area,
        f"review_{s['id']}",
        f"{STEPS}/review.py",
        [
            "--dataset",
            ds,
            "--labels",
            f"{S}/labels",
            "--ortho",
            f"{Do}/ortho.jpg",
            "--out",
            D,
            "--review",
            area.review_file(s["id"]),
            "--others",
            *area.reviews(),
            *(
                ["--suspects", f"{area.dbg}/model_suspects.json"]
                if os.path.exists(f"{area.dbg}/model_suspects.json")
                else []
            ),
        ],
        [f"{D}/review.html"],
        [f"{ds}/scene.json"] + area.reviews() + [f"{area.dbg}/model_suspects.json"],
        gpu=True,
        force=force,
    )
    step(
        area,
        f"geom_{s['id']}",
        f"{STEPS}/geom_check.py",
        [
            "--labels",
            f"{S}/labels",
            "--vectors",
            f"{S}/vectors/vectors.geojson",
            "--ground-pcd",
            f"{Do}/map_ground.pcd",
            "--map",
            area.map,
            "--out",
            D,
        ],
        [f"{D}/geom_check.json"],
        [f"{ds}/scene.json"],
        force=force,
    )
    return json.load(open(f"{D}/geom_check.json")).get("summary")


def verify(area, s, force):
    """verification agent (hdlabel/agents/verify.py): judge the scene's open evidence items, apply the decisions,
    state accepted / needs_revision / rejected / uncertain -> debug/scenes/<scene>/verification.json."""
    from hdlabel.agents.verify import verify_scene

    sid = s["id"]
    D = area.dbg_dir(sid)
    if not os.path.exists(f"{D}/check/items.json"):
        raise StepError(f"{sid}: no evidence pack (debug step)", None)
    o = area.owners.get(sid, sid)
    r = verify_scene(area.agents, f"{D}/check", area.review_file(sid), f"{D}/reviewed", f"{D}/geom_check.json",
                     f"{area.scene_dir(o)}/vectors/vectors.geojson", f"{D}/verification.json",
                     area.cfg.get("verification", {}))  # fmt: skip
    nd = r["new_decisions"].get("items", {})
    new = f"{len(nd)} items decided ({sum(v['decision'] != 'ok' for v in nd.values())} not ok), " if nd else ""
    return f"{new}{r['state']}: {'; '.join(r['reasons'])}"


def qa(area, s, force):
    """labels of the scene from each drive alone (2 drives), compared (raster + vectors)."""
    drives = scene_drives(area, s)
    if len(drives) < 2:
        return "one drive only"
    S = area.scene_dir(area.owners.get(s["id"], s["id"]))
    Q = f"{area.dbg}/qa/{s['id']}"
    d1 = s["drive"]
    d2 = next(d for d in drives if d != d1)
    for d in (d1, d2):
        step(
            area,
            f"qa_labels_{s['id']}_{d}",
            f"{STEPS}/project_labels.py",
            [
                "--ground",
                f"{S}/surface/map_ground.npz",
                "--out",
                f"{Q}/labels_{d}",
                "--drive",
                area.data(d),
                area.calib(d),
                *projection_args(area),
            ],
            [f"{Q}/labels_{d}/ground_map.npz"],
            [f"{S}/surface/map_ground.npz"],
            gpu=True,
            force=force,
        )
        step(
            area,
            f"qa_vectors_{s['id']}_{d}",
            f"{STEPS}/vectorize.py",
            [
                f"{Q}/labels_{d}",
                "--out",
                f"{Q}/vectors_{d}",
                *marking_args(area),
                *curb_args(area, [f"{area.data(d)}/gt.npy"]),
                *marking_gate_args(area, [f"{area.data(d)}/gt.npy"]),
            ],
            [f"{Q}/vectors_{d}/vectors.geojson"],
            [f"{Q}/labels_{d}/ground_map.npz"],
            force=force,
        )
    step(
        area,
        f"qa_cmp_{s['id']}",
        f"{EVAL}/compare_labels.py",
        [
            f"{Q}/labels_{d1}/ground_map.npz",
            f"{Q}/labels_{d2}/ground_map.npz",
            "--gt",
            f"{area.data(d1)}/gt.npy",
            "--out",
            f"{Q}/labels",
        ],
        [f"{Q}/labels.json"],
        [f"{Q}/labels_{d2}/ground_map.npz"],
        force=force,
    )
    step(
        area,
        f"qa_vcmp_{s['id']}",
        f"{EVAL}/compare_vectors.py",
        [
            f"{Q}/vectors_{d1}/vectors.geojson",
            f"{Q}/vectors_{d2}/vectors.geojson",
            "--labels-a",
            f"{Q}/labels_{d1}/ground_map.npz",
            "--labels-b",
            f"{Q}/labels_{d2}/ground_map.npz",
            "--gt",
            f"{area.data(d1)}/gt.npy",
            "--samples",
            area.ds_dir(s["id"]),
            "--out",
            f"{Q}/vectors",
        ],
        [f"{Q}/vectors.json"],
        [f"{Q}/vectors_{d2}/vectors.geojson"],
        force=force,
    )


def summarize_qa(area):
    rows = []
    for f in sorted(glob.glob(f"{area.dbg}/qa/*/labels.json")):
        sid = f.split("/")[-2]
        L = json.load(open(f))
        vf = f.replace("labels.json", "vectors.json")
        V = json.load(open(vf)) if os.path.exists(vf) else {"kinds": {}}
        rows.append(
            dict(
                scene=sid,
                agreement=L["agreement"],
                **{f"{g}_F": L["thin"].get(g, {}).get("f") for g in ("marking", "crosswalk", "curb")},
                lane_pointF=V["kinds"].get("lane_line", {}).get("point", {}).get("0.2m", {}).get("f"),
            )
        )
    json.dump(rows, open(f"{area.dbg}/qa/summary.json", "w"), indent=1)
    for r in rows:
        print("  " + "  ".join(f"{k} {v}" for k, v in r.items()))


def scene_split(area, s):
    """train / val / test by location: the map is cut into tile_m tiles, the tiles listed in test_tiles are the fixed
    test, every val_every-th other tile (stable hash) is validation, so the same place never appears in two splits
    (drives repeat routes). Returns (split, tile)."""
    sp = area.cfg["dataset"].get("split", {})
    tile, every = sp.get("tile_m", 500), sp.get("val_every", 5)
    tr = np.load(f"{area.data(s['drive'])}/gt.npy")
    tr = tr[(tr[:, 0] >= s["t_range"][0]) & (tr[:, 0] <= s["t_range"][1]), 1:3]
    ix, iy = (int(v) for v in np.floor(tr.mean(0) / tile))
    if [ix, iy] in [list(t) for t in sp.get("test_tiles") or []]:
        return "test", [ix, iy]
    return ("val" if ((ix * 73856093) ^ (iy * 19349663)) % every == 0 else "train"), [ix, iy]


def lineage(area) -> dict:
    """what made the data: code commit, tool variants + code hashes, agent models + prompt versions, benchmark lock."""
    from hdlabel.agents import registry
    from hdlabel.agents.verify import prompt_version

    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    lk = f"{ROOT}/bench/{area.name}/LOCK.json"
    return dict(commit=git("rev-parse", "--short", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                tools=registry.versions(), agent_plan=area.agents.plan, agents=area.agents.roles,
                prompts=dict(verifier=prompt_version()),
                bench=json.load(open(lk))["data"][:12] if os.path.exists(lk) else None)  # fmt: skip


def dataset_index(area):
    """dataset/dataset.json: format, classes, default BEV, scenes with split and sample counts (the training entry point)."""
    from hdlabel.common import CLASSES
    from hdlabel.loader import ARROW_DIRECTIONS, VECTOR_CLASSES

    out, excluded = [], []
    gate = area.debug and area.cfg.get("verification", {}).get("enabled", True)
    for s in area.scenes():
        f = f"{area.ds_dir(s['id'])}/scene.json"
        if not os.path.exists(f):
            continue
        sc = json.load(open(f))
        pf = f"{area.scene_dir(area.owners.get(s['id'], s['id']))}/plan.json"
        vf = f"{area.dbg_dir(s['id'])}/verification.json"
        v = json.load(open(vf)) if os.path.exists(vf) else dict(state="unverified", reasons=[])
        if gate and v["state"] != "accepted":  # only verified data is training data
            excluded.append(dict(scene=s["id"], state=v["state"], reasons=v.get("reasons", [])))
            continue
        split, tile = scene_split(area, s)
        out.append(
            dict(
                scene=s["id"],
                drive=s["drive"],
                path=s["id"],
                samples=sc["samples"],
                qa_ok=sc["qa_ok"],
                split=split,
                tile=tile,
                bags=sc["bags"],
                t_range=s["t_range"],
                verification=dict(state=v["state"], verifier=v.get("verifier"), prompt=v.get("prompt")),
                plan=json.load(open(pf)) if os.path.exists(pf) else None,
            )
        )
    n = {k: sum(o["qa_ok"] for o in out if o["split"] == k) for k in ("train", "val", "test")}
    idx = dict(
        format=4,
        name=area.name,
        cameras=[int(c) for c in area.cfg["cameras"]],
        classes=CLASSES,
        vector_classes=VECTOR_CLASSES,
        arrow_directions=ARROW_DIRECTIONS,
        bev=area.cfg["dataset"]["bev"],
        split=dict(area.cfg["dataset"].get("split", {}), samples=n),
        samples=sum(o["samples"] for o in out),
        qa_ok=sum(o["qa_ok"] for o in out),
        scenes=out,
        excluded=excluded,
        lineage=lineage(area),
    )
    idx["version"] = hashlib.sha1(json.dumps(idx, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(area.dataset, exist_ok=True)
    f = f"{area.dataset}/dataset.json"  # unchanged: keep its time (statistics are not redone)
    if not os.path.exists(f) or json.load(open(f)) != json.loads(json.dumps(idx)):
        json.dump(idx, open(f, "w"), indent=1)
    print(
        f"  dataset: {sum(o['qa_ok'] for o in out)} / {sum(o['samples'] for o in out)} samples pass QA in {len(out)} scenes "
        f"(train {n['train']}, val {n['val']}) -> {area.dataset}/dataset.json"
    )


def prune(area, d, force=False):
    """delete undistorted frames of a drive that neither the dataset (hard links) nor labelling (keyframes) uses. The
    drive is extracted again automatically when a later export needs other frames (pruned.json records the export)."""
    D = area.data(d)
    if any(not os.path.exists(f"{area.ds_dir(s['id'])}/scene.json") for s in area.scenes() if s["drive"] == d):
        return "not all scenes exported"
    keep = {os.path.realpath(f) for f in glob.glob(f"{D}/kf_cam*/*.jpg")}
    n = freed = 0
    for f in glob.glob(f"{D}/und_cam*/*.jpg"):
        st = os.stat(f)
        if st.st_nlink == 1 and f not in keep:
            freed += st.st_size
            os.remove(f)
            n += 1
    json.dump(dict(export=export_key(area), removed=n), open(f"{D}/pruned.json", "w"))
    return f"removed {n} frames ({freed / 1e9:.1f} GB)"


def system_summary(area) -> dict:
    """for the acceptance page: verifier models, benchmark lock and accuracy, adopted tool variants."""
    from hdlabel.agents import registry

    out = {}
    v = area.cfg.get("verification", {})
    ms = [area.agents.model_of(r) for r in (v.get("role", "verifier"), v.get("confirm_role")) if r]
    out["verifier"] = " + deletions confirmed by ".join(f"{m['provider']}/{m['model'] or 'default model'}" for m in ms)
    lk = f"{ROOT}/bench/{area.name}/LOCK.json"
    if os.path.exists(lk):
        L = json.load(open(lk))
        out["benchmark"] = f"locked by {L['locked_by']} on {L['date']}"
    for f in sorted(glob.glob(f"{ROOT}/bench/{area.name}/verifier_*.json")):
        a = json.load(open(f))
        out[f"verifier accuracy ({a['judge']})"] = (
            f"{a['accuracy']} on {a['n']} reference elements; deletions correct {a['wrong_precision']}, "
            f"wrong elements found {a['wrong_recall']}"
        )
    for name, t in registry.load()["tools"].items():
        for i, h in enumerate(t.get("history", [])):
            ev = h.get("evidence") or {}
            gain = f", {ev['target']} better by {ev['gain']}" if ev.get("gain") is not None else ""
            out[f"tool {name} change {i + 1} ({h['date']})"] = (
                f"{h['previous']} -> {h['active']} by {h['by']}{gain} ({h.get('note', '')})"
            )
    return out


def do_audit(area):
    """debug/audit.json|html: scenes ranked by risk of the automatic checks + a random sample (hdlabel/audit.py)."""
    from hdlabel.agent_review import summary
    from hdlabel.audit import audit, scene_terms, write_html

    rows = []
    for s in area.scenes():
        sid = s["id"]
        vec = f"{area.scene_dir(area.owners.get(sid, sid))}/vectors/vectors.geojson"
        if os.path.exists(f"{area.ds_dir(sid)}/scene.json") and os.path.exists(vec):
            rf = area.review_file(sid)
            rv = summary(json.load(open(rf))) if os.path.exists(rf) else None
            sc = json.load(open(f"{area.ds_dir(sid)}/scene.json"))
            terms = scene_terms(area.ds_dir(sid), area.dbg_dir(sid), vec, f"{area.dbg}/qa/{sid}", rv)
            vf = f"{area.dbg_dir(sid)}/verification.json"
            st = json.load(open(vf))["state"] if os.path.exists(vf) else None
            rows.append(dict(scene=sid, terms=terms, review=rv, samples=sc["samples"], qa_ok=sc["qa_ok"], state=st))
    if not rows:
        return
    res = audit(rows, **area.cfg.get("audit", {}))
    json.dump(res, open(f"{area.dbg}/audit.json", "w"), indent=1)
    write_html(res, f"{area.dbg}/audit.html", lambda sid: f"scenes/{sid}", area.name, system_summary(area))
    top = ", ".join(f"{x['scene']} {x['risk']}" for x in res["scenes"][:3])
    state = (
        "ready to accept"
        if res["accept"]
        else (
            f"not ready ({len(res['not_reviewed'])} not reviewed, {len(res['rejected'])} rejected, "
            f"{len(res['over_max_risk'])} over max risk)"
        )
    )
    print(f"  audit: {state}; riskiest: {top}; "
          f"random sample: {res['random_sample']} -> {area.dbg}/audit.html")  # fmt: skip


def review(area, scenes, decisions, partial):
    """without --decisions: the scenes whose evidence pack (debug/scenes/<scene>/check) still has items to decide;
    with --scene ID --decisions FILE: apply the decisions to review/<scene>.json (hdlabel/agent_review.py)."""
    from hdlabel.agent_review import apply

    if decisions:
        if not scenes or len(scenes) != 1:
            raise SystemExit("review --decisions needs exactly one --scene")
        sid = scenes[0]
        rv = apply(f"{area.dbg_dir(sid)}/check", json.load(open(decisions)), area.review_file(sid),
                   f"{area.dbg_dir(sid)}/reviewed", partial)  # fmt: skip
        print(f"{sid}: {rv['verdict']} -> {area.review_file(sid)}; ./autolabel run applies it")
        return
    pending = 0
    for s in area.scenes():
        sid = s["id"]
        if scenes and sid not in scenes:
            continue
        f = f"{area.dbg_dir(sid)}/check/items.json"
        if not os.path.exists(f):
            print(f"  {sid}: no evidence pack yet (./autolabel debug)")
            continue
        pk = json.load(open(f))
        done = os.path.exists(area.review_file(sid)) and "checked" in json.load(open(area.review_file(sid)))
        if pk["items"] or pk["tiles"] or not done:
            pending += 1
            print(f"  {sid}: {len(pk['items'])} items, {len(pk['tiles'])} tiles -> {os.path.dirname(f)}")
    print(f"{pending} scenes to review (docs/AGENT_REVIEW.md)" if pending else "every scene is reviewed")


def status(area):
    print(f"{area.name}: map {area.map}, {len(area.drives)} drives\n  output {area.out}")
    for stage, items in area.status.items():
        bad = {k: v for k, v in items.items() if not v["ok"]}
        print(f"  {stage}: {len(items) - len(bad)} ok, {len(bad)} failed")
        for k, v in bad.items():
            print(f"    FAILED {k}: {v.get('note')} (log {v.get('log')})")
    if os.path.exists(f"{area.dataset}/dataset.json"):
        d = json.load(open(f"{area.dataset}/dataset.json"))
        print(
            f"  dataset: {d['qa_ok']} / {d['samples']} samples pass QA ({d['split']['samples']}) in {len(d['scenes'])} scenes"
        )
    if os.path.exists(f"{area.dbg}/audit.json"):
        r = json.load(open(f"{area.dbg}/audit.json"))
        st = (
            "ready to accept"
            if r["accept"]
            else (
                f"not ready: not reviewed {len(r.get('not_reviewed', []))}, rejected {len(r.get('rejected', []))}, "
                f"over max risk {len(r['over_max_risk'])}"
            )
        )
        print(f"  acceptance: {st}; riskiest: "
              + ", ".join(f"{x['scene']} {x['risk']}" for x in r["scenes"][:3]) + f" ({area.dbg}/audit.html)")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "command",
        choices=[
            "run",
            "ingest",
            "scenes",
            "calib",
            "label",
            "export",
            "debug",
            "qa",
            "verify",
            "audit",
            "status",
            "prune",
            "validate",
            "review",
            "signoff",
            *AGENT_COMMANDS,
        ],
    )
    p.add_argument("config", nargs="?", help="project config (overrides configs/default.yaml); validate: dataset.json")
    p.add_argument("rest", nargs="*", help="agent commands: action and arguments (hdlabel/agents/cli.py)")
    p.add_argument("--map", help="point cloud map (.pcd)")
    p.add_argument("--logs", nargs="+", help="globs: log directories (one drive each) or .mcap files")
    p.add_argument(
        "--tf-static",
        help="camera mounting for bags without /tf_static: .json / .yaml / .mcap (see configs/default.yaml)",
    )
    p.add_argument("--name", help="output name (output/<name>)")
    p.add_argument("--output", help="output root (default: output)")
    p.add_argument("--drive", nargs="+")
    p.add_argument("--scene", nargs="+")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-debug", action="store_true", help="no debug images / PCDs (large runs)")
    p.add_argument(
        "--jobs",
        type=int,
        help="parallel items (default: config jobs.cpu or 4); GPU steps are limited to jobs.gpu at once",
    )
    p.add_argument("--out", help="validate: write the report (json) here")
    p.add_argument("--decisions", help="review: decisions on the scene's evidence pack (docs/AGENT_REVIEW.md)")
    p.add_argument("--partial", action="store_true", help="review: apply although some items have no decision")
    p.add_argument("--signoff", help="signoff: the json saved by debug/audit.html")
    for o in ("--by", "--role", "--id", "--variant", "--overlay", "--code", "--target", "--note", "--kind", "--title",
              "--text", "--tags", "--budget", "--loop"):  # fmt: skip
        p.add_argument(o, help="agent commands (hdlabel/agents/cli.py)")
    for o in ("--adopt", "--rebase", "--scan", "--dry"):
        p.add_argument(o, action="store_true", help="agent commands (hdlabel/agents/cli.py)")
    a = p.parse_args()
    if a.command in AGENT_COMMANDS:  # ./autolabel bench build CONFIG: the action words are not the config
        words = [x for x in [a.config, *a.rest] if x is not None]
        cfgs = [x for x in words if x.endswith((".yaml", ".yml"))]
        a.config, a.rest = (cfgs[0] if cfgs else None), [x for x in words if x not in cfgs]
        if a.command == "kb":
            from hdlabel.agents.cli import kb_path
            from hdlabel.agents.knowledge import main as kb

            opts = sum(([f"--{k}", str(getattr(a, k))] for k in ("kind", "title", "text", "tags") if getattr(a, k)), [])
            return kb(a.rest + opts, kb_path())
    if a.command == "validate":
        from hdlabel.validate import main as validate

        sys.exit(validate([a.config] + (["--out", a.out] if a.out else [])))
    area = Area(
        load_config(a.config, pointcloud_map=a.map, logs=a.logs, tf_static=a.tf_static, name=a.name, output=a.output),
        debug=False if a.no_debug else None,
    )
    cmd = a.command
    if cmd in AGENT_COMMANDS:
        from hdlabel.agents import cli

        area.owners = label_owners(area, area.scenes()) if area.scenes() else {}
        return getattr(cli, f"{cmd}_cmd")(area, a)
    if cmd == "status":
        return status(area)
    if cmd == "review":
        return review(area, a.scene, a.decisions, a.partial)
    if cmd == "signoff":
        from hdlabel.agent_review import signoff

        if not a.signoff:
            raise SystemExit("signoff: give --signoff FILE (saved by debug/audit.html)")
        rej = signoff(json.load(open(a.signoff)), area.review_file, f"{area.out}/signoff.json")
        print(f"sign-off recorded -> {area.out}/signoff.json; rejected scenes: {rej or 'none'}"
              + ("; run ./autolabel run to drop their samples" if rej else ""))  # fmt: skip
        return
    if not area.map:
        raise SystemExit("no point cloud map: set pointcloud_map in the config or give --map")
    if not area.drives:
        raise SystemExit(f"no drives: no .mcap found for logs {area.cfg.get('logs')} (config logs / drives or --logs)")
    jobs = a.jobs or area.cfg.get("jobs", {}).get("cpu", 4)
    drives = [d for d in area.drives if not a.drive or d in a.drive]
    pick = lambda: [
        s for s in area.scenes() if (not a.scene or s["id"] in a.scene) and (not a.drive or s["drive"] in a.drive)
    ]
    failed = set()

    def each(stage, fn, items, key, n=jobs):
        """fn(area, item, force) for every item in parallel; an exception fails that item only (recorded, logged)."""
        print(f"== {stage} ({len(items)})", flush=True)
        t_stage, done = time.time(), [0]

        def progress():
            done[0] += 1
            el = time.time() - t_stage
            if len(items) > 1:
                eta = el / done[0] * (len(items) - done[0])
                print(f"  [{stage} {done[0]}/{len(items)}, {el / 60:.1f} min, ~{eta / 60:.1f} min left]", flush=True)

        def one(it):
            try:
                _one(it)
            finally:
                with area.lock:
                    progress()

        def _one(it):
            k = key(it)
            if any(k.startswith(f) or f.startswith(k) for f in failed):
                area.record(stage, k, False, note="input failed")
                return
            try:
                note = fn(area, it, a.force)
                area.record(stage, k, True, note=note)
                if note:
                    print(f"  {k}: {note}", flush=True)
            except Exception as e:
                failed.add(k)
                log = getattr(e, "log", None)
                area.record(
                    stage,
                    k,
                    False,
                    log,
                    note=str(e) if isinstance(e, StepError) else traceback.format_exc(limit=3)[-400:],
                )
                if not log:
                    print(f"  {k}: FAILED {e}", flush=True)  # failures inside step() print themselves

        with ThreadPoolExecutor(max(1, n)) as pool:
            list(pool.map(one, items))

    t0 = time.time()
    if cmd in ("run", "ingest"):
        each("ingest", ingest, drives, str, n=min(jobs, 2))  # ingest is itself multi-threaded
    if cmd in ("run", "scenes") or not area.scenes():
        print("== scenes")
        split_scenes(area)
    if cmd in ("run", "calib"):
        each("calib", calibrate, [d for d in drives if d not in failed], str)
    area.owners = label_owners(area, area.scenes()) if area.scenes() else {}
    if cmd in ("run", "label"):  # scenes with their own labels first, then the sharing ones
        each("label", label, [s for s in pick() if area.owners.get(s["id"], s["id"]) == s["id"]], lambda s: s["id"])
        each(
            "label (shared)",
            label,
            [s for s in pick() if area.owners.get(s["id"], s["id"]) != s["id"]],
            lambda s: s["id"],
        )
    if cmd in ("run", "export"):
        each("export", export, pick(), lambda s: s["id"])
        dataset_index(area)
        try:
            step(
                area,
                "dataset_stats",
                f"{EVAL}/dataset_stats.py",
                [f"{area.dataset}/dataset.json", "--out", f"{area.dbg}/dataset_stats.json"],
                [f"{area.dbg}/dataset_stats.json"],
                [f"{area.dataset}/dataset.json", *glob.glob(f"{area.dataset}/*/scene.json")],
                force=a.force,
            )
        except StepError:
            failed.add("dataset_stats")
    if (cmd == "run" and area.debug) or cmd == "debug":
        each("debug", debug, pick(), lambda s: s["id"])
    vcfg = area.cfg.get("verification", {})
    if (cmd == "run" and area.debug and vcfg.get("enabled", True)) or cmd == "verify":
        # data improvement loop: verify -> apply the deletions (label / export / debug again) -> verify again
        for rnd in range(int(vcfg.get("max_rounds", 3)) + 1):  # the last round only verifies
            ok = [s for s in pick() if s["id"] not in failed]
            each(f"verify (round {rnd + 1})", verify, ok, lambda s: s["id"])
            states = {s["id"]: json.load(open(f"{area.dbg_dir(s['id'])}/verification.json"))["state"]
                      for s in ok if os.path.exists(f"{area.dbg_dir(s['id'])}/verification.json")}  # fmt: skip
            todo = [s for s in ok if states.get(s["id"]) == "needs_revision"]
            print("  states: " + ", ".join(f"{k} {sum(v == k for v in states.values())}" for k in
                                           ("accepted", "needs_revision", "rejected", "uncertain")))  # fmt: skip
            if not todo:
                break
            if rnd == int(vcfg.get("max_rounds", 3)):
                print(f"  {len(todo)} scenes still need revision after {rnd} revisions (verification.max_rounds)")
                break
            ids = {s["id"] for s in todo} | {area.owners.get(s["id"], s["id"]) for s in todo}
            each("revise: label", label, [s for s in area.scenes() if s["id"] in ids], lambda s: s["id"])
            each("revise: export", export, todo, lambda s: s["id"])
            dataset_index(area)
            each("revise: debug", debug, todo, lambda s: s["id"])
        dataset_index(area)  # the training data = accepted scenes
    if (cmd == "run" and area.debug) or cmd in ("debug", "audit", "verify"):
        print("== audit")
        do_audit(area)
    if cmd == "qa":
        each("qa", qa, pick(), lambda s: s["id"])
        summarize_qa(area)
    if cmd == "prune" or (cmd == "run" and area.cfg.get("prune_frames")):
        each("prune", prune, drives, str)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {area.out}")
    if failed:
        print("FAILED (see ./autolabel status):", *sorted(failed), sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
