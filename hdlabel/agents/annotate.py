"""Annotation planning: which variant of each tool labels a scene.

A registry variant may say when it applies: `when: {fact: [min, max], ...}` (all must hold; null = open end). For each
tool the first variant (in registry order) whose conditions hold for the scene is used, else the active one. So a
method that is better only under some conditions (measured per condition on the benchmark) is used only there. The
plan is written to work/scenes/<scene>/plan.json and into the dataset lineage (reproducible per job).

Scene facts (from the scene's own data, no model): drives (fused drives), multi_level (share of surface cells with
more than one height layer), lidar_density (LiDAR ground points per m2 of connected surface), path_m, turn_deg.
"""

from __future__ import annotations

import json
import os


def facts(area, s: dict, drives: list[str]) -> dict:
    out = dict(drives=len(drives), path_m=s.get("path_m"), turn_deg=s.get("turn_deg"))
    f = f"{area.scene_dir(s['id'])}/surface/map_ground.json"
    if os.path.exists(f):
        su = json.load(open(f)).get("surface", {})
        res = json.load(open(f)).get("surf_res", 0.2)
        if su.get("layer_cells"):
            out["multi_level"] = round(su["multi_layer_cells"] / su["layer_cells"], 3)
        if su.get("connected_cells"):
            out["lidar_density"] = round(su["ground_points"] / (su["connected_cells"] * res * res), 1)
    return out


def holds(when: dict, fx: dict) -> bool:
    for k, (lo, hi) in (when or {}).items():
        v = fx.get(k)
        if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
            return False
    return True


def plan(reg: dict, fx: dict) -> dict:
    """tool -> variant for a scene with facts fx."""
    out = {}
    for name, t in reg.get("tools", {}).items():
        pick = t.get("active", "default")
        for vn, v in (t.get("variants") or {}).items():
            if isinstance(v, dict) and v.get("when") is not None and holds(v["when"], fx):
                pick = vn
                break
        out[name] = pick
    return out


def overlay(reg: dict, chosen: dict) -> dict:
    """config overlay of the variants chosen for a scene that differ from the active ones."""
    from hdlabel.agents.registry import _merge

    out = {}
    for name, vn in chosen.items():
        t = reg["tools"][name]
        if vn != t.get("active", "default"):
            out = _merge(out, {k: v for k, v in t["variants"][vn].items() if k != "when"})
    return out
