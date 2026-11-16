"""Tool registry (tools/registry.yaml): what each tool does, its variants and which one every job uses.

A variant is a config overlay (keys of configs/default.yaml) that switches a tool's model, algorithm or parameters. The
active variants are applied between configs/default.yaml and the project config (run.load_config), so adopting a
better variant changes what the next run produces, and the stamps (run.step: arguments + code hash) redo exactly the
steps it affects. Adoption is gated (hdlabel/agents/bench.py: the protected benchmark must show no regression) and
recorded in the tool's history with its evidence; rollback restores the previous active variant.
"""

from __future__ import annotations

import copy
import datetime
import os

import yaml

from hdlabel.common import ROOT

PATH = os.path.join(ROOT, "tools", "registry.yaml")


def load(path=PATH) -> dict:
    return yaml.safe_load(open(path)) if os.path.exists(path) else {"tools": {}}


def save(reg: dict, path=PATH):
    head = []
    if os.path.exists(path):  # keep the explanatory header
        for line in open(path):
            if not line.startswith("#") and line.strip():
                break
            head.append(line)
    with open(path, "w") as f:
        f.writelines(head)
        yaml.safe_dump(reg, f, sort_keys=False, allow_unicode=False, width=118)


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def overlay(reg: dict | None = None) -> dict:
    """the config overlay of every tool's active variant."""
    reg = reg or load()
    out = {}
    for name, t in reg.get("tools", {}).items():
        v = (t.get("variants") or {}).get(t.get("active", "default"))
        if v is None:
            raise ValueError(f"tool {name}: active variant {t.get('active')} is not defined")
        out = _merge(out, v)
    return out


def with_variant(reg: dict, tool: str, variant: str, spec: dict | None = None) -> dict:
    """a copy of the registry with tool's active variant set (spec: define it first). For experiments."""
    r = copy.deepcopy(reg)
    t = r["tools"][tool]
    if spec is not None:
        t.setdefault("variants", {})[variant] = spec
    if variant not in t.get("variants", {}):
        raise ValueError(f"tool {tool} has no variant {variant}")
    t["active"] = variant
    return r


def adopt(tool: str, variant: str, spec: dict | None, by: str, evidence: dict, path=PATH, note="") -> dict:
    """make variant the active one (after the gate passed): history keeps the previous one and the evidence."""
    reg = load(path)
    t = reg["tools"][tool]
    if spec is not None:
        t.setdefault("variants", {})[variant] = spec
    if variant not in t.get("variants", {}):
        raise ValueError(f"tool {tool} has no variant {variant}")
    t.setdefault("history", []).append(
        dict(date=datetime.date.today().isoformat(), active=variant, previous=t.get("active", "default"), by=by,
             note=note, evidence=evidence)
    )  # fmt: skip
    t["active"] = variant
    save(reg, path)
    return t


def rollback(tool: str, by: str, path=PATH) -> str:
    reg = load(path)
    t = reg["tools"][tool]
    prev = next((h["previous"] for h in reversed(t.get("history", [])) if h["active"] == t["active"]), "default")
    t.setdefault("history", []).append(
        dict(date=datetime.date.today().isoformat(), active=prev, previous=t["active"], by=by, note="rollback")
    )
    t["active"] = prev
    save(reg, path)
    return prev


def versions(reg: dict | None = None) -> dict:
    """tool -> {variant, code (hash of its code), model}: stored with every dataset (lineage)."""
    from hdlabel.run import code_version

    reg = reg or load()
    out = {}
    for name, t in reg.get("tools", {}).items():
        s = os.path.join(ROOT, t["script"]) if t.get("script") else None
        out[name] = dict(variant=t.get("active", "default"),
                         code=code_version(s)[:12] if s and os.path.exists(s) else None,
                         model=(t.get("model") or {}).get("id"))  # fmt: skip
    return out


def describe(reg: dict | None = None, tools=None) -> str:
    """compact text of the registry for an agent prompt."""
    reg = reg or load()
    lines = []
    for name, t in reg.get("tools", {}).items():
        if tools and name not in tools:
            continue
        lines.append(
            f"- {name} [{t.get('stage')}] {t.get('purpose')}. algorithm: {t.get('algorithm')}. "
            f"good: {'; '.join(t.get('good', []))}. bad: {'; '.join(t.get('bad', []))}. cost: {t.get('cost')}. "
            f"license: {t.get('license')}. active: {t.get('active')} of {list((t.get('variants') or {}))}"
        )
    return "\n".join(lines)
