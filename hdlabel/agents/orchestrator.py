"""Orchestrator: looks at the state of an area and runs the improvement that is due (./autolabel evolve).

  data loop   scenes not verified / needing revision    -> ./autolabel run (verify -> revise -> re-verify)
  tool loop   a problem many scenes share               -> analysis -> research -> engineer (worktree) -> benchmark
                                                           -> gate -> adopt (merge + registry) -> reprocess affected
  model loop  accepted data changed since the last NN    -> train + evaluate -> failure analysis -> knowledge / problems
  scan        nothing due, or --scan                     -> technology scan per tool (newer models / methods)
Problems are ranked by measured impact (wrong length / count from the verification and the training analysis); a
problem is not attacked again after `evolve.max_attempts` failed attempts unless something changed (other model of
the engineer / researcher, new knowledge). Every step is recorded in the knowledge store; agent calls are limited by
the budget (agents.budget_usd per command, evolve.max_hours).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

from hdlabel.agents.llm import ProviderError
from hdlabel.common import ROOT

PROMPTS = os.path.join(os.path.dirname(__file__), "prompts")

# which tool a verified problem points to first (the agents may conclude otherwise in the analysis)
SUSPECT = {("curb", "false_positive"): "vectorize", ("curb", "offset"): "vectorize", ("curb", "lidar_conflict"): "vectorize",
           ("marking", "false_positive"): "vectorize", ("marking", "offset"): "project", ("marking", "missed"): "segment",
           ("curb", "missed"): "segment", (None, "missed"): "segment", ("systematic", "rejected"): "calib"}  # fmt: skip
GROUP = dict(lane_line="marking", stop_line="marking", arrow="marking", marking_area="marking", crosswalk="marking",
             curb="curb")  # fmt: skip


def problems(area) -> list[dict]:
    """issues of all verified scenes grouped by (group, type), largest impact first."""
    groups = {}
    seen = set()
    for f in glob.glob(f"{area.dbg}/scenes/*/verification.json"):
        v = json.load(open(f))
        for i in v.get("issues", []):
            key = (i.get("kind"), round(i["at"][0], 1), round(i["at"][1], 1), i["type"])
            if key in seen:  # the same element seen from two scenes
                continue
            seen.add(key)
            g = GROUP.get(i.get("kind"), i.get("kind"))
            p = groups.setdefault((g, i["type"]), dict(group=g, type=i["type"], count=0, examples=[], scenes=set()))
            p["count"] += 1
            p["scenes"].add(v["scene"])
            if len(p["examples"]) < 6:
                p["examples"].append(dict(scene=v["scene"], kind=i.get("kind"), note=i.get("note"),
                                          evidence=f"{area.dbg}/scenes/{v['scene']}/{i['evidence']}" if i.get("evidence") else None))  # fmt: skip
        if v.get("state") == "rejected":
            p = groups.setdefault(
                ("systematic", "rejected"),
                dict(group="systematic", type="rejected", count=0, examples=[], scenes=set()),
            )
            p["count"] += 1
            p["scenes"].add(v["scene"])
    out = []
    for p in groups.values():
        if p["type"] == "uncertain":
            continue  # a person looks at those; not a tool problem by itself
        p["scenes"] = sorted(p["scenes"])
        p["tool"] = SUSPECT.get((p["group"], p["type"]), "vectorize")
        p["id"] = f"{p['group']}:{p['type']}"
        out.append(p)
    return sorted(out, key=lambda p: -p["count"])


def attempts(kb, pid: str) -> list[dict]:
    return [r for r in kb.list(kind="attempt", limit=200) if r["data"].get("problem") == pid]


def research(area, kb, problem: dict, extra="") -> dict:
    """researcher role (web search) -> candidates; each recorded in the knowledge store."""
    from hdlabel.agents import registry

    q = f"{problem['group']} {problem['type']} {problem['tool']} " + " ".join(
        e.get("note") or "" for e in problem["examples"]
    )
    prompt = (f"Problem ({problem['id']}, {problem['count']} cases in {len(problem['scenes'])} scenes): "
              f"{problem.get('statement') or problem['id']}\nExamples (verifier notes): "
              + "; ".join(str(e.get("note")) for e in problem["examples"]) + f"\n{extra}\n\nCurrent tools:\n"
              + registry.describe() + "\n\nKnown (knowledge store):\n" + kb.context(q))  # fmt: skip
    rep = area.agents.ask(
        "researcher", prompt, system=open(f"{PROMPTS}/researcher.md").read(), tools=("WebSearch", "WebFetch")
    )
    res = rep.json()
    for c in res.get("candidates", []):
        kb.add("research", c.get("name", "?"), f"{c.get('idea')} | apply: {c.get('apply')} | expected: {c.get('expected')}",
               [problem["tool"], problem["group"], problem["type"], c.get("type", "")], status="open",
               data=dict(c, problem=problem["id"], model=f"{rep.provider}/{rep.model}"), area=area.name)  # fmt: skip
    return res


def worktree(branch: str) -> str:
    """a fresh git worktree of HEAD for a candidate (the main checkout must be committed: the worktree starts from it)."""
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"commit the main checkout first (a worktree starts from HEAD): {dirty.splitlines()[:5]}")
    d = os.path.join(ROOT, "output", "worktrees", branch)
    if not os.path.exists(d):
        subprocess.run(["git", "worktree", "add", "-q", "-b", branch, d, "HEAD"], cwd=ROOT, check=True)
    return d


def touched(wt: str) -> list[str]:
    """files the worktree changed against the main checkout's HEAD (committed or not)."""
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    a = subprocess.run(["git", "diff", "--name-only", base], cwd=wt, capture_output=True, text=True).stdout.split()
    b = [
        x[3:]
        for x in subprocess.run(
            ["git", "status", "--porcelain"], cwd=wt, capture_output=True, text=True
        ).stdout.splitlines()
    ]
    return sorted(set(a) | set(b))


def engineer(area, kb, problem: dict, candidate: dict, branch: str) -> dict:
    """engineer role in a fresh worktree: implement candidate as a variant of the tool, commit. Returns its answer."""
    from hdlabel.agents import bench, registry

    wt = worktree(branch)
    task = (f"Tool: {problem['tool']}\nProblem: {problem.get('statement') or problem['id']} ({problem['count']} cases)\n"
            f"Evidence examples: {json.dumps(problem['examples'])[:2500]}\n\nApproach to implement: {json.dumps(candidate)}\n\n"
            f"Registry entry: {registry.describe(tools=[problem['tool']])}\nKnown: {kb.context(problem['id'] + ' ' + candidate.get('name', ''))}\n"
            f"Main checkout (read only, has the venv and the data): {ROOT}\nPython: {ROOT}/.venv/bin/python\n"
            f"Scene data to look at (read only): {area.work}/scenes, {area.dbg}/scenes\n")  # fmt: skip
    rep = area.agents.ask("engineer", task, system=open(f"{PROMPTS}/engineer.md").read(),
                          tools=("Read", "Edit", "Write", "Bash", "Glob", "Grep"), cache=False, cwd=wt)  # fmt: skip
    ans = rep.json()
    bad = [f for f in touched(wt) if f.startswith(bench.PROTECTED)]
    if bad:
        raise RuntimeError(f"candidate touches protected paths: {bad}")
    return dict(ans, worktree=wt, cost_usd=rep.cost_usd)


def tool_loop(area, kb, problem: dict, cfg: dict) -> dict:
    """one attempt at a problem: research -> engineer -> experiment + gate -> adopt / record the failure."""
    from argparse import Namespace

    from hdlabel.agents import cli

    rid = kb.add("attempt", f"tool loop on {problem['id']}", json.dumps(problem)[:1500], [problem["tool"], problem["group"], problem["type"]],
                 status="open", data=dict(problem=problem["id"]), area=area.name)  # fmt: skip
    res = research(area, kb, problem)
    cands = res.get("candidates", [])
    if not cands:
        kb.update(rid, status="failed", text="no candidates found")
        return dict(problem=problem["id"], result="no candidates")
    pick = next((c for c in cands if c.get("name") in (res.get("recommendation") or "")), cands[0])
    branch = f"evolve-{problem['tool']}-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        eng = engineer(area, kb, problem, pick, branch)
    except (ProviderError, ValueError, RuntimeError) as e:
        kb.update(
            rid, status="failed", text=f"engineer failed: {e}"[:1000], data=dict(problem=problem["id"], candidate=pick)
        )
        return dict(problem=problem["id"], result=f"engineer failed: {e}")
    tool = problem["tool"]
    a = Namespace(id=branch, variant=f"{tool}={eng['variant']}", overlay=json.dumps(eng.get("overlay") or {}), code=eng["worktree"],
                  target=cfg.get("targets", {}).get(problem["id"], f"{problem['group']}.fp"), adopt=False, rebase=False,
                  by="agent", note=pick.get("name"), config=None)  # fmt: skip
    r = cli.experiment_cmd(area, a)
    ok = r["gate"]["adopt"]
    if ok:  # merge the candidate's commit, then make its variant active (both are local commits a person can revert)
        subprocess.run(
            ["git", "merge", "--no-ff", "-q", "-m", f"Merge {branch}: {pick.get('name')}", branch], cwd=ROOT, check=True
        )
        from hdlabel.agents import registry

        registry.adopt(
            tool,
            eng["variant"],
            None,
            "agent",
            dict(experiment=branch, gain=r["gate"]["gain"], target=a.target),
            note=pick.get("name"),
        )
    kb.update(rid, status="done" if ok else "failed",
              text=f"{pick.get('name')}: gain {r['gate']['gain']} on {a.target}; " + ("; ".join(r["gate"]["fails"]) or "adopted"),
              data=dict(problem=problem["id"], candidate=pick, gate=r["gate"], branch=branch))  # fmt: skip
    return dict(problem=problem["id"], candidate=pick.get("name"), adopted=ok, gate=r["gate"])


def scan(area, kb) -> list[dict]:
    """periodic technology scan: per tool with a model / algorithm, are there better ones now?"""
    from hdlabel.agents import registry

    out = []
    for name, t in registry.load()["tools"].items():
        if t.get("stage") not in ("drive", "scene"):
            continue
        prob = dict(id=f"scan:{name}", group=name, type="scan", tool=name, count=0, scenes=[], examples=[],
                    statement=f"Is there a clearly better method / model for the tool '{name}' ({t.get('purpose')}) than "
                              f"{t.get('algorithm')} {(t.get('model') or {}).get('id', '')}? Known weaknesses: {t.get('bad')}")  # fmt: skip
        try:
            out.append(dict(tool=name, **research(area, kb, prob)))
        except (ProviderError, ValueError) as e:
            out.append(dict(tool=name, error=str(e)))
    return out


def evolve(area, a) -> dict:
    """one orchestration pass (see module doc); returns what was done."""
    from hdlabel.agents.cli import knowledge

    kb = knowledge()
    cfg = area.cfg.get("evolve", {})
    if a.budget:
        area.agents.budget = float(a.budget)
    if a.scan:
        return dict(action="scan", result=scan(area, kb))
    states = {}
    for s in area.scenes():
        f = f"{area.dbg_dir(s['id'])}/verification.json"
        states[s["id"]] = json.load(open(f))["state"] if os.path.exists(f) else "unverified"
    todo = [s for s, st in states.items() if st in ("unverified", "needs_revision")]
    if todo:
        r = subprocess.run([f"{ROOT}/autolabel", "run", a.config], cwd=ROOT)
        return dict(action="data loop", scenes=todo, returncode=r.returncode)
    probs = [p for p in problems(area) if p["count"] >= cfg.get("min_cases", 5)]
    for p in probs:
        n = len([x for x in attempts(kb, p["id"]) if x["status"] == "failed"])
        if n < cfg.get("max_attempts", 2) and not a.dry:
            return dict(action="tool loop", **tool_loop(area, kb, p, cfg))
    if a.dry:
        return dict(
            action="dry", states=states, problems=[{k: v for k, v in p.items() if k != "examples"} for p in probs]
        )
    ds = f"{area.dataset}/dataset.json"
    last = kb.list(kind="training", limit=1)
    if os.path.exists(ds) and (not last or last[0]["data"].get("dataset_mtime") != os.path.getmtime(ds)):
        from hdlabel.agents import training

        return dict(action="model loop", **training.loop(area, kb))
    return dict(action="scan", result=scan(area, kb))
