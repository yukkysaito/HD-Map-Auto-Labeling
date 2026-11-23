"""Agent commands of ./autolabel (hdlabel/run.py dispatches here):

kb search|add|list|show ...          knowledge store (hdlabel/agents/knowledge.py)
tools [list|adopt|rollback] ...      tool registry (tools/registry.yaml)
bench build|lock|status|eval|verifier [--role R] [--by NAME]   protected benchmark (hdlabel/agents/bench.py)
bench rb-propose [--scene S ..] [--out DIR] | rb-apply rb_review.json --by person:NAME [--out DIR]
                                     road boundary group: LiDAR proposal -> person review -> lock (bench_rb.py)
experiment --id ID [--variant TOOL=NAME] [--overlay JSON] [--code DIR] [--target METRIC] [--adopt]
                                     candidate in isolation -> metrics -> gate vs the baseline (-> adopt)
impact [--variant TOOL=NAME]         the steps / scenes a change would redo (dry run of the stamps)
evolve [--scan] [--budget USD]       orchestrator: pick and run the next improvement (hdlabel/agents/orchestrator.py)
train                                train + evaluate the small NN on the accepted data, failure analysis
"""

from __future__ import annotations

import json
import os

from hdlabel.common import ROOT


def kb_path() -> str:
    return os.path.join(ROOT, "knowledge", "kb.sqlite")


def knowledge():
    from hdlabel.agents.knowledge import Knowledge

    return Knowledge(kb_path())


def judge_fn(area, role):
    from hdlabel.agents.verify import judge_items

    return lambda rows: judge_items(area.agents, rows, role, jobs=area.cfg.get("verification", {}).get("jobs", 4))


def bench_cmd(area, a):
    from hdlabel.agents import bench, bench_review

    act = a.rest[0] if a.rest else "status"
    if act == "build":
        items = bench.collect(area.reviews(), area.dbg_dir)
        roles = [r for r in ("verifier", "adjudicator") if area.agents.available(r)]
        fns = {
            f"{area.agents.model_of(r)['provider']}/{area.agents.model_of(r)['model']}": judge_fn(area, r)
            for r in roles
        }
        ref = bench.build(area.name, items, fns, a.by or "agent")
        n = sum(e["ref"] is not None for e in ref["items"])
        print(f"reference: {n} of {len(ref['items'])} elements with a majority of {ref['judges']}; "
              f"holdout scenes {ref['holdout_scenes']} -> {bench.bench_dir(area.name)} (lock it: bench lock --by NAME)")  # fmt: skip
    elif act == "lock":
        if not a.by:
            raise SystemExit("bench lock --by NAME (the person who checked the reference)")
        bench.lock(area.name, a.by)
        print(f"locked by {a.by}")
    elif act == "status":
        lk = bench.check_lock(area.name)
        ref = json.load(open(f"{bench.bench_dir(area.name)}/reference.json"))
        from collections import Counter

        c = Counter((e["group"], e["ref"], e["split"]) for e in ref["items"])
        print(f"locked by {lk['locked_by']} {lk['date']}; judges {ref['judges']}")
        for k, v in sorted(c.items(), key=str):
            print(f"  {k}: {v}")
        rbf = f"{bench.bench_dir(area.name)}/road_boundary.json"
        if os.path.exists(rbf):
            rb = json.load(open(rbf))
            ws = [w for w in rb["windows"] if w["complete"]]
            print(
                f"  road_boundary: {len(ws)} complete windows of {len(rb['windows'])} (holdout "
                f"{sum(w['split'] == 'holdout' for w in ws)}), {sum(len(w['lines']) for w in ws)} lines, by {rb['by']}"
            )
        for j in ref["judges"]:  # each reference judge against the majority of the others (leave-one-out)
            res = {e["key"]: dict(decision=e["judges"][j]) for e in ref["items"] if j in e["judges"]}
            acc = bench.verifier_accuracy(area.name, res, j)
            print(
                f"  judge {j}: accuracy {acc['accuracy']} on {acc['n']} (finds wrong: recall {acc['wrong_recall']}, "
                f"precision {acc['wrong_precision']}); "
                + ", ".join(f"{g} {v['accuracy']}" for g, v in acc["per_group"].items())
            )
    elif act == "verifier":
        role = a.role or "verifier"
        ref = json.load(open(f"{bench.bench_dir(area.name)}/reference.json"))
        rows = [dict(key=e["key"], image=f"{bench.bench_dir(area.name)}/{e['evidence']}", kind=e["kind"], reason=e["why"],
                     size_m=e["size_m"], confidence=None) for e in ref["items"] if e["ref"] is not None]  # fmt: skip
        if a.role:  # one model
            res = judge_fn(area, role)(rows)
            m = area.agents.model_of(role)
            judge = f"{m['provider']}/{m['model']}"
        else:  # the production verifier: verifier + confirmer (verify.judge_confirmed)
            from hdlabel.agents.verify import judge_confirmed

            vc = area.cfg.get("verification", {})
            res = judge_confirmed(area.agents, rows, vc)
            ms = [area.agents.model_of(r) for r in (vc.get("role", "verifier"), vc.get("confirm_role")) if r]
            m = ms[0]
            judge = "+".join(f"{x['provider']}/{x['model']}" for x in ms)
        acc = bench.verifier_accuracy(area.name, res, judge)
        out = f"{bench.bench_dir(area.name)}/verifier_{judge.replace('/', '_')}.json"
        json.dump(dict(acc, results=res), open(out, "w"), indent=1)
        knowledge().add("verification", f"verifier accuracy {judge}", json.dumps({k: v for k, v in acc.items() if k != "confusion"}),
                        ["verifier", m["provider"], str(m["model"])], data=acc, area=area.name)  # fmt: skip
        print(json.dumps({k: v for k, v in acc.items()}, indent=1))
    elif act == "rb-propose":
        from hdlabel.agents import bench_rb

        out = a.out or bench_rb.draft_dir(area)
        d = bench_rb.propose_area(area, out, a.scene, f"{bench.bench_dir(area.name)}/reference.json")
        print(f"{len(d['windows'])} windows, {sum(len(w['lines']) for w in d['windows'])} proposed lines ({d['stats']})"
              f"\nopen {out}/review.html, save rb_review.json, then: bench rb-apply <config> rb_review.json "
              f"--by person:NAME" + (f" --out {out}" if a.out else ""))  # fmt: skip
    elif act == "rb-apply":
        from hdlabel.agents import bench_rb

        if len(a.rest) < 2 or not (a.by or "").startswith("person:"):
            raise SystemExit("bench rb-apply <config> rb_review.json --by person:NAME [--out DRAFT_DIR]")
        dec = json.load(open(a.rest[1]))
        stat = bench_rb.apply(area.name, a.out or bench_rb.draft_dir(area), dec.get("windows", dec), a.by)
        print(f"road boundary reference written and locked by {a.by}: {stat}")
    elif act == "review":
        print(
            f"open {bench_review.review_page(area.name)} in a browser; then: bench apply <config> bench_review.json --by person:NAME"
        )
    elif act == "apply":
        if len(a.rest) < 2 or not (a.by or "").startswith("person:"):
            raise SystemExit("bench apply <config> bench_review.json --by person:NAME")
        dec = json.load(open(a.rest[1]))
        stat = bench_review.apply_review(area.name, dec.get("decisions", dec), a.by)
        print(f"reference updated and locked by {a.by}: {stat}")
    else:
        raise SystemExit(
            f"bench {act}: unknown (build | lock | status | verifier | review | apply | rb-propose | rb-apply)"
        )


def tools_cmd(area, a):
    from hdlabel.agents import registry

    act = a.rest[0] if a.rest else "list"
    if act == "list":
        reg = registry.load()
        ver = registry.versions(reg)
        for name, t in reg["tools"].items():
            h = (t.get("history") or [{}])[-1]
            print(f"{name:<11} {t.get('active'):<16} code {ver[name]['code']}  {t.get('purpose')}"
                  + (f"  [last change {h.get('date')}: {h.get('note') or h.get('active')}]" if h else ""))  # fmt: skip
    elif act == "rollback":
        tool = a.rest[1]
        prev = registry.rollback(tool, a.by or "agent")
        knowledge().add(
            "fix", f"rollback {tool} -> {prev}", f"by {a.by or 'agent'}", ["rollback", tool], area=area.name
        )
        print(f"{tool}: active variant -> {prev}; ./autolabel run redoes the affected steps")
    else:
        raise SystemExit(f"tools {act}: unknown (list | rollback); adoption goes through `experiment --adopt`")


def parse_variant(v: str | None):
    if not v:
        return None, None
    tool, _, name = v.partition("=")
    return tool, name


def experiment_cmd(area, a):
    """baseline (cached per code + lock) and candidate runs, metrics, gate, optional adoption."""
    from hdlabel.agents import bench, experiment, registry

    missing = [f"--{k}" for k in ("id", "target") if not getattr(a, k)]
    if missing:  # before any run: the baseline alone takes minutes
        raise SystemExit(f"experiment needs {' and '.join(missing)} (e.g. --id my-test --target marking.fp)")
    tool, var = parse_variant(a.variant)
    reg = registry.load()
    overlay = json.loads(a.overlay) if a.overlay else {}
    spec = None
    if tool:
        t = reg["tools"][tool]
        spec = t["variants"].get(var) if var in t.get("variants", {}) else overlay
        overlay = registry._merge(spec or {}, overlay)
    drive_writes = reg["tools"][tool].get("writes", []) if tool and reg["tools"][tool].get("stage") == "drive" else []
    scenes = experiment.bench_scenes(area)
    base_id = "baseline"
    bm = f"{experiment.exp_root(area, base_id)}/metrics.json"
    if (
        a.rebase
        or not os.path.exists(bm)
        or json.load(open(bm)).get("lock") != bench.check_lock(area.name)["data"][:12]
        or json.load(open(bm)).get("tools") != registry.versions()  # an adoption or a code change since
    ):
        experiment.prepare(area, base_id, {}, scenes)
        experiment.run(area, base_id)
        experiment.measure(area, base_id)
    base = json.load(open(bm))
    experiment.prepare(area, a.id, overlay, scenes, drive_writes)
    experiment.run(area, a.id, os.path.abspath(a.code) if a.code else ROOT, drive_level=bool(drive_writes))
    cand = experiment.measure(area, a.id)
    g = bench.gate(area.name, base, cand, a.target)
    g = experiment.spec_gate(g, experiment.features(f"{experiment.exp_root(area, a.id)}/work", scenes))
    res = dict(id=a.id, tool=tool, variant=var, overlay=overlay, code=a.code, base=base, cand=cand, gate=g)
    json.dump(res, open(f"{experiment.exp_root(area, a.id)}/result.json", "w"), indent=1)
    kb = knowledge()
    kb.add("tool_eval", f"experiment {a.id}: {tool}={var} target {a.target} gain {g['gain']} -> "
           f"{'pass' if g['adopt'] else 'fail'}", "; ".join(g["fails"]) or "all criteria met",
           [x for x in ("experiment", tool, var, a.target) if x], status="done" if g["adopt"] else "failed",
           data=dict(gate=g, seconds=cand.get("seconds")), area=area.name)  # fmt: skip
    print(json.dumps(g, indent=1))
    if a.adopt and g["adopt"] and tool:
        registry.adopt(tool, var, spec, a.by or "agent", dict(experiment=a.id, gain=g["gain"], target=a.target,
                                                               rows=g["rows"]), note=a.note or "")  # fmt: skip
        print(f"adopted {tool}={var}")
    return res


def impact_cmd(area, a):
    """which steps would run again with the current code / registry (or with --variant TOOL=NAME): a dry run of the
    stamps (run.step), nothing is executed, written or removed; steps reading a redone step's outputs (or a file that
    would be rewritten, like a scene's fixes.json) are included."""
    from hdlabel import run

    tool, var = parse_variant(a.variant)
    if tool:
        from hdlabel.agents import registry

        area.cfg = run.load_config(a.config, registry=registry.with_variant(registry.load(), tool, var))
    run.DRY_RUN, run.DRY_RUN_OUT = [], set()
    try:
        area.owners = run.label_owners(area, area.scenes())
        for d in area.drives:
            try:
                run.ingest(area, d, False)
            except Exception:
                pass
        for s in area.scenes():
            for fn in (run.label, run.export) + ((run.debug,) if area.debug else ()):
                try:
                    fn(area, s, False)
                except Exception:
                    pass
        redo = list(run.DRY_RUN)
    finally:
        run.DRY_RUN = None
    for tag, script in redo:
        print(f"  {tag}: {script}")
    print(f"{len(redo)} steps would run")
    return redo


def train_cmd(area, a):
    from hdlabel.agents import training

    r = training.loop(area, knowledge())
    print(json.dumps(r, indent=1))
    return r


def evolve_cmd(area, a):
    from hdlabel.agents import orchestrator

    loops = int(a.loop or 1)
    out = []
    for _ in range(loops):
        r = orchestrator.evolve(area, a)
        out.append(r)
        print(json.dumps(r, indent=1, default=str)[:4000])
        if r["action"] in ("dry", "scan"):
            break
    return out


def spec_cmd(area, a):
    """./autolabel spec <config> [status | apply | aligned --by NAME]: the label spec (spec/labels.yaml) vs the tools."""
    from hdlabel.agents import orchestrator

    act = a.rest[0] if a.rest else "status"
    if act == "status":
        st = orchestrator.spec_state()
        print(f"spec {st['version']}; tools aligned to {st['aligned'].get('version')} ({st['aligned'].get('commit')}, "
              f"{st['aligned'].get('date')}, by {st['aligned'].get('by')})")  # fmt: skip
        print(
            "tools match the spec"
            if not st["changed"]
            else f"spec changed -> ./autolabel spec <config> apply\n{st['diff']}"
        )
        return st
    if act == "apply":
        r = orchestrator.spec_loop(area, knowledge(), dry=a.dry)
        print(json.dumps(r, indent=1, default=str)[:4000])
        if r.get("merged"):
            print(
                "next: ./autolabel impact <config> (what will be redone), ./autolabel run <config> (label + verify again)"
            )
        return r
    if act == "aligned":  # a person brought the tools in line by hand
        if not a.by:
            raise SystemExit("spec aligned --by NAME (who checked that the tools follow the spec)")
        orchestrator.mark_aligned(a.by, a.note or "")
        print("recorded in tools/registry.yaml (commit it)")
        return None
    raise SystemExit(f"spec {act}: unknown (status | apply | aligned)")
