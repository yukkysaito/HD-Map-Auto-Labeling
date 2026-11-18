"""Agent layer without any model: replay provider, knowledge store, registry, benchmark lock / gate, verifier states."""

import json
import os

import pytest
import yaml

from hdlabel.agents import bench, registry
from hdlabel.agents.knowledge import Knowledge
from hdlabel.agents.llm import Agents, BudgetExceeded, ProviderError, parse_json
from hdlabel.agents.verify import judge_items, scene_state


def test_parse_json_from_prose():
    assert parse_json('Sure:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('result {"items": {"0": {"decision": "ok"}}} done')["items"]["0"]["decision"] == "ok"
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_replay_cache_ledger_and_budget(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8fake")
    table = tmp_path / "replay.json"
    table.write_text(json.dumps({"*": {"items": {"0": {"decision": "drop", "confidence": 0.9, "reason": "shadow"}}}}))
    ag = Agents(dict(roles=dict(verifier=dict(provider="replay")), replay=str(table)), str(tmp_path / "agent"))
    assert ag.available("verifier")
    rows = [dict(key="0", image=str(img), kind="curb", reason="spot", confidence=0.5, size_m=2.0)]
    r = judge_items(ag, rows)
    assert r["0"]["decision"] == "drop" and r["0"]["confidence"] == 0.9
    judge_items(ag, rows)  # second time from the cache
    led = [json.loads(x) for x in open(tmp_path / "agent" / "ledger.jsonl")]
    assert [e["cached"] for e in led] == [False, True]
    # a role without a provider fails clearly; the verifier turns failures into unsure
    ag2 = Agents(dict(roles={}), str(tmp_path / "a2"))
    assert not ag2.available("verifier")
    with pytest.raises(ProviderError):
        ag2.ask("verifier", "x")
    assert judge_items(ag2, rows)["0"]["decision"] == "unsure"
    ag3 = Agents(dict(roles=dict(r=dict(provider="claude_cli")), budget_usd=1), str(tmp_path / "a3"))
    ag3.spent = 1.0
    with pytest.raises(BudgetExceeded):
        ag3.ask("r", "x", cache=False)


def test_knowledge_search(tmp_path):
    kb = Knowledge(str(tmp_path / "kb.sqlite"))
    a = kb.add("problem", "curb on planter edges", "the segmentation calls planter edges curb", ["curb", "vectorize"])
    kb.add("research", "LiDAR step check", "keep curbs with a height step", ["curb"], status="open")
    assert kb.search("planter")[0]["id"] == a
    assert {r["kind"] for r in kb.search("curb")} == {"problem", "research"}
    assert kb.list(kind="research", status="open")[0]["title"] == "LiDAR step check"
    b = kb.add("fix", "newer", "", ["curb"], supersedes=a)
    assert kb.get(a)["status"] == "superseded" and kb.get(b)["supersedes"] == a
    assert "planter" in kb.context("planter edges")


def test_registry_overlay_adopt_rollback(tmp_path):
    p = str(tmp_path / "registry.yaml")
    reg = {"tools": {"vectorize": {"script": "hdlabel/steps/vectorize.py", "variants": {"default": {},
                                                                                      "strict": {"markings": {"x": 1}}},
                                   "active": "default", "history": []}}}  # fmt: skip
    yaml.safe_dump(reg, open(p, "w"))
    assert registry.overlay(registry.load(p)) == {}
    registry.adopt("vectorize", "strict", None, "test", {"gain": 0.1}, path=p)
    assert registry.overlay(registry.load(p)) == {"markings": {"x": 1}}
    assert registry.rollback("vectorize", "test", path=p) == "default"
    t = registry.load(p)["tools"]["vectorize"]
    assert t["active"] == "default" and [h["active"] for h in t["history"]] == ["strict", "default"]
    with pytest.raises(ValueError):
        registry.with_variant(registry.load(p), "vectorize", "missing")


def test_bench_lock_gate_and_majority(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "bench_dir", lambda name: str(tmp_path / name))
    d = tmp_path / "t"
    os.makedirs(d / "evidence")
    items = [dict(key="0", group="curb", ref="ok", split="dev", size_m=5.0, at=[0.0, 0.0], judges={"a": "ok", "b": "ok"}),
             dict(key="1", group="curb", ref="wrong", split="holdout", size_m=3.0, at=[10.0, 0.0],
                  judges={"a": "drop", "b": "off"})]  # fmt: skip
    (d / "reference.json").write_text(json.dumps(dict(items=items)))
    (d / "criteria.yaml").write_text(open(bench.CRITERIA).read())
    with pytest.raises(RuntimeError):
        bench.check_lock("t")
    bench.lock("t", "person:test")
    line = lambda x0, x1: {"type": "Feature", "properties": {"kind": "curb"},  # noqa: E731
                           "geometry": {"type": "LineString", "coordinates": [[x0, 0.0, 0.0], [x1, 0.0, 0.0]]}}  # fmt: skip
    base = {s: bench.evaluate("t", [line(-1, 1), line(9, 11)], [0.1, 0.5], split=s) for s in (None, "holdout")}
    cand = {s: bench.evaluate("t", [line(-1, 1)], [0.1], split=s) for s in (None, "holdout")}
    assert base[None]["curb"]["recall"] == 1.0 and base[None]["curb"]["fp"] == 1.0
    assert cand[None]["curb"]["fp"] == 0.0 and cand[None]["curb"]["lidar"] == 1.0
    g = bench.gate("t", {"all": base[None], "holdout": base["holdout"]}, {"all": cand[None], "holdout": cand["holdout"]},
                   "curb.fp")  # fmt: skip
    assert g["adopt"] and g["gain"] == 1.0
    g2 = bench.gate("t", {"all": cand[None], "holdout": cand["holdout"]}, {"all": base[None], "holdout": base["holdout"]},
                    "curb.fp")  # fmt: skip
    assert not g2["adopt"]
    (d / "criteria.yaml").write_text("min_improvement: 0\nno_regression: {}\n")  # tampering is detected
    with pytest.raises(RuntimeError):
        bench.check_lock("t")
    assert bench.majority({"a": "ok", "b": "drop", "c": "off"}) == "wrong"
    assert bench.majority({"a": "ok", "b": "drop", "c": "unsure"}) is None


def test_scene_state_rules():
    feat = {"type": "Feature", "properties": {"kind": "curb"},
            "geometry": {"type": "LineString", "coordinates": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}}  # fmt: skip
    ok = dict(kind="curb", at=[9.0, 9.0], why="random spot check", decision="ok", size_m=1.0)
    rv = dict(checked=[ok] * 9, fixes=[dict(kind="curb", at=[1.0, 0.0], why="random spot check", decision="drop", drop=True)],
              issues=[], tiles=[])  # fmt: skip
    cfg = dict(max_unsure=5, reject_spot_wrong=5)
    assert scene_state(rv, None, [feat], cfg)[0] == "needs_revision"  # the dropped curb is still in the vectors
    assert scene_state(rv, None, [], cfg)[0] == "accepted"
    bad = dict(rv, fixes=[dict(f, at=[float(i), 5.0]) for i, f in enumerate(rv["fixes"] * 5)])
    assert scene_state(bad, None, [], cfg)[0] == "rejected"
    geom = {"curb": {"profiles": 100, "step_within_0_3m": 0.1, "median_offset_m": 0.8}}
    assert scene_state(rv, geom, [], cfg)[0] == "rejected"
    assert scene_state(None, None, [], cfg, verifier_ok=False)[0] == "uncertain"


def test_annotation_plan_conditions():
    from hdlabel.agents import annotate

    reg = {"tools": {"vectorize": {"active": "default", "variants": {
        "default": {}, "dense": {"when": {"lidar_density": [30, None]}, "markings": {"curb_filter": {"enabled": True}}}}}}}  # fmt: skip
    assert annotate.plan(reg, {"lidar_density": 50})["vectorize"] == "dense"
    assert annotate.plan(reg, {"lidar_density": 10})["vectorize"] == "default"
    assert annotate.plan(reg, {})["vectorize"] == "default"  # unknown fact: the condition does not hold
    ov = annotate.overlay(reg, {"vectorize": "dense"})
    assert ov == {"markings": {"curb_filter": {"enabled": True}}}  # `when` is not a setting
    assert registry.overlay(registry.with_variant(reg, "vectorize", "dense")) == ov


def test_api_providers_request_and_parse(tmp_path, monkeypatch):
    """the OpenAI (Responses API) and Anthropic (Messages API) providers, with stand-in SDK modules: the request they
    build (model, system, image as base64, web search tool) and the parsing of the answer through the verifier."""
    import sys
    import types

    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8fake")
    answer = '{"items": {"0": {"decision": "off", "confidence": 0.7, "reason": "0.6 m left"}}}'
    sent = {}

    class OpenAI:
        def __init__(self, timeout):
            self.responses = self

        def create(self, **kw):
            sent["openai"] = kw
            return types.SimpleNamespace(output_text=answer, usage=types.SimpleNamespace(total_tokens=1234))

    class Messages:
        def create(self, **kw):
            sent["anthropic"] = kw
            u = types.SimpleNamespace(input_tokens=1000, output_tokens=200)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=answer)], usage=u)

    anth = types.SimpleNamespace(Anthropic=lambda timeout: types.SimpleNamespace(messages=Messages()), NOT_GIVEN=None)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setitem(sys.modules, "anthropic", anth)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    roles = dict(verifier=dict(provider="openai", model="gpt-x"), confirmer=dict(provider="anthropic", model="c-x"))
    price = dict(openai=10.0, anthropic={"c-x": [1.0, 5.0]})
    ag = Agents(dict(roles=roles, price_per_mtok=price), str(tmp_path / "agent"))
    rows = [dict(key="0", image=str(img), kind="curb", reason="spot", confidence=0.5, size_m=2.0)]
    for role, prov in (("verifier", "openai"), ("confirmer", "anthropic")):
        assert ag.available(role)
        r = judge_items(ag, rows, role)["0"]
        assert r["decision"] == "off" and r["confidence"] == 0.7 and r["model"] == f"{prov}/{roles[role]['model']}"
    o, a = sent["openai"], sent["anthropic"]
    assert (
        o["model"] == "gpt-x"
        and o["instructions"]
        and o["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    )
    assert a["model"] == "c-x" and a["system"] and a["messages"][0]["content"][0]["source"]["type"] == "base64"
    ag.ask("verifier", "find papers", tools=("WebSearch",), cache=False)
    assert sent["openai"]["tools"] == [{"type": "web_search"}]
    led = [json.loads(x) for x in open(tmp_path / "agent" / "ledger.jsonl")]
    assert led[0]["cost_usd"] == pytest.approx(1234 / 1e6 * 10.0)  # token providers priced by price_per_mtok
    assert led[1]["cost_usd"] == pytest.approx((1000 * 1.0 + 200 * 5.0) / 1e6)  # per model: input, output price
    # an SDK error (e.g. an expired key) becomes ProviderError and the verifier answers unsure, not a crash
    monkeypatch.setattr(OpenAI, "create", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("401 expired key")))
    assert judge_items(ag, [dict(rows[0], kind="lane_line")], "verifier")["0"]["decision"] == "unsure"


def test_plans_subscription_and_api(tmp_path, monkeypatch):
    """agents.plan picks a role set from the default config; subscription CLIs get no API keys."""
    from hdlabel.agents import llm
    from hdlabel.run import load_config

    cfg = load_config(None)["agents"]
    monkeypatch.delenv("HDLABEL_AGENT_PLAN", raising=False)
    sub = Agents(cfg, str(tmp_path / "s"))
    assert sub.plan == "subscription" and sub.model_of("verifier")["provider"] == "claude_cli"
    assert sub.model_of("confirmer")["provider"] == "codex_cli"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    out = llm._run(["sh", "-c", "echo ${ANTHROPIC_API_KEY:-none}"], 10)
    assert out.strip() == "none"  # the CLI uses the logged-in plan, not the key
    monkeypatch.setenv("HDLABEL_AGENT_PLAN", "api")
    api = Agents(dict(cfg, roles={"verifier": {"model": "claude-opus-5-5"}}), str(tmp_path / "a"))
    assert api.model_of("verifier") == dict(provider="anthropic", model="claude-opus-5-5", timeout=300)  # override
    assert api.model_of("confirmer")["provider"] == "openai" and api.model_of("researcher")["provider"] == "anthropic"
    assert llm._run(["sh", "-c", "echo ${ANTHROPIC_API_KEY:-none}"], 10).strip() == "secret"
    monkeypatch.setenv("HDLABEL_AGENT_PLAN", "nope")
    with pytest.raises(ValueError):
        Agents(cfg, str(tmp_path / "n"))


def test_bench_person_review(tmp_path, monkeypatch):
    """review page lists disputed elements first; the person's decisions replace the majority and lock the reference."""
    monkeypatch.setattr(bench, "bench_dir", lambda name: str(tmp_path / name))
    d = tmp_path / "t"
    os.makedirs(d / "evidence")
    items = [dict(key="0", kind="curb", group="curb", ref="ok", split="dev", size_m=5.0, at=[0.0, 0.0], why="spot",
                  evidence="evidence/a.jpg", judges={"a": "ok", "b": "ok"}),
             dict(key="1", kind="curb", group="curb", ref=None, split="dev", size_m=3.0, at=[9.0, 0.0], why="spot",
                  evidence="evidence/b.jpg", judges={"a": "ok", "b": "drop"})]  # fmt: skip
    (d / "reference.json").write_text(json.dumps(dict(items=items, judges=["a", "b"])))
    (d / "criteria.yaml").write_text(open(bench.CRITERIA).read())
    from hdlabel.agents import bench_review

    monkeypatch.setattr(bench_review, "bench_dir", lambda name: str(tmp_path / name))
    page = open(bench_review.review_page("t")).read()
    assert page.index('data-key="1"') < page.index('data-key="0"')  # disputed first
    with pytest.raises(ValueError):
        bench_review.apply_review("t", {"0": "wrong"}, "agent")
    st = bench_review.apply_review("t", {"0": "wrong", "1": "ok", "x": "ok"}, "person:test")
    assert st == dict(decided=2, changed=1, added=1, unsure=0)
    ref = json.load(open(d / "reference.json"))
    assert [e["ref"] for e in ref["items"]] == ["wrong", "ok"] and "person:test" in ref["judges"]
    assert bench.check_lock("t")["locked_by"] == "person:test"


def test_spec_gate_blocks_output_outside_the_spec():
    from hdlabel.agents.experiment import spec_gate

    g = dict(adopt=True, fails=[], target="marking.fp", gain=0.05, rows=[])
    line = {"type": "LineString", "coordinates": [[0, 0, 0], [1, 0, 0]]}
    bad = spec_gate(g, [{"type": "Feature", "geometry": line, "properties": {"kind": "made_up_kind"}}])
    assert not bad["adopt"] and "spec/labels.yaml" in bad["fails"][-1] and bad["spec_errors"] == 1
    assert spec_gate(g, []) == g  # conforming output: the benchmark decision stands
