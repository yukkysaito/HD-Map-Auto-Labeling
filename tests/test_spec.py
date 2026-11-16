"""The label spec (spec/labels.yaml): rules rendered for the verifier, conformance of vector features."""

from hdlabel import spec
from hdlabel.agents.verify import prompt_version, system_prompt


def test_rules_from_spec_reach_the_verifier():
    sp = spec.load()
    text = system_prompt()
    assert "{SPEC_RULES}" not in text
    for k, v in sp["kinds"].items():
        assert (f"\n{k} (" in text) == (v["group"] in ("marking", "curb"))  # verified kinds only
    assert "<table>" in spec.rules(html=True)


def test_spec_change_changes_the_prompt_version(tmp_path, monkeypatch):
    v0 = prompt_version()
    p = tmp_path / "labels.yaml"
    p.write_text(open(spec.PATH).read().replace("as painted.", "as painted, changed.", 1))
    monkeypatch.setattr(spec, "PATH", str(p))
    monkeypatch.setattr(spec.load, "__defaults__", (str(p),))
    assert prompt_version() != v0  # the next run verifies again under the new rules


def test_conform():
    sp = spec.load()
    line = {"type": "LineString", "coordinates": [[0, 0, 0], [1, 0, 0]]}
    ok = {
        "type": "Feature",
        "geometry": line,
        "properties": dict(kind="curb", length=1.0, confidence=0.9, observations=3),
    }
    assert spec.conform([ok], sp) == ([], [])
    miss = {"type": "Feature", "geometry": line, "properties": dict(kind="curb", length=1.0)}
    poly = {"type": "Feature", "geometry": line, "properties": dict(kind="crosswalk", confidence=1, observations=1)}
    odd = {"type": "Feature", "geometry": line, "properties": dict(kind="hedge")}
    extra = {"type": "Feature", "geometry": line, "properties": dict(ok["properties"], colour="red")}
    errs, warns = spec.conform([miss, poly, odd, extra], sp)
    assert any("'confidence' missing" in e for e in errs) and any("geometry LineString" in e for e in errs)
    assert any("hedge" in e for e in errs) and warns == ["curb: attribute 'colour' is not in the spec"]


def test_spec_loop_end_to_end(tmp_path_factory, monkeypatch):
    """spec change -> engineer (stand-in) edits the tools in a worktree -> checks -> merge -> aligned; an engineer that
    edits the spec itself is rejected."""
    import json
    import subprocess
    import types

    from hdlabel.agents import orchestrator, registry
    from hdlabel.agents.knowledge import Knowledge
    from hdlabel.agents.llm import Reply

    tmp_path = tmp_path_factory.mktemp("repo")
    git = lambda *c: subprocess.run(["git", *c], cwd=tmp_path, check=True, capture_output=True, text=True).stdout  # noqa: E731
    (tmp_path / "spec").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "spec" / "labels.yaml").write_text("kinds: {}\n")
    (tmp_path / "tools" / "registry.yaml").write_text("tools: {}\n")
    (tmp_path / "tool.py").write_text("X = 1\n")
    (tmp_path / ".gitignore").write_text("output/\n")
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    for k, v in (
        ("GIT_AUTHOR_NAME", "t"),
        ("GIT_AUTHOR_EMAIL", "t@t"),
        ("GIT_COMMITTER_NAME", "t"),
        ("GIT_COMMITTER_EMAIL", "t@t"),
    ):
        monkeypatch.setenv(k, v)
    sp = str(tmp_path / "spec" / "labels.yaml")
    rp = str(tmp_path / "tools" / "registry.yaml")
    monkeypatch.setattr(orchestrator, "ROOT", str(tmp_path))
    monkeypatch.setattr(
        spec, "version", lambda path=sp: __import__("hashlib").sha1(open(path, "rb").read()).hexdigest()[:10]
    )
    monkeypatch.setattr(registry, "load", lambda path=rp: __import__("yaml").safe_load(open(path)))
    monkeypatch.setattr(registry, "save", lambda reg, path=rp: __import__("yaml").safe_dump(reg, open(path, "w")))
    monkeypatch.setattr(orchestrator, "check_worktree", lambda wt: dict(ok=True, lint=True, tests="1 passed"))
    orchestrator.mark_aligned("person:test")
    git("commit", "-qam", "aligned")
    assert not orchestrator.spec_state()["changed"]

    (tmp_path / "spec" / "labels.yaml").write_text("kinds: {lane_line: {attributes: {piece_index: order}}}\n")
    git("commit", "-qam", "spec change")
    st = orchestrator.spec_state()
    assert st["changed"] and "piece_index" in st["diff"]

    def engineer(touch_spec):
        def ask(role, task, system="", tools=(), cache=True, cwd=None):
            assert role == "engineer" and "piece_index" in task and cwd
            open(f"{cwd}/tool.py", "w").write("X = 2  # piece_index\n")
            if touch_spec:
                open(f"{cwd}/spec/labels.yaml", "a").write("# relaxed\n")
            subprocess.run(["git", "commit", "-qam", "spec: piece_index"], cwd=cwd, check=True)
            return Reply(json.dumps(dict(summary="piece_index added", open_points=[])), "replay", None)

        return types.SimpleNamespace(agents=types.SimpleNamespace(ask=ask), name="t", work=str(tmp_path))

    kb = Knowledge(str(tmp_path_factory.mktemp("kb") / "kb.sqlite"))
    bad = orchestrator.spec_loop(engineer(True), kb)
    assert (
        not bad["merged"] and bad["protected"] == ["spec/labels.yaml"] and "X = 1" in (tmp_path / "tool.py").read_text()
    )
    good = orchestrator.spec_loop(engineer(False), kb)
    assert good["merged"] and "piece_index" in (tmp_path / "tool.py").read_text()
    assert not orchestrator.spec_state()["changed"]  # recorded as aligned
    assert [r["status"] for r in kb.list(kind="spec")] == ["done", "failed"]
