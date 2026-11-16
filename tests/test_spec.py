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
