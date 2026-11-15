"""acceptance audit (hdlabel/audit.py): risk ranking, reproducible random sample, batch verdict."""

from hdlabel.audit import audit, risk


def test_risk_and_verdict():
    good = dict(curb=0.0, height=0.0, qa=0.0, calib=0.0)
    bad = dict(curb=0.5, height=0.2, qa=0.8, calib=1.0)
    assert risk(good) == 0.0 and risk(bad) > 0.5
    scenes = [dict(scene=f"s{i}", terms=good) for i in range(8)] + [dict(scene="x", terms=bad)]
    r = audit(scenes, sample=3, seed=1, max_risk=0.3)
    assert r["scenes"][0]["scene"] == "x" and not r["accept"] and r["over_max_risk"] == ["x"]
    assert len(r["random_sample"]) == 3
    assert audit([dict(s) for s in scenes], sample=3, seed=1)["random_sample"] == r["random_sample"]  # reproducible
