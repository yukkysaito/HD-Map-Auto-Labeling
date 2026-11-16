"""review decisions -> review file (hdlabel/agent_review.py), and what the acceptance page computes from it."""

import json

import pytest

from hdlabel.agent_review import apply, signoff, summary
from hdlabel.audit import audit


def pack(tmp_path):
    d = tmp_path / "check"
    d.mkdir()
    items = [dict(n=k, kind="lane_line", group="marking", at=[float(k), 0.0], reason=r, image=f"item_{k:03d}.jpg")
             for k, r in enumerate(["low confidence", "low confidence", "random spot check", "random spot check"])]  # fmt: skip
    tiles = [dict(n=0, center=[5.0, 5.0], image="tile_0.jpg"), dict(n=1, center=[9.0, 9.0], image="tile_1.jpg")]
    for f in [i["image"] for i in items] + [t["image"] for t in tiles]:
        (d / f).write_bytes(b"jpg")
    (d / "items.json").write_text(json.dumps(dict(scene="s", items=items, tiles=tiles)))
    return str(d)


def test_apply_and_summary(tmp_path):
    d, rp = pack(tmp_path), str(tmp_path / "review" / "s.json")
    dec = dict(reviewer="agent:test", items={"0": "drop: shadow", "1": "ok", "2": "off: 0.6 m left", "3": "unsure"},
               tiles={"0": "ok", "1": "missed: dashed line"}, verdict="accept")  # fmt: skip
    with pytest.raises(ValueError):
        apply(d, dict(dec, items={"0": "ok"}), rp, str(tmp_path / "reviewed"))  # every item needs a decision
    rv = apply(d, dec, rp, str(tmp_path / "reviewed"))
    assert rv["fixes"] == [dict(kind="lane_line", at=[0.0, 0.0], why="low confidence", decision="drop", size_m=None,
                                evidence="reviewed/lane_line_0.00_0.00.jpg", drop=True, note="shadow")]  # fmt: skip
    assert len(rv["checked"]) == 3 and len(rv["issues"]) == 3 and rv["verdict"] == "fixed" and not rv["reject"]
    assert (tmp_path / "reviewed" / "tile_9.00_9.00.jpg").exists()
    s = summary(json.load(open(rp)))
    assert (s["low"], s["low_wrong"], s["spot"], s["spot_wrong"], s["unsure"]) == (2, 1, 2, 1, 1)
    assert (s["tiles"], s["tiles_missed"]) == (2, 1)
    # applying again (e.g. a second look) replaces, it does not duplicate
    rv = apply(d, dict(dec, items={"0": "ok", "1": "ok", "2": "ok", "3": "ok"}), rp, str(tmp_path / "reviewed"))
    assert not rv["fixes"] and len(rv["checked"]) == 4 and len(rv["issues"]) == 1  # the missed tile remains


def test_acceptance_needs_reviews_and_signoff_rejects(tmp_path):
    good = dict(curb=0.0, qa=0.0)
    rv = dict(reject=False, spot=10, spot_wrong=1)
    r = audit([dict(scene="a", terms=good, review=rv), dict(scene="b", terms=good, review=None)], sample=1)
    assert not r["accept"] and r["not_reviewed"] == ["b"] and (r["spot"], r["spot_wrong"]) == (10, 1)
    r = audit([dict(scene="a", terms=good, review=rv), dict(scene="b", terms=good, review=rv)], sample=1)
    assert r["accept"]
    files = {s: str(tmp_path / f"{s}.json") for s in "ab"}
    rej = signoff(dict(by="p", accept=True, reject={"b": "lines shifted"}), files.get, str(tmp_path / "signoff.json"))
    assert rej == ["b"] and json.load(open(files["b"]))["reject"] and (tmp_path / "signoff.json").exists()
