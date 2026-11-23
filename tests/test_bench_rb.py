"""Road boundary group of the benchmark: LiDAR kerb proposal, person decisions -> locked reference, metrics, gate,
and benchmarks without the group."""

import json
import os

import numpy as np
import pytest

from hdlabel.agents import bench, bench_rb
from hdlabel.surface import write_pcd


def _road(tmp_path, kerb=3.5, length=60.0, step=0.15):
    """a straight road along x (z = 0), kerbs at y = +-kerb rising `step` m to a flat sidewalk; dense points."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, length + 5, 400_000)
    y = rng.uniform(-14, 14, 400_000)
    z = np.where(np.abs(y) > kerb, step, 0.0) + rng.normal(0, 0.005, len(x))
    f = str(tmp_path / "map.pcd")
    write_pcd(f, np.c_[x, y, z])
    t = np.arange(0, length, 0.5)
    poses = np.c_[np.arange(len(t)) * 1e8, t, np.zeros_like(t), np.zeros_like(t), np.zeros((len(t), 4))]
    return f, poses


def test_walk_kerb_and_obstacle():
    reach = 6.0
    v = np.arange(-reach, reach, bench_rb.BIN) + bench_rb.BIN / 2
    h = np.where(v > 3.0, 0.15, 0.0)
    h = np.where(v < -2.0, 1.5, h)  # a wall on the right
    off, stop, _ = bench_rb.walk(h, 1)
    assert stop == "kerb" and abs(off - 3.0) < 0.15
    assert bench_rb.walk(h, -1)[:2] == (None, "high")
    assert bench_rb.walk(np.zeros_like(h), 1)[:2] == (None, "reach")


def test_propose_review_apply_evaluate(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "bench_dir", lambda name: str(tmp_path / name))
    mp, poses = _road(tmp_path)
    out = str(tmp_path / "draft")
    d = bench_rb.propose(mp, [dict(id="s1", drive="d", t_range=[0, 1e12])], lambda drv: poses, out, holdout=[],
                         win_m=30.0)  # fmt: skip
    assert len(d["windows"]) == 2 and all(w["split"] == "dev" for w in d["windows"])
    for w in d["windows"]:
        ys = sorted(np.mean([p[1] for p in ln["coords"]]) for ln in w["lines"])
        assert len(ys) == 2 and abs(ys[0] + 3.5) < 0.15 and abs(ys[1] - 3.5) < 0.15
        assert all(ln["source"] == "lidar_step" for ln in w["lines"])
    assert os.path.exists(f"{out}/review.html") and bench_rb.render_overlays(out)

    # an older benchmark without the group: unchanged digest, no road_boundary metrics
    b = tmp_path / "t"
    os.makedirs(b / "evidence")
    (b / "reference.json").write_text(json.dumps(dict(items=[])))
    (b / "criteria.yaml").write_text(open(bench.CRITERIA).read())
    bench.lock("t", "person:test")
    data0 = bench.check_lock("t")["data"]
    feat = lambda kind, c, **p: {"type": "Feature", "properties": dict(kind=kind, **p),  # noqa: E731
                                 "geometry": {"type": "LineString", "coordinates": c}}  # fmt: skip
    feats = [feat("road_boundary", [[x, 3.5, 0] for x in range(0, 60, 2)], observed=True),
             feat("road_boundary", [[x, -9.0, 0] for x in range(0, 60, 2)], observed=False),  # edge of the observed area
             feat("curb", [[x, -3.45, 0] for x in range(0, 30, 2)])]  # fmt: skip
    assert "road_boundary" not in bench.evaluate("t", feats)

    w0, w1 = d["windows"]
    dec = {w0["id"]: dict(complete=True, lines={w0["lines"][1]["id"]: "wrong"}, drawn=[[[1.0, -3.5], [28.0, -3.5]]]),
           w1["id"]: dict(complete=False)}  # fmt: skip
    with pytest.raises(ValueError):
        bench_rb.apply("t", out, dec, "agent")
    st = bench_rb.apply("t", out, dec, "person:test")
    assert st == dict(windows=2, complete=1, accepted=1, rejected=1, drawn=1)
    lk = bench.check_lock("t")
    assert lk["locked_by"] == "person:test" and lk["data"] != data0
    assert os.path.exists(b / "evidence" / f"rb_{w0['id']}.jpg")
    rb = json.load(open(b / "road_boundary.json"))
    assert {ln["source"] for w in rb["windows"] if w["complete"] for ln in w["lines"]} == {"lidar_step", "drawn"}

    m = bench.evaluate("t", feats)["road_boundary"]
    assert m["windows"] == 1 and m["precision"] > 0.95 and m["recall"] > 0.95 and m["ragged"] == 0.0
    zig = [[x * 0.25, 4.0 + 0.15 * (-1) ** x, 0] for x in range(0, 240)]  # ragged and 0.35-0.65 m off
    m2 = bench.evaluate("t", [feat("road_boundary", zig, observed=True), *feats[1:]])["road_boundary"]
    assert m2["ragged"] > 0.9 and m2["recall"] < 0.8  # only the drawn right kerb is covered (by the curb)
    assert bench.evaluate("t", feats, split="holdout").get("road_boundary") is None

    base, cand = {"all": dict(road_boundary=m2), "holdout": {}}, {"all": dict(road_boundary=m), "holdout": {}}
    assert bench.better("road_boundary.ragged") == "lower" and bench.better("road_boundary.recall") == "higher"
    assert bench.gate("t", base, cand, "road_boundary.ragged")["adopt"]
    assert not bench.gate("t", cand, base, "road_boundary.ragged")["adopt"]
    assert bench.gate("t", base, cand, "road_boundary.precision")["gain"] >= 0

    (b / "road_boundary.json").write_text(json.dumps(dict(rb, windows=[])))  # tampering is detected
    with pytest.raises(RuntimeError):
        bench.check_lock("t")
    with pytest.raises(RuntimeError):  # and blocks applying another review on top of it
        bench_rb.apply("t", out, dec, "person:test")


def test_ragged_and_within():
    from shapely.geometry import LineString

    assert bench.ragged_m([[0, 0], [10, 0]]) == (pytest.approx(10.0), 0.0)
    t, r = bench.ragged_m([[x * 0.2, 0.1 * (-1) ** x] for x in range(51)])
    assert r / t > 0.9
    a, b = LineString([(0, 0), (10, 0)]), LineString([(0, 0.2), (5, 0.2)])
    assert bench._within([a], [b], 0.3) == pytest.approx(0.52, abs=0.02)
