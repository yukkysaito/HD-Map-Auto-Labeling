"""Road boundary shape (hdlabel/boundary.py, vectorize.py --boundary-shape) on synthetic rings and a street."""

import json
import types

import numpy as np

from hdlabel import boundary
from hdlabel.run import boundary_args, load_config
from hdlabel.steps import vectorize

G = ["road", "marking", "crosswalk", "curb", "sidewalk", "offroad", "other"]
C = {g: i + 1 for i, g in enumerate(G)}
RES = 0.1


def rect(w=20.0, h=10.0):
    return boundary.resample_ring(np.array([[0, 0], [w, 0], [w, h], [0, h]], float), RES)


def test_zigzag_clean_shapes_are_zero():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    for r in (1.5, 3.0, 10.0):  # small islands, kerb corners, curves: turning one way only
        ring = boundary.resample_ring(np.c_[r * np.cos(t), r * np.sin(t)], RES)
        assert boundary.ring_zigzag(ring, RES).max() < 1.0
    assert boundary.ring_zigzag(rect(), RES).max() < 1.0  # straight edges and 90 degree corners
    snapped = np.round(rect() / RES) * RES  # raster outline
    assert boundary.ring_zigzag(snapped, RES).max() < 1.0


def test_zigzag_flags_only_the_ragged_stretch():
    ring = rect()
    m = (ring[:, 1] < 1e-6) & (ring[:, 0] > 5) & (ring[:, 0] < 15)
    ring[m, 1] = 0.2 * np.sin(2 * np.pi * ring[m, 0])  # 0.2 m zig-zag every metre on one side
    z = boundary.ring_zigzag(ring, RES)
    assert z[m].mean() > 100
    far = (ring[:, 1] > 1.0) | (ring[:, 0] < 3) | (ring[:, 0] > 17)
    assert z[far].max() < 1.0


def test_smooth_ring():
    ring = rect()
    m = (ring[:, 1] < 1e-6) & (ring[:, 0] > 5) & (ring[:, 0] < 15)
    ring[m, 1] = 0.1 * np.sin(2 * np.pi * ring[m, 0] / 0.6)  # small wiggles of a real edge
    sm = boundary.smooth_ring(ring, 0.5 / RES)
    assert np.abs(sm[m, 1]).max() < 0.02
    straight = (np.abs(ring[:, 1] - 10.0) < 1e-6) & (ring[:, 0] > 3) & (ring[:, 0] < 17)
    assert np.allclose(sm[straight], ring[straight], atol=1e-6)  # a straight edge stays where it is
    assert np.allclose(boundary.smooth_ring(ring, 0), ring)


def test_drop_short_runs():
    f = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1], bool)
    assert boundary.drop_short_runs(f, 3).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # closed: the run 11 -> 0 -> 1 wraps around the end (3 points)
    assert boundary.drop_short_runs(f, 3, closed=True).tolist() == [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert boundary.drop_short_runs(np.ones(12, bool), 9, closed=True).all()  # one whole ring


def street():
    """x along the road (0..60 m), y across: road below a sidewalk. x < 30: a straight kerb edge at y 10; x >= 30:
    road-coloured paving against scattered sidewalk cells, the edge zig-zags 0.5 m every 0.75 m."""
    cls = np.zeros((600, 200), np.uint8)
    x = (np.arange(600) + 0.5) * RES
    y = (np.arange(200) + 0.5) * RES
    edge = np.where(x < 30, 10.0, 10.0 + 0.5 * np.sign(np.sin(2 * np.pi * x / 1.5)))
    road = y[None, :] < edge[:, None]
    cls[road] = C["road"]
    cls[~road] = C["sidewalk"]
    return cls


def boundary_props(path):
    return [f for f in json.load(open(path))["features"] if f["properties"]["kind"] == "road_boundary"]


def boundary_runs(path):
    return [(np.array(f["geometry"]["coordinates"])[:, :2], f["properties"]["observed"]) for f in boundary_props(path)]


def dense(xy):
    return np.concatenate([np.linspace(p, q, 50, endpoint=False) for p, q in zip(xy[:-1], xy[1:])] + [xy[-1:]])


def observed_length(runs, x0, x1):
    """observed boundary length with x in [x0, x1) near the edge (y 9..11)."""
    out = 0.0
    for xy, obs in runs:
        if not obs:
            continue
        xy = dense(xy)
        mid = (xy[1:] + xy[:-1]) / 2
        k = (mid[:, 0] >= x0) & (mid[:, 0] < x1) & (mid[:, 1] > 9) & (mid[:, 1] < 11)
        out += float(np.linalg.norm(np.diff(xy, axis=0), axis=1)[k].sum())
    return out


def test_vectorize_boundary_shape(tmp_path):
    cls = street()
    n = int(cls.size)
    lab = tmp_path / "labels"
    lab.mkdir()
    votes = np.zeros((n, 8), np.float16)
    votes[np.arange(n), np.maximum(cls.ravel(), 1) - 1] = 1
    H = np.zeros((300, 100), np.float32)
    np.savez(lab / "ground_map.npz", cls=cls, height=H, height_w=np.ones_like(H), groups=np.array(G),
             origin=np.zeros(3), lo=np.zeros(2), res=np.float64(RES), surf_res=np.float64(0.2), cells=np.arange(n),
             votes=votes, n=np.full(n, 30, np.float32), snap=np.bool_(False))  # fmt: skip
    vectorize.main([str(lab), "--out", str(tmp_path / "a")])
    vectorize.main([str(lab), "--out", str(tmp_path / "b"), "--boundary-shape"])
    a, b = boundary_runs(tmp_path / "a" / "vectors.geojson"), boundary_runs(tmp_path / "b" / "vectors.geojson")
    assert observed_length(a, 33, 57) > 24  # default: the zig-zag outline is an observed boundary
    assert observed_length(b, 33, 57) == 0  # --boundary-shape: unknown
    assert 23 < observed_length(b, 3, 27) <= 24.5  # the straight kerb edge stays
    straight = np.concatenate([dense(xy) for xy, obs in b if obs])
    straight = straight[(straight[:, 0] > 2) & (straight[:, 0] < 28)]
    assert len(straight) and np.abs(straight[:, 1] - 10.0).max() < 0.15  # on the edge
    # reason: only with --boundary-shape, on every unknown run; the zig-zag stretch has no consistent edge, the
    # outline along the grid border (unobserved outside) is the edge of the view
    assert not any("reason" in f["properties"] for f in boundary_props(tmp_path / "a" / "vectors.geojson"))
    fb = boundary_props(tmp_path / "b" / "vectors.geojson")
    assert all(("reason" in f["properties"]) != f["properties"]["observed"] for f in fb)
    why = {}
    for f in fb:
        if not f["properties"]["observed"]:
            xy = dense(np.array(f["geometry"]["coordinates"])[:, :2])
            on_zigzag = (xy[:, 0] > 33) & (xy[:, 0] < 57) & (np.abs(xy[:, 1] - 10) < 1)
            why.setdefault(f["properties"]["reason"], []).append(on_zigzag.mean())
    assert set(why) == {"edge_of_view", "no_consistent_edge"}
    assert max(why["edge_of_view"]) < 0.1 and max(why["no_consistent_edge"]) > 0.5
    pa = json.load(open(tmp_path / "a" / "vectors.json"))["params"]
    pb = json.load(open(tmp_path / "b" / "vectors.json"))["params"]
    assert "boundary_shape" not in pa and "boundary_smooth" not in pa  # default: the parameters as before
    assert pb["boundary_shape"] is True and pb["boundary_zigzag"] == 40.0


def test_config_default_off():
    cfg = load_config(None)
    area = types.SimpleNamespace(cfg=cfg)
    assert cfg["markings"]["boundary_shape"]["enabled"] is False
    assert boundary_args(area) == []  # default: the vectorize command line is unchanged
    cfg["markings"]["boundary_shape"]["enabled"] = True
    args = boundary_args(area)
    assert args[0] == "--boundary-shape" and args[args.index("--boundary-zigzag") + 1] == 40.0
    assert args[args.index("--boundary-min-run") + 1] == 3.0
