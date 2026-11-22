"""Marking road-level gate (hdlabel/markingcheck.py, vectorize variant marking_level) on a synthetic street."""

import json
import types

import numpy as np

from hdlabel import markingcheck
from hdlabel.agents import registry
from hdlabel.run import load_config, marking_gate_args
from hdlabel.steps import vectorize

G = ["road", "marking", "crosswalk", "curb", "sidewalk", "offroad", "other"]
C = {g: i + 1 for i, g in enumerate(G)}
RES = 0.1
LO = np.array([0.0, 0.0])


def street():
    """x along the road (0..60 m), y across: road 0..10 m (z 0, 2 % cross slope), kerb 10..10.3, raised sidewalk
    10.3..14 (z + 0.15 m) with tactile blocks segmented as marking at y 11..11.5 (x 5..15), a road painted area at
    y 4..6 (x 30..34), a lane line at y 5 (x 40..55), and a strip without LiDAR ground at x 20..25. Ego drives at y 3."""
    cls = np.zeros((600, 200), np.uint8)
    y = (np.arange(200) + 0.5) * RES
    cls[:, y < 10] = C["road"]
    cls[:, (y >= 10) & (y < 10.3)] = C["curb"]
    cls[:, (y >= 10.3) & (y < 14)] = C["sidewalk"]
    cls[:, y >= 14] = C["offroad"]
    cls[50:150, 110:115] = C["marking"]  # tactile blocks on the sidewalk
    cls[300:340, 40:60] = C["marking"]  # painted area on the road
    cls[400:550, 49:51] = C["marking"]  # lane line
    ys = (np.arange(100) + 0.5) * 0.2
    H = np.tile(np.where(ys < 10.1, 0.02 * ys, 0.02 * ys + 0.15), (300, 1))  # surface grid 0.2 m
    H[100:125] = np.nan  # x 20..25: no LiDAR ground
    traj = np.c_[np.linspace(0, 60, 61), np.full(61, 3.0)]
    return cls, H, [traj]


def box(i0, i1, j0, j1):
    return np.argwhere(np.ones((i1 - i0, j1 - j0), bool)) + [i0, j0]


def test_rules_on_street():
    cls, H, trajs = street()
    Hc = markingcheck.upsample(H, 2, cls.shape)
    lane = np.c_[np.arange(400, 550), np.full(150, 50)]
    area = box(300, 340, 40, 60)
    tactile = box(50, 150, 110, 115)
    no_lidar = np.c_[np.arange(200, 250), np.full(50, 112)]  # on the sidewalk but without LiDAR ground: not judged
    why, st, info = markingcheck.check_markings([lane, area, tactile, no_lidar], cls, C, LO, RES, Hc, trajs)
    assert info["trajectories"] == 1 and info["reachable_cells"] > 0
    assert why[0] == [] and why[1] == []  # paint on the road (the plane follows the cross slope)
    assert st[0]["on_level"] == 1.0
    assert {"off_level", "raised", "cut_off"} <= set(why[2])  # behind the kerb step
    assert st[2]["median_cm"] > 10
    assert why[3] == [] and st[3]["known"] == 0.0


def test_vectorize_variant(tmp_path):
    cls, H, trajs = street()
    n = int(cls.size)
    lab = tmp_path / "labels"
    lab.mkdir()
    votes = np.zeros((n, 8), np.float16)
    votes[np.arange(n), np.maximum(cls.ravel(), 1) - 1] = 1
    np.savez(lab / "ground_map.npz", cls=cls, height=H.astype(np.float32), height_w=np.isfinite(H).astype(np.float32),
             groups=np.array(G), origin=np.zeros(3), lo=LO, res=np.float64(RES), surf_res=np.float64(0.2),
             cells=np.arange(n), votes=votes, n=np.full(n, 30, np.float32), snap=np.bool_(False))  # fmt: skip
    gt = tmp_path / "gt.npy"
    np.save(gt, np.c_[np.arange(61), trajs[0], np.zeros(61)])  # t x y z
    vectorize.main([str(lab), "--out", str(tmp_path / "a")])
    vectorize.main([str(lab), "--out", str(tmp_path / "b"), "--marking-gate", "--gt", str(gt)])
    fa = json.load(open(tmp_path / "a" / "vectors.geojson"))["features"]
    fb = json.load(open(tmp_path / "b" / "vectors.geojson"))["features"]
    assert not any(f["properties"]["kind"] == "marking_candidate" for f in fa)
    cand = [f["properties"] for f in fb if f["properties"]["kind"] == "marking_candidate"]
    assert cand and all(p["candidate_of"] in ("marking_area", "lane_line") and p["marking_check"] for p in cand)
    kept = [f for f in fb if f["properties"]["kind"] in ("marking_area", "lane_line")]
    assert kept and all(f["properties"]["marking_level"]["on_level"] == 1.0 for f in kept)
    info_a = json.load(open(tmp_path / "a" / "vectors.json"))
    info_b = json.load(open(tmp_path / "b" / "vectors.json"))
    assert "marking_gate" not in info_a and "mark_tol" not in info_a["params"] and "gt" not in info_a["params"]
    assert info_b["marking_gate"]["candidates"] == len(cand)


def test_config_default_off_and_variant():
    reg = registry.with_variant(registry.load(), "vectorize", "default")
    cfg = load_config(None, registry=reg)
    area = types.SimpleNamespace(cfg=cfg)
    assert cfg["markings"]["marking_gate"]["enabled"] is False
    assert marking_gate_args(area, ["a/gt.npy"]) == []  # default: the vectorize command line is unchanged
    over = registry.load()["tools"]["vectorize"]["variants"]["marking_level"]
    area.cfg = registry._merge(cfg, over)
    args = marking_gate_args(area, ["a/gt.npy"])
    assert args[0] == "--marking-gate" and "--gt" not in args  # the curb filter of the variant passes --gt
    area.cfg["markings"]["curb_filter"]["enabled"] = False
    assert marking_gate_args(area, ["a/gt.npy"])[:3] == ["--marking-gate", "--gt", "a/gt.npy"]


def test_edge_line_along_the_kerb_is_judged_only_by_the_step():
    """an edge line next to the kerb reads ~5 cm above the road plane (kerb and gutter mixed into its surface cells):
    off_level with kerb_m = 0, kept with kerb_m > 0; the tactile blocks behind the kerb step stay raised."""
    cls, H, trajs = street()
    cls[:, 97:100] = C["marking"]  # edge line at y 9.7..10 along the kerb
    H[:, 48:50] += 0.05  # its cells read 5 cm high
    Hc = markingcheck.upsample(H, 2, cls.shape)
    edge = np.c_[np.repeat(np.arange(300, 400), 3), np.tile([97, 98, 99], 100)]
    tactile = box(50, 150, 110, 115)
    why, st, _ = markingcheck.check_markings([edge, tactile], cls, C, LO, RES, Hc, trajs)
    assert why[0] == ["off_level"] and "kerb" not in st[0]
    why, st, _ = markingcheck.check_markings([edge, tactile], cls, C, LO, RES, Hc, trajs, dict(kerb_m=0.5))
    assert why[0] == [] and st[0]["kerb"] == 1.0
    assert "raised" in why[1]
