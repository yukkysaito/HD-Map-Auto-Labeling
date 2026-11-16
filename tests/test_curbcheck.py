"""Curb filter (hdlabel/curbcheck.py, vectorize variant curb_adjacency) on a synthetic street cross-section."""

import types

import numpy as np

from hdlabel import curbcheck
from hdlabel.agents import registry
from hdlabel.run import curb_args, load_config

G = ["road", "marking", "crosswalk", "manhole", "curb", "sidewalk", "offroad", "other"]
C = {g: i + 1 for i, g in enumerate(G)}
RES = 0.1
LO = np.array([0.0, 0.0])


def street():
    """x along the road (0..60 m), y across: road 0..10 m, kerb 10..10.3, sidewalk 10.3..13, hedge 13..14,
    plaza-side planter border 14..14.3, lawn behind; the ego drives at y = 3."""
    cls = np.zeros((600, 200), np.uint8)
    y = (np.arange(200) + 0.5) * RES
    cls[:, y < 10] = C["road"]
    cls[:, (y >= 10) & (y < 10.3)] = C["curb"]
    cls[:, (y >= 10.3) & (y < 13)] = C["sidewalk"]
    cls[:, (y >= 13) & (y < 14)] = C["offroad"]
    cls[:, (y >= 14) & (y < 14.3)] = C["curb"]
    cls[:, y >= 14.3] = C["offroad"]
    cls[200:230, 113:118] = C["curb"]  # piece on the sidewalk at a bollard (x 20..23, y 11.3..11.8)
    traj = np.c_[np.linspace(0, 60, 61), np.full(61, 3.0)]
    return cls, [traj]


def test_rules_on_street():
    cls, trajs = street()
    kerb = np.c_[np.linspace(1, 59, 59), np.full(59, 10.15)]
    planter = np.c_[np.linspace(1, 59, 59), np.full(59, 14.15)]
    bollard = np.c_[[20.0, 23.0], [11.55, 11.55]]
    t = np.linspace(0, 2 * np.pi, 40)
    loop = np.c_[30 + 0.8 * np.cos(t), 5 + 0.8 * np.sin(t)]  # a drainage grating ring on the road
    xs = np.linspace(40, 50, 41)
    wavy = np.c_[xs, 16 + 0.6 * np.sign(np.sin(xs * np.pi))]  # zig-zag through the lawn
    why, st = curbcheck.check_curbs([kerb, planter, bollard, loop, wavy], cls, C, LO, RES, trajs)
    assert st["trajectories"] == 1 and st["reach_cells"] > 0
    assert why[0] == []  # the road kerb is kept
    assert "not_road_adjacent" in why[1]  # behind sidewalk + hedge
    assert "not_road_adjacent" in why[2]
    assert "loop" in why[3]
    assert "wavy" in why[4]


def test_long_rings_and_no_trajectory():
    cls, trajs = street()
    t = np.linspace(0, 2 * np.pi, 200)
    island = np.c_[30 + 5 * np.cos(t), 5 + 3 * np.sin(t)]  # a kerbed traffic island ring (~25 m): not a "loop"
    assert "loop" not in curbcheck.shape_reasons(island, curbcheck.DEFAULTS)
    planter = np.c_[np.linspace(1, 59, 59), np.full(59, 14.15)]
    why, st = curbcheck.check_curbs([planter], cls, C, LO, RES, [])  # no trajectory: shape / edge rules only
    assert why == [[]] and st["reach_cells"] == 0


def test_config_default_off_and_variant():
    reg = registry.with_variant(registry.load(), "vectorize", "default")  # whatever variant is active now
    cfg = load_config(None, registry=reg)
    area = types.SimpleNamespace(cfg=cfg)
    assert cfg["markings"]["curb_filter"]["enabled"] is False
    assert curb_args(area, ["a/gt.npy"]) == []  # default: the vectorize command line is unchanged
    over = registry.load()["tools"]["vectorize"]["variants"]["curb_adjacency"]
    area.cfg = registry._merge(cfg, over)
    args = curb_args(area, ["a/gt.npy", "b/gt.npy"])
    assert args[:4] == ["--curb-filter", "--gt", "a/gt.npy", "b/gt.npy"]
