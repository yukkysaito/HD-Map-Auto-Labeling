"""Road-level refinement of the road class (hdlabel/roadlevel.py)."""

import copy
import os
import types

import numpy as np

from hdlabel.common import CLASSES
from hdlabel.roadlevel import DEFAULTS, refine, refine_scene
from hdlabel.run import load_config, projection_args

ROAD, CURB, SIDEWALK, OFFROAD = (CLASSES.index(c) for c in ("road", "curb", "sidewalk", "offroad"))
RES, K = 0.1, 2  # label grid 0.1 m, surface grid 0.2 m


def scene(step=0.15, ramp_m=1.0, strip_m=2.5):
    """20 x 20 m of road labels, driven along x = 15 m; a strip x = 4 - strip_m .. 4 m (off-road before it) lies `step`
    above the road behind a kerb that rises over ramp_m (the surface fit smooths a kerb over ~1 m), curb labels at
    x = 4.0..4.5 m with a 0.5 m label gap (the strip touches the road in the labels)."""
    cls = np.full((200, 200), ROAD, np.uint8)
    cls[: int(round((4 - strip_m) / RES))] = OFFROAD
    cls[40:45, :] = CURB
    cls[40:45, 100:105] = ROAD
    x = (np.arange(100) + 0.5) * RES * K
    h = np.interp(x, [0.0, 4.0, 4.0 + ramp_m, 20.0], [step, step, 0.0, 0.0])
    H = np.repeat(h[:, None], 100, 1)
    traj = np.c_[np.full(50, 15.0), np.linspace(0.5, 19.5, 50), np.zeros(50)]
    return cls, H, traj


def test_raised_strip_behind_a_kerb_becomes_sidewalk():
    cls, H, traj = scene()
    out, st = refine(cls, H, K, np.zeros(2), RES, [traj])
    assert (out[15:38] == SIDEWALK).all() and st["raised"] == 1 and (out[:15] == OFFROAD).all()
    assert (out[48:] == ROAD).all() and (out[40:45, :100] == CURB).all()  # road from the foot of the kerb, curb kept


def test_wide_raised_area_abstains():
    cls, H, traj = scene(strip_m=4.0)  # a plaza, car park or side street up a lowered kerb: road or not, unknown
    out, st = refine(cls, H, K, np.zeros(2), RES, [traj])
    assert (out[:38] == 0).all() and st["abstain_m2"] > 0 and (out[48:] == ROAD).all()


def test_road_level_strip_stays_road():
    cls, H, traj = scene(step=0.0)
    out, st = refine(cls, H, K, np.zeros(2), RES, [traj])
    assert (out == cls).all() and st["raised"] == 0


def test_lower_area_gentle_slope_and_high_step_stay_road():
    for step, ramp in ((-0.25, 1.0), (0.4, 8.0), (0.5, 1.0)):  # sunken (a lower street?), a 5 % ramp, a wall / other
        # level: no kerb, no decision
        cls, H, traj = scene(step, ramp)
        out, _ = refine(cls, H, K, np.zeros(2), RES, [traj])
        assert (out == cls).all(), (step, ramp)


def test_driven_hump_and_no_trajectory_unchanged():
    cls, H, traj = scene()
    out, _ = refine(cls, H, K, np.zeros(2), RES, [])
    assert (out == cls).all()
    H2 = H.copy()
    H2[70:80, 40:60] = 0.12  # a steep-sided hump the vehicle drove over
    out, _ = refine(cls, H2, K, np.zeros(2), RES, [traj])
    assert (out[140:160, 80:120] == ROAD).all() and (out[15:38] == SIDEWALK).all()


def test_refine_scene_reads_ground_and_trajectories(tmp_path):
    cls, H, traj = scene()
    origin = np.array([1000.0, 2000.0, 5.0])
    w = np.ones_like(H)
    w[:, :5] = 0  # no LiDAR ground there
    np.savez(tmp_path / "ground.npz", height=H, height_w=w, surf_res=K * RES)
    (tmp_path / "d").mkdir()
    np.save(tmp_path / "d" / "gt.npy", np.c_[np.arange(len(traj)), traj + origin, np.zeros((len(traj), 4))])
    out, st = refine_scene(cls, str(tmp_path / "ground.npz"), [str(tmp_path / "d")], origin, np.zeros(2), RES)
    assert (out[15:38, 20:] == SIDEWALK).all() and (out[50:, :10] == ROAD).all() and st["seeds"] > 0


def test_kerb_face_alone_is_not_raised():
    cls, H, traj = scene()
    cls[:40] = SIDEWALK  # the segmentation got the sidewalk, but no curb: road labels all up the smoothed kerb face
    cls[40:45] = ROAD
    out, st = refine(cls, H, K, np.zeros(2), RES, [traj])
    assert (out == cls).all() and st["raised"] == 0


def test_config_default_off():
    area = types.SimpleNamespace(cfg=load_config(None))
    assert area.cfg["projection"]["road_level"]["enabled"] is False
    assert "--road-level" not in projection_args(area)  # default: the project_labels command line is unchanged
    area.cfg = copy.deepcopy(area.cfg)
    area.cfg["projection"]["road_level"] = dict(enabled=True, step_m=0.1)
    args = projection_args(area)
    i = args.index("--road-level")
    assert args[i + 1 : i + 1 + len(DEFAULTS)] == [0.1 if k == "step_m" else v for k, v in DEFAULTS.items()]
    step = os.path.join(os.path.dirname(__file__), "..", "hdlabel", "steps", "project_labels.py")
    assert f'"--road-level",\n        nargs={len(DEFAULTS)},' in open(step).read()  # the step takes all DEFAULTS
