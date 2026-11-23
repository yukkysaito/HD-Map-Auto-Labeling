"""Drive alignment before fusion (hdlabel/steps/project_labels.py drive_align / --align) on synthetic markings: the
correction undoes a known pose error, an unconstrained shift is flagged ambiguous, and a whole drive projected with a
pose error is moved back onto the reference drive and fused (where --max-shift alone rejects it)."""

import json
import os
import types

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hdlabel.common import CLASSES, GROUPS
from hdlabel.mapproj import Projector
from hdlabel.run import load_config, projection_args
from hdlabel.steps import project_labels as PL

ORIGIN = np.array([1000.0, 2000.0, 5.0])  # map frame offset (like real maps); the road is along +x at y = 0 (relative)
ROAD, MARK = GROUPS["road"][0], GROUPS["marking"][0]


def paint(x, y, lanes_only=False):
    """marking at map xy relative to ORIGIN: dashed lane lines (3 m every 8 m) and solid edges along x, a stop line and
    crosswalk stripes; lanes_only: just short dashes (1 m every 2 m), which fit equally well every 2 m along the road."""
    ax, ay = np.abs(y - 1.75), np.abs(y + 1.75)
    if lanes_only:
        return (np.minimum(ax, ay) < 0.075) & (np.mod(x, 2.0) < 1.0)
    m = (np.minimum(ax, ay) < 0.075) & (np.mod(x, 8.0) < 3.0)
    m |= np.abs(np.abs(y) - 5.25) < 0.075
    m |= (np.abs(x - 41.0) < 0.22) & (y > -5.25) & (y < 0)  # stop line
    m |= (x > 45) & (x < 49) & (np.mod(y + 5.25, 0.9) < 0.45) & (np.abs(y) < 5.25)  # crosswalk stripes
    return m


def ground(tmp_path):
    H = np.zeros((450, 125), np.float32)  # 0.2 m surface: x -10 .. 80, y -12.5 .. 12.5
    f = str(tmp_path / "g.npz")
    np.savez(f, height=H, height_w=np.ones_like(H), lo=np.array([-10.0, -12.5]), surf_res=0.2, origin=ORIGIN)
    return f


def acc_of(pj, T=None, lanes_only=False):
    """accumulator observing every cell; marking where the paint is, moved by the 3x3 map-frame transform T."""
    xy = np.stack(np.unravel_index(pj.cells, pj.Lshape), 1) * pj.res + pj.lo + 0.5 * pj.res
    if T is not None:  # the content moved by T: a cell shows the paint at T^-1(cell)
        xy = (xy - T[:2, 2]) @ np.linalg.inv(T[:2, :2]).T
    acc = pj.new_acc()
    m = torch.from_numpy(paint(xy[:, 0], xy[:, 1], lanes_only)).to(pj.dev)
    acc["votes"][:, list(GROUPS).index("road")] = (~m).float()
    acc["votes"][:, list(GROUPS).index("marking")] = m.float()
    return acc


def rot(deg, t, about=(0.0, 0.0)):
    G = PL.rigid(t[0], t[1], deg, np.asarray(about) + ORIGIN[:2])[[0, 1, 3]][:, [0, 1, 3]]
    G[:2, 2] -= ORIGIN[:2] - G[:2, :2] @ ORIGIN[:2]  # same transform in coordinates relative to ORIGIN
    return G


def test_drive_align_undoes_a_known_error(tmp_path):
    pj = Projector(ground(tmp_path))
    ref = acc_of(pj)
    for deg, t in ((0.0, (0.63, -0.42)), (0.35, (-1.2, 0.5)), (-0.2, (0.3, 1.1))):
        E = rot(deg, t, about=(20.0, 3.0))
        al = PL.drive_align(pj, ref, acc_of(pj, E), 0.05, max_m=2.0, rot_deg=0.6, rot_step=0.15)
        assert al["in_range"] and al["ambiguity"] < 0.9, al
        G = PL.rigid(al["dx_m"], al["dy_m"], al["rot_deg"], al["pivot"])
        pts = np.array([[0.0, -5.0], [70.0, 5.0], [41.0, -3.0], [10.0, 5.0]])
        moved = (pts @ E[:2, :2].T + E[:2, 2]) + ORIGIN[:2]  # where the error put these marking points (map frame)
        back = moved @ G[:2, :2].T + G[:2, 3] - ORIGIN[:2]
        assert np.abs(back - pts).max() < 0.06, (deg, t, al, back - pts)
        assert al["overlap_best"] > 2 * al["overlap_zero"]


def test_drive_align_flags_what_it_cannot_measure(tmp_path):
    pj = Projector(ground(tmp_path))
    ref = acc_of(pj, lanes_only=True)
    E = rot(0.0, (0.0, 0.5))
    al = PL.drive_align(pj, ref, acc_of(pj, E, lanes_only=True), 0.05)
    assert al["in_range"] and al["ambiguity"] > 0.9  # the dashes fit as well 2 m farther along the road
    ref = acc_of(pj)
    for t in ((2.4, 0.0), (3.0, 1.0)):  # beyond max_m; beyond the search window
        assert not PL.drive_align(pj, ref, acc_of(pj, rot(0.0, t)), 0.05, max_m=2.0)["in_range"]
    few = acc_of(pj)
    few["votes"][:, list(GROUPS).index("marking")] = 0
    assert "dx_m" not in PL.drive_align(pj, ref, few, 0.05)


def cam_T(x, y, yaw):
    """map-frame base_link pose (relative to ORIGIN) of a vehicle at (x, y, yaw)."""
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, 0.0]
    return T


K = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
PITCH = np.radians(25.0)
R_BASE_CAM = np.array([[0, -np.sin(PITCH), np.cos(PITCH)], [-1, 0, 0], [0, -np.cos(PITCH), -np.sin(PITCH)]])
T_BASE_CAM = np.array([1.5, 0.0, 1.6])


def write_drive(root, name, lane_y, E):
    """one camera looking at the road from a vehicle driving along +x at lane_y; the images show the true paint, the
    recorded poses (gt.npy) carry the pose error E (3x3, relative to ORIGIN)."""
    d = f"{root}/{name}"
    for sub in ("kf_cam0", "seg_cam0", "und_cam0"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    json.dump(dict(K=K.tolist()), open(f"{d}/und_cam0/K.json", "w"))
    v, u = np.mgrid[0:240, 0:320]
    gt = []
    for k, x in enumerate(np.arange(0.0, 60.0, 2.0)):
        st = 10**18 + k * 10**8
        Tb = cam_T(x, lane_y + 0.3 * np.sin(x / 9), 0.04 * np.cos(x / 7))
        Rc, C = Tb[:3, :3] @ R_BASE_CAM, Tb[:3, :3] @ T_BASE_CAM + Tb[:3, 3]
        dirs = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], -1).astype(float) @ np.linalg.inv(K).T @ Rc.T
        t = np.where(dirs[..., 2] < -0.01, -C[2] / np.minimum(dirs[..., 2], -0.01), np.nan)
        P = C[:2] + t[..., None] * dirs[..., :2]
        seg = np.where(paint(P[..., 0], P[..., 1]), MARK, ROAD).astype(np.uint8)
        seg[np.isnan(t)] = 0
        cv2.imwrite(f"{d}/seg_cam0/{st}.png", seg)
        cv2.imwrite(f"{d}/kf_cam0/{st}.jpg", np.where(seg[..., None] == MARK, 230, 90).astype(np.uint8).repeat(3, 2))
        Te = np.eye(4)
        Te[:2, :2], Te[:2, 3] = E[:2, :2], E[:2, 2]
        Tr = Te @ Tb
        q = Rotation.from_matrix(Tr[:3, :3]).as_quat()
        gt.append([st, *(Tr[:3, 3] + ORIGIN), *q])
    np.save(f"{d}/gt.npy", np.array(gt))
    return d


def test_misplaced_drive_is_aligned_and_fused(tmp_path):
    g = ground(tmp_path)
    os.makedirs(tmp_path / "calib")
    json.dump(dict(R_base_cam=R_BASE_CAM.tolist(), t_base_cam=T_BASE_CAM.tolist(), time_offset_s=0.0),
              open(tmp_path / "calib/cam0.json", "w"))  # fmt: skip
    ref = write_drive(tmp_path, "ref", -1.75, np.eye(3))
    E = rot(0.25, (0.7, -0.3), about=(30.0, 0.0))
    bad = write_drive(tmp_path, "bad", 1.75, E)
    common = ["--ground", g, "--drive", ref, str(tmp_path / "calib"), "--drive", bad, str(tmp_path / "calib"),
              "--no-snap", "--mode", "gather", "--rmax", "15", "--max-shift", "0.5"]  # fmt: skip
    runs = {}
    for tag, extra in (("ref", None), ("gate", []), ("align", ["--align", "2.0"])):
        argv = ["--ground", g, "--drive", ref, str(tmp_path / "calib"), "--no-snap", "--mode", "gather", "--rmax", "15"]
        PL.main(["--out", str(tmp_path / tag)] + (argv if extra is None else common + extra))
        runs[tag] = (np.load(tmp_path / tag / "ground_map.npz")["cls"], json.load(open(tmp_path / tag / "stats.json")))
    fu = runs["gate"][1]["bad/fusion"]
    assert not fu["fused"] and fu["shift_m"] > 0.5 and "align" not in fu
    fu = runs["align"][1]["bad/fusion"]
    assert fu["fused"] and fu["aligned"] and fu["align"]["used"] == "aligned", fu
    al = fu["align"]
    assert al["residual_m"] <= 0.15 and abs(al["rot_deg"] + 0.25) < 0.1, fu
    G = PL.rigid(al["dx_m"], al["dy_m"], al["rot_deg"], al["pivot"])
    pts = np.array([[5.0, -5.0], [55.0, 5.0], [41.0, -3.0]])
    back = (pts @ E[:2, :2].T + E[:2, 2] + ORIGIN[:2]) @ G[:2, :2].T + G[:2, 3] - ORIGIN[:2]
    assert np.abs(back - pts).max() < 0.1, back - pts  # the correction undoes the pose error
    # the aligned drive adds the other lane, and its paint sits on the reference drive's
    (cr, _), (ca, _) = runs["ref"], runs["align"]
    assert (ca > 0).sum() > 1.2 * (cr > 0).sum()
    mk = CLASSES.index("marking")
    both = (cr > 0) & (ca > 0)
    a, b = (cr == mk) & both, (ca == mk) & both
    assert (a & b).sum() / (a | b).sum() > 0.6


def test_config_default_off():
    area = types.SimpleNamespace(cfg=load_config(None, registry={"tools": {}}))  # without the adopted variants
    args = projection_args(area)
    assert "--align" not in args and args[args.index("--max-shift") + 1] == 0.5  # unchanged commands
    area.cfg["fusion"]["align"] = True
    args = projection_args(area)
    assert args[args.index("--align") + 1] == 2.0
