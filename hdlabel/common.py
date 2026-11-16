"""Shared helpers: label classes, image stamps, poses, extrinsics, configs. No HD map is used anywhere."""

from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# segmentation class ids -> label classes. Label id = index + 1 (0 = unobserved), "other" last. Defaults: Mapillary Vistas
# (the default model); another model's ids come from the config `segmentation.class_map` (run.py passes it to the steps
# as $HDLABEL_CLASS_MAP, JSON {group: [ids], ..., "vertical": [ids]}).
GROUPS = {
    "road": [13, 14, 7, 10, 36, 43],  # Road, Service Lane, Bike Lane, Parking, Catch Basin, Pothole
    "marking": [24],  # Lane Marking - General
    "crosswalk": [23, 8],  # Lane Marking - Crosswalk, Crosswalk - Plain
    "manhole": [41],
    "curb": [2, 9],  # Curb, Curb Cut
    "sidewalk": [15, 11],  # Sidewalk, Pedestrian Area
    "offroad": [29, 30, 3, 4, 5, 6, 17, 16, 12, 26],
}
# ids of vertical structures (fence, guard rail, barrier, wall, bridge, building, vegetation): not on the ground surface,
# projected onto it they would smear, so they are not voted
VERTICAL = [3, 4, 5, 6, 16, 17, 30]
if os.environ.get("HDLABEL_CLASS_MAP"):
    _cm = json.loads(os.environ["HDLABEL_CLASS_MAP"])
    if set(_cm) - {"vertical"} != set(GROUPS):
        raise ValueError(f"segmentation.class_map needs exactly the groups {list(GROUPS)} (+ optional vertical)")
    GROUPS = {g: list(_cm[g]) for g in GROUPS}
    VERTICAL = list(_cm.get("vertical", []))
CLASSES = ["unobserved"] + list(GROUPS) + ["other"]  # raster label ids 0..8
PALETTE = np.array(
    [
        [0, 0, 0],
        [90, 90, 90],
        [255, 255, 255],
        [255, 220, 0],
        [0, 120, 255],
        [200, 60, 60],
        [80, 160, 80],
        [60, 110, 40],
        [120, 70, 150],
    ],
    np.uint8,
)  # RGB per raster label id


def link_file(src: str, dst: str) -> None:
    """relative symlink dst -> src; on file systems without symlinks (exFAT, FAT, some network shares) a hard link,
    else a copy. Readers must not rely on the link (use the directory of the original, see Camera.K)."""
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def stamps_of(img_dir: str) -> tuple[list[str], np.ndarray]:
    """Images are named <stamp_ns>.jpg; returns sorted paths and stamps (int64 ns)."""
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    return files, np.array([int(os.path.basename(f)[:-4]) for f in files], dtype=np.int64)


class PoseInterp:
    """base_link pose in the map frame from /localization/kinematic_state (gt.npy: stamp, x, y, z, qx, qy, qz, qw)."""

    def __init__(self, gt_path: str):
        g = np.load(gt_path)
        g = g[np.argsort(g[:, 0])]
        self.g = g[np.r_[True, np.diff(g[:, 0]) > 0]]
        self.slerp = Slerp(self.g[:, 0], R.from_quat(self.g[:, 4:8]))

    def __call__(self, stamps_ns) -> np.ndarray:
        t = np.clip(np.asarray(stamps_ns, dtype=np.float64), self.g[0, 0], self.g[-1, 0])
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3] = self.slerp(t).as_matrix()
        T[:, :3, 3] = np.stack([np.interp(t, self.g[:, 0], self.g[:, k]) for k in (1, 2, 3)], 1)
        return T


class Extrinsic:
    """camera -> base_link transform plus camera time offset (image stamp + offset = pose time)."""

    def __init__(self, R_base_cam, t_base_cam, time_offset_s: float = 0.0):
        self.R = np.asarray(R_base_cam, dtype=np.float64)
        self.t = np.asarray(t_base_cam, dtype=np.float64)
        self.offset = float(time_offset_s)

    @classmethod
    def load(cls, path: str) -> "Extrinsic":
        e = json.load(open(path))
        return cls(e["R_base_cam"], e["t_base_cam"], e["time_offset_s"])

    def save(self, path: str, **extra) -> None:
        json.dump(
            dict(R_base_cam=self.R.tolist(), t_base_cam=self.t.tolist(), time_offset_s=self.offset, **extra),
            open(path, "w"),
            indent=1,
        )

    @property
    def T(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


def tf_to_base(tf: dict, base: str = "base_link") -> dict:
    """{child: (parent, 4x4 T_parent_child)} -> {frame: 4x4 T_base_frame (list)} for every frame chained to base."""
    out = {}
    for f in tf:
        T, fr = np.eye(4), f
        while fr != base and fr in tf:
            fr, Tp = tf[fr]
            T = np.asarray(Tp) @ T
        if fr == base:
            out[f] = T.tolist()
    return out


def load_tf_static(path: str, base: str = "base_link", topic: str = "/tf_static") -> dict:
    """camera mounting for bags without /tf_static -> {frame: 4x4 T_base_frame}. Accepted:
    .json        {frame: 4x4 T_base_frame} (what ingest writes as tf_static.json)
    .yaml / .yml list of {frame, parent, translation: [x, y, z], rotation: [qx, qy, qz, qw]} (static transforms)
    .mcap        a bag with /tf_static (e.g. a full recording of the same vehicle)"""

    def T_of(t, q):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    if path.endswith(".json"):
        return json.load(open(path))
    tf = {}
    if path.endswith((".yaml", ".yml")):
        for e in load_yaml(path):
            tf[e["frame"].lstrip("/")] = (e["parent"].lstrip("/"), T_of(e["translation"], e["rotation"]))
    elif path.endswith(".mcap"):
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        with open(path, "rb") as f:
            for _, _, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=[topic]
            ):
                for x in m.transforms:
                    t, q = x.transform.translation, x.transform.rotation
                    tf[x.child_frame_id.lstrip("/")] = (
                        x.header.frame_id.lstrip("/"),
                        T_of([t.x, t.y, t.z], [q.x, q.y, q.z, q.w]),
                    )
    else:
        raise ValueError(f"{path}: tf_static must be .json, .yaml or .mcap")
    return tf_to_base(tf, base)


def device() -> str:
    """torch device of the GPU steps: $HDLABEL_DEVICE (set by run.py from the config `device`), else cuda if available."""
    import torch

    d = os.environ.get("HDLABEL_DEVICE", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d


def load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(open(path))


def bag_ranges(bags: list[str]) -> dict:
    """{bag file name: [first, last] message log time (ns)} from the mcap summaries (no message is read)."""
    from mcap.reader import make_reader

    out = {}
    for b in bags:
        try:
            with open(b, "rb") as f:
                st = make_reader(f).get_summary().statistics
            out[os.path.basename(b)] = [int(st.message_start_time), int(st.message_end_time)]
        except Exception as e:  # broken / unindexed file: no range
            out[os.path.basename(b)] = None
            print(f"warning: no summary in {b}: {e}")
    return out


def bag_of(ingest: dict, stamp_ns: int) -> str | None:
    """source bag file (name) of a stamp: the bag whose time range contains it (else the nearest)."""
    rng = {b: r for b, r in (ingest.get("bag_ranges") or {}).items() if r}
    if not rng:
        return os.path.basename(ingest["bags"][0]) if len(ingest.get("bags") or []) == 1 else None
    return min(
        rng,
        key=lambda b: (
            0 if rng[b][0] <= stamp_ns <= rng[b][1] else min(abs(stamp_ns - rng[b][0]), abs(stamp_ns - rng[b][1]))
        ),
    )


def stop_lines(
    lines: list[dict], min_width: float = 0.35, min_length: float = 3.0, max_cos: float = 0.35, radius: float = 15.0
) -> set[int]:
    """indices of the lane-line candidates that are stop lines: wide paint (stop lines ~0.45 m, lane lines ~0.15 m), long
    enough, and ACROSS the traffic direction, which is taken from the long lane lines (solid / dashed, >= 6 m) within
    `radius` m (median |cos| between the line and their segments). lines: dicts with xy [N, 2], length, width, style."""
    from scipy.spatial import cKDTree

    mids, dirs, owner = [], [], []
    for k, l in enumerate(lines):
        if l["length"] < 6 or l["style"] not in ("solid", "dashed"):
            continue
        xy = np.asarray(l["xy"], float)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        q = np.stack(
            [np.interp(np.arange(0, s[-1] + 1e-9, 1.0), s, xy[:, j]) for j in (0, 1)], 1
        )  # 1 m steps (any input density)
        for p0, p1 in zip(q[:-1], q[1:]):
            d = p1 - p0
            n = np.linalg.norm(d)
            if n > 0.3:
                mids.append((p0 + p1) / 2)
                dirs.append(d / n)
                owner.append(k)
    if len(mids) < 3:
        return set()
    tree, dirs, owner = cKDTree(mids), np.array(dirs), np.array(owner)
    out = set()
    for k, l in enumerate(lines):
        if l["length"] < min_length or (l["width"] or 0) < min_width:
            continue
        d = l["xy"][-1] - l["xy"][0]
        d = d / max(np.linalg.norm(d), 1e-9)
        idx = [i for i in tree.query_ball_point(l["xy"].mean(0), radius) if owner[i] != k]
        if len(idx) >= 3 and np.median(np.abs(dirs[idx] @ d)) < max_cos:
            out.add(k)
    return out


VEC_COLOR = dict(
    lane_line=(255, 255, 255),
    stop_line=(255, 255, 0),
    curb=(255, 80, 0),
    crosswalk=(255, 0, 255),
    marking_area=(0, 255, 0),
    drivable_area=(0, 160, 0),
    arrow=(0, 0, 255),
)  # BGR, for debug drawings


def draw_bev(cls: np.ndarray, els: list, x, y, res: float, scale: int = 2, dim: float = 0.5) -> np.ndarray:
    """debug image (BGR) of an ego-frame BEV: raster classes (dimmed) + vectors (bev.SceneMap.vectors): lane lines white
    (dashed yellow), curbs blue, crosswalks magenta, marking areas green, arrows red with directions (S / L / R),
    road boundaries red (observed) / grey (unknown)."""
    import cv2

    bev = cv2.resize(
        (PALETTE[cls] * dim).astype(np.uint8)[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    pix = lambda P: np.round(
        np.c_[(y[1] - np.asarray(P)[:, 1]) / res, (x[1] - np.asarray(P)[:, 0]) / res] * scale
    ).astype(np.int32)
    for e in els:
        col = VEC_COLOR.get(e["kind"], (0, 0, 255) if e.get("observed") else (128, 128, 128))
        if e["kind"] == "lane_line" and e.get("style") != "solid":
            col = (0, 255, 255)
        if "rings" in e:
            cv2.polylines(bev, [pix(r) for r in e["rings"]], True, col, 1, cv2.LINE_AA)
            continue
        if e["kind"] == "arrow":
            cv2.polylines(bev, [pix(o) for o in e.get("outlines", [])], True, col, 1, cv2.LINE_AA)
            p0, p1 = pix(e["points"])[[0, -1]]
            cv2.arrowedLine(bev, tuple(map(int, p0)), tuple(map(int, p1)), col, 1, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(
                bev,
                "+".join(d[0].upper() for d in e.get("directions", [])),
                (int(p1[0]) + 3, int(p1[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1,
                cv2.LINE_AA,
            )
            continue
        cv2.polylines(bev, [pix(e["points"])], False, col, 2 if scale > 1 else 1, cv2.LINE_AA)
    return bev
