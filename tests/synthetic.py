"""Synthetic test / demo data: a rosbag (mcap, ROS 2) and a point cloud map of a straight road, so the whole pipeline
can run without real logs (CI, quick start). Nothing here is real data.

World (map frame, flat road surface at z = Z0): a 3-lane road along +x with dashed lane lines, solid edge lines, curbs,
sidewalks and grass; a straight arrow, a stop line and a crosswalk. One front camera (pinhole, no distortion) renders
the road surface by ray casting; the vehicle weaves slightly (yaw changes) along the road. Colours are the palette of
the `color_stub` segmentation model (steps/segment.py), so the labels come out exactly.

usage: python tests/synthetic.py OUT_DIR   -> OUT_DIR/logs/<drive>/synthetic.mcap, OUT_DIR/pointcloud_map.pcd, OUT_DIR/config.yaml
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdlabel.arrowshape import TEMPLATES
from hdlabel.steps.segment import STUB_PALETTE
from hdlabel.surface import write_pcd

X0, Y0, Z0 = 5000.0, -3000.0, 10.0  # map origin of the synthetic road (any large offset, like real map frames)
TEX = 0.03  # texture resolution (m)
XR, YR = (-30.0, 230.0), (-16.0, 16.0)  # textured area (road frame)
W, H, F = 640, 400, 380.0  # image size, focal length (px)
CAM_T = np.array([1.5, 0.0, 1.6])  # camera in base_link (m)
CAM_PITCH = np.radians(12.0)  # looking down
STOP_X, ARROW_X, CW_X = 150.0, 132.0, 156.0

C = {k: np.array(v, np.uint8) for k, v in STUB_PALETTE.items()}  # name -> RGB


def texture() -> np.ndarray:
    """road surface colours, [nx, ny, 3] RGB, cell (i, j) at x = XR0 + i * TEX, y = YR0 + j * TEX (road frame)."""
    nx, ny = int((XR[1] - XR[0]) / TEX), int((YR[1] - YR[0]) / TEX)
    x = XR[0] + (np.arange(nx) + 0.5) * TEX
    y = YR[0] + (np.arange(ny) + 0.5) * TEX
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = np.empty((nx, ny, 3), np.uint8)
    T[:] = C["grass"]
    T[np.abs(Y) < 9.0] = C["sidewalk"]
    T[np.abs(Y) < 6.2] = C["curb"]
    T[np.abs(Y) < 6.0] = C["road"]
    edge = np.abs(np.abs(Y) - 5.6) < 0.075  # solid edge lines
    dash = (np.abs(np.abs(Y) - 2.0) < 0.075) & (np.mod(X, 10.0) < 5.0) & (X < STOP_X)  # dashed lane lines
    T[edge | dash] = C["marking"]
    T[(np.abs(X - STOP_X) < 0.3) & (Y > -6.0) & (Y < 0.0)] = C["marking"]  # stop line (0.6 m) across the right half
    cw = (X > CW_X) & (X < CW_X + 4.0) & (np.abs(Y) < 6.0) & (np.mod(Y, 1.0) < 0.45)  # crosswalk stripes
    T[cw] = C["crosswalk"]
    # straight arrow in the middle right lane (template: rows along the arrow, tip at the last row; 2 cm)
    tpl = TEMPLATES["S"] > 0.5
    rows, cols = tpl.shape
    s = 0.02 / TEX
    tpl = (
        cv2.resize(
            tpl.astype(np.uint8), (max(1, round(cols * s)), max(1, round(rows * s))), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    i0 = int((ARROW_X - XR[0]) / TEX)
    j0 = int((-4.0 - YR[0]) / TEX) - tpl.shape[1] // 2  # lane centre y = -4 m
    region = T[i0 : i0 + tpl.shape[0], j0 : j0 + tpl.shape[1]]
    region[tpl] = C["marking"]  # template: driver's left at the last column = +j = +y
    return T


def trajectory(dt: float = 0.1, speed: float = 6.0) -> np.ndarray:
    """[N, 5] t (s), x, y, yaw (road frame) of base_link: along +x, weaving +-0.8 m (yaw up to ~5 deg)."""
    t = np.arange(0.0, (195.0 - 5.0) / speed, dt)
    x = 5.0 + speed * t
    y = -4.0 + 0.8 * np.sin(2 * np.pi * x / 40.0)
    yaw = np.arctan(0.8 * 2 * np.pi / 40.0 * np.cos(2 * np.pi * x / 40.0))
    return np.c_[t, x, y, yaw]


R_BASE_CAM = R.from_euler("y", CAM_PITCH).as_matrix() @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # optical


def render(T: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """camera image (RGB) of the flat textured road from base_link pose (x, y, yaw), z = road surface."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1) @ R_BASE_CAM.T  # base_link
    Rz = R.from_euler("z", yaw).as_matrix()
    d = rays @ Rz.T
    c = Rz @ CAM_T + np.array([x, y, 0.0])
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C["sky"]
    down = d[..., 2] < -1e-3
    t = -c[2] / np.where(down, d[..., 2], -1.0)
    px, py = c[0] + t * d[..., 0], c[1] + t * d[..., 1]
    i = ((px - XR[0]) / TEX).astype(int)
    j = ((py - YR[0]) / TEX).astype(int)
    ok = down & (t < 80) & (i >= 0) & (i < T.shape[0]) & (j >= 0) & (j < T.shape[1])
    img[ok] = T[i[ok], j[ok]]
    return img


def point_map(path: str) -> None:
    """flat ground points every 0.3 m (road level), curbs raised 0.15 m beyond |y| > 6 m (map frame)."""
    x, y = np.meshgrid(np.arange(XR[0], XR[1], 0.3), np.arange(YR[0] - 10, YR[1] + 10, 0.3), indexing="ij")
    z = np.where(np.abs(y) > 6.0, 0.15, 0.0)
    P = np.stack([x.ravel() + X0, y.ravel() + Y0, z.ravel() + Z0], 1)
    write_pcd(path, P, dict(intensity=np.zeros(len(P), np.float32)))


MSGDEFS = {
    "builtin_interfaces/Time": "int32 sec\nuint32 nanosec",
    "std_msgs/Header": "builtin_interfaces/Time stamp\nstring frame_id",
    "geometry_msgs/Point": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/Vector3": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/Quaternion": "float64 x\nfloat64 y\nfloat64 z\nfloat64 w",
    "geometry_msgs/Pose": "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation",
    "geometry_msgs/PoseWithCovariance": "geometry_msgs/Pose pose\nfloat64[36] covariance",
    "geometry_msgs/Twist": "geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular",
    "geometry_msgs/TwistWithCovariance": "geometry_msgs/Twist twist\nfloat64[36] covariance",
    "geometry_msgs/Transform": "geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation",
    "geometry_msgs/TransformStamped": "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform",
    "sensor_msgs/RegionOfInterest": "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify",
}
TOP = {
    "nav_msgs/msg/Odometry": "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
    "geometry_msgs/TwistWithCovariance twist",
    "sensor_msgs/msg/CompressedImage": "std_msgs/Header header\nstring format\nuint8[] data",
    "sensor_msgs/msg/CameraInfo": "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
    "float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\nuint32 binning_y\n"
    "sensor_msgs/RegionOfInterest roi",
    "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms",
}


def msgdef(name: str) -> str:
    """ROS 2 message definition with all its dependencies (mcap ros2msg schema)."""
    text, seen, todo = TOP[name], set(), [TOP[name]]
    while todo:
        for line in todo.pop().splitlines():
            t = line.split()[0].split("[")[0]
            if "/" in t and t not in seen:
                seen.add(t)
                todo.append(MSGDEFS[t])
                text += "\n" + "=" * 80 + f"\nMSG: {t}\n" + MSGDEFS[t]
    return text


def stamp(ns: int) -> dict:
    return dict(sec=int(ns // 10**9), nanosec=int(ns % 10**9))


def quat(Rm: np.ndarray) -> dict:
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return dict(x=x, y=y, z=z, w=w)


def write_bag(path: str, t0_ns: int = 1_800_000_000 * 10**9) -> None:
    from mcap_ros2.writer import Writer

    T = texture()
    traj = trajectory()
    K = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]
    with open(path, "wb") as f:
        w = Writer(f)
        sch = {n: w.register_msgdef(n, msgdef(n)) for n in TOP}
        h = lambda ns, frame: dict(stamp=stamp(ns), frame_id=frame)
        tf = dict(
            transforms=[
                dict(
                    header=h(t0_ns, "base_link"),
                    child_frame_id="camera0/camera_optical_link",
                    transform=dict(translation=dict(zip("xyz", CAM_T)), rotation=quat(R_BASE_CAM)),
                )
            ]
        )
        w.write_message("/tf_static", sch["tf2_msgs/msg/TFMessage"], tf, log_time=t0_ns)
        info = dict(
            header=h(t0_ns, "camera0/camera_optical_link"), height=H, width=W, distortion_model="plumb_bob",
            d=[0.0] * 5, k=K, r=[1.0, 0, 0, 0, 1, 0, 0, 0, 1], p=[F, 0, W / 2, 0, 0, F, H / 2, 0, 0, 0, 1, 0],
            binning_x=0, binning_y=0, roi=dict(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
        )  # fmt: skip
        for t, x, y, yaw in traj:
            ns = t0_ns + int(round(t * 1e9))
            Rm = R.from_euler("z", yaw).as_matrix()
            odo = dict(
                header=h(ns, "map"), child_frame_id="base_link",
                pose=dict(pose=dict(position=dict(x=x + X0, y=y + Y0, z=Z0), orientation=quat(Rm)), covariance=[0.0] * 36),
                twist=dict(twist=dict(linear=dict(x=6.0, y=0.0, z=0.0), angular=dict(x=0.0, y=0.0, z=0.0)), covariance=[0.0] * 36),
            )  # fmt: skip
            w.write_message("/localization/kinematic_state", sch["nav_msgs/msg/Odometry"], odo, log_time=ns)
            ok, jpg = cv2.imencode(".jpg", render(T, x, y, yaw)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            img = dict(header=h(ns, "camera0/camera_optical_link"), format="jpeg", data=jpg.tobytes())
            w.write_message(
                "/sensing/camera/camera0/camera_info",
                sch["sensor_msgs/msg/CameraInfo"],
                info | dict(header=img["header"]),
                log_time=ns,
            )
            w.write_message(
                "/sensing/camera/camera0/image_raw/compressed", sch["sensor_msgs/msg/CompressedImage"], img, log_time=ns
            )
        w.finish()


CONFIG = """# synthetic demo / CI: tests/synthetic.py
name: synthetic
output: {out}
pointcloud_map: {out}/pointcloud_map.pcd
logs: ["{out}/logs/*"]
review_dir: {out}/review
cameras: {{0: {{}}}}
reference_camera: 0
image_scale: 1.0
segmentation: {{model: color_stub}}
calib: {{scenes: 1, time_offset: 0.0}}
scenes: {{length_m: 100, context_m: 40, corridor_m: 20}}
dataset: {{every_s: 0.5, val_every_s: null, history_s: 1.0, history_dt_s: 0.5, bev: {{x: [-10, 40], y: [-15, 15], res: 0.2}},
          split: {{tile_m: 1000, val_every: 1000}}}}
jobs: {{cpu: 2, gpu: 1}}
"""


def make(out: str) -> str:
    """writes the bag, the map and a config; returns the config path."""
    os.makedirs(f"{out}/logs/2030-01-01-00-00-00", exist_ok=True)
    write_bag(f"{out}/logs/2030-01-01-00-00-00/synthetic.mcap")
    point_map(f"{out}/pointcloud_map.pcd")
    cfg = f"{out}/config.yaml"
    open(cfg, "w").write(CONFIG.format(out=os.path.abspath(out)))
    return cfg


if __name__ == "__main__":
    print(make(sys.argv[1]))
