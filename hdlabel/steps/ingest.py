"""Ingest one drive: one pass over its rosbags (mcap) -> undistorted camera images, poses, calibration, keyframes.

Topics, cameras and image settings come from --sensors (the `topics`, `base_frame`, `cameras`, `image_scale`,
`jpeg_quality` keys of configs/default.yaml): the pose (nav_msgs/Odometry), the wheel speed (optional,
VelocityReport: longitudinal_velocity, lateral_velocity, heading_rate), /tf_static and per camera camera_info + the
compressed image. Images are scaled and undistorted to a pinhole model on the fly (no raw copy is kept): cameras with
`hfov` get a pinhole with that horizontal FoV (wide lenses), the others the valid-pixel crop (alpha 0). Configured
cameras that are not in the bag are skipped.
Keyframes (symlinks, used for segmentation and labelling): every --kf-dist m or --kf-yaw deg of wheel odometry.

Writes DATA/:
  und_cam<N>/<stamp_ns>.jpg, und_cam<N>/K.json   undistorted images (stamp = image header stamp), pinhole K and size
  kf_cam<N>/                                     keyframe symlinks into und_cam<N>
  gt.npy        [stamp, x, y, z, qx, qy, qz, qw]  base_link in the map frame (localization)
  wheel.npy     [stamp, v_long, v_lat, heading_rate]
  camera_info.json   K, D, model, size, frame per camera (original resolution)
  tf_static.json     {frame: 4x4 T_base_link_frame} (official calibration; may be missing in filtered bags)
  ingest.json        cameras found, frame counts, bags + their time ranges (bag_ranges, for tracing samples back to
                     their bag), errors (unreadable bags / images: skipped, the rest is used) and issues found in the
                     data: pose_gaps ([t0, t1] ns: localization missing > --max-gap s or jumping faster than
                     --max-speed m/s; samples there fail QA), camera_gaps (frames missing > 3 periods, per camera),
                     missing_cameras, no tf_static
"""

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from mcap.reader import NonSeekingReader, make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import bag_ranges, load_yaml, stamps_of, tf_to_base


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sensors", required=True, help="sensor settings: JSON text or a .json / .yaml file (see above)")
    p.add_argument("--kf-dist", type=float, default=0.5)
    p.add_argument("--kf-yaw", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("bags", nargs="+")
    p.add_argument("--max-gap", type=float, default=0.5)
    p.add_argument("--max-speed", type=float, default=50.0)
    a = p.parse_args(argv)
    cfg = json.loads(a.sensors) if a.sensors.lstrip().startswith("{") else load_yaml(a.sensors)
    cams = {int(c): (v or {}) for c, v in cfg["cameras"].items()}
    scale, quality = float(cfg.get("image_scale", 0.5)), int(cfg.get("jpeg_quality", 95))
    TOPICS = cfg["topics"]
    BASE = cfg.get("base_frame", "base_link")
    # per camera: its own topics (cameras.<id>.image / camera_info) or the {id} templates of `topics`
    IMAGE = {cams[c].get("image") or TOPICS["image"].format(id=c): c for c in cams}
    INFO = {cams[c].get("camera_info") or TOPICS["camera_info"].format(id=c): c for c in cams}
    stamp = lambda h: h.stamp.sec * 10**9 + h.stamp.nanosec

    def write(path: str, data):
        """write a file (bytes, text, a JSON-able object or an array -> .npy) unless it already has this content: ingesting
        the same bags again (new settings, after prune) keeps the times of unchanged outputs, so nothing downstream is redone."""
        if isinstance(data, np.ndarray):
            b = io.BytesIO()
            np.save(b, data)
            data = b.getvalue()
        elif not isinstance(data, (bytes, str)):
            data = json.dumps(data, indent=1)
        if isinstance(data, str):
            data = data.encode()
        if not os.path.exists(path) or open(path, "rb").read() != data:
            open(path, "wb").write(data)

    class Undistorter:
        def __init__(self, cam, info):
            K = np.array(info.k).reshape(3, 3)
            K[:2] *= scale
            D = np.array(info.d)
            w, h = round(info.width * scale), round(info.height * scale)
            self.size0 = (round(info.width * scale), round(info.height * scale))
            hfov = cams[cam].get("hfov")
            if hfov:
                f = w / 2 / np.tan(np.radians(hfov) / 2)
                Kn = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
                x = y = 0
                rw, rh = w, h
            else:
                Kn, (x, y, rw, rh) = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
            self.m1, self.m2 = cv2.initUndistortRectifyMap(K, D, None, Kn, (w, h), cv2.CV_16SC2)
            self.cam = cam
            self.crop = (x, y, rw, rh)
            Kc = Kn.copy()
            Kc[0, 2] -= x
            Kc[1, 2] -= y
            self.dir = f"{a.out}/und_cam{cam}"
            os.makedirs(self.dir, exist_ok=True)
            write(f"{self.dir}/K.json", dict(K=Kc.tolist(), size=[int(rw), int(rh)], hfov=hfov, scale=scale))

        def __call__(self, data: bytes, st: int):
            path = f"{self.dir}/{st}.jpg"
            if os.path.exists(path):
                return
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                errors.append(f"cam{self.cam}: undecodable image {st}")
                return
            im = cv2.resize(im, self.size0, interpolation=cv2.INTER_AREA)
            x, y, rw, rh = self.crop
            cv2.imwrite(
                path,
                cv2.remap(im, self.m1, self.m2, cv2.INTER_LINEAR)[y : y + rh, x : x + rw],
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )

    # cheap check first (bag summaries only): logs recorded without cameras are useless for labelling
    have, errors, unindexed = set(), [], set()
    for b in a.bags:
        try:
            with open(b, "rb") as f:
                sm = make_reader(f).get_summary()
            if sm is None:
                raise ValueError("no summary section")
            have |= {c.topic for c in sm.channels.values()}
        except Exception as e:  # truncated / unfinished bag: read it sequentially
            unindexed.add(b)
            errors.append(f"{os.path.basename(b)}: no summary ({type(e).__name__}: {e}); read sequentially")
    if not unindexed and not set(IMAGE) & have:
        os.makedirs(a.out, exist_ok=True)
        json.dump(
            dict(bags=sorted(a.bags), cameras=[], missing_cameras=sorted(cams), note="no camera topics in the bags"),
            open(f"{a.out}/ingest.json", "w"),
            indent=1,
        )
        print(f"no camera topics in {len(a.bags)} bags -> nothing to label")
        raise SystemExit

    gt, wheel, info, und, pending, tf = [], [], {}, {}, {c: [] for c in cams}, {}
    topics = (
        [t for t in (TOPICS["pose"], TOPICS.get("velocity"), TOPICS.get("tf_static")) if t] + list(INFO) + list(IMAGE)
    )
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = []

        def submit(c, x):
            """queue one image; bounded queue (memory), worker errors are raised here."""
            jobs.append(pool.submit(und[c], *x))
            while len(jobs) > 1000:
                jobs.pop(0).result()

        def messages(b):
            """messages of one bag; a broken bag (truncated, corrupt chunk) is used up to the error, which is recorded."""
            try:
                with open(b, "rb") as f:
                    if b in unindexed:  # no index: stream from the start (up to the break)
                        for sc, ch, m, d in NonSeekingReader(
                            f, decoder_factories=[DecoderFactory()]
                        ).iter_decoded_messages(topics=topics, log_time_order=False):
                            yield sc, ch, m, d
                    else:
                        yield from make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                            topics=topics
                        )
            except Exception as e:
                errors.append(f"{os.path.basename(b)}: read error after {len(gt)} poses ({type(e).__name__}: {e})")
                print("ERROR", errors[-1], flush=True)

        for b in sorted(a.bags):
            for _, ch, _, d in messages(b):
                t = ch.topic
                if t == TOPICS["pose"]:
                    p_, q = d.pose.pose.position, d.pose.pose.orientation
                    gt.append([stamp(d.header), p_.x, p_.y, p_.z, q.x, q.y, q.z, q.w])
                elif t == TOPICS.get("velocity"):
                    wheel.append([stamp(d.header), d.longitudinal_velocity, d.lateral_velocity, d.heading_rate])
                elif t == TOPICS.get("tf_static"):  # latched, possibly several messages: merge all
                    for x in d.transforms:
                        T = np.eye(4)
                        q, tr = x.transform.rotation, x.transform.translation
                        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                        T[:3, 3] = [tr.x, tr.y, tr.z]
                        tf[x.child_frame_id.lstrip("/")] = (x.header.frame_id.lstrip("/"), T)
                elif t in INFO:
                    c = INFO[t]
                    if c not in info:
                        info[c] = d
                        und[c] = Undistorter(c, d)
                        [submit(c, x) for x in pending[c]]
                        pending[c] = []
                else:
                    c = IMAGE[t]
                    x = (bytes(d.data), stamp(d.header))
                    if c in und:
                        submit(c, x)
                    else:
                        pending[c].append(x)
            print("read", os.path.basename(b), flush=True)
        for j in jobs:
            j.result()
    if len(gt) < 10:
        raise SystemExit(f"no localization in the bags ({len(gt)} poses); errors: {errors}")
    for name, rows in (("gt", gt), ("wheel", wheel)):
        arr = np.array(rows, dtype=np.float64).reshape(-1, 8 if name == "gt" else 4)
        write(f"{a.out}/{name}.npy", arr[np.argsort(arr[:, 0])])
    write(
        f"{a.out}/camera_info.json",
        {
            str(c): dict(
                K=list(d.k), D=list(d.d), model=d.distortion_model, w=d.width, h=d.height, frame=d.header.frame_id
            )
            for c, d in info.items()
        },
    )

    tfb = tf_to_base(tf, BASE)
    if tfb:
        write(f"{a.out}/tf_static.json", tfb)

    # keyframes from wheel odometry (distance / heading change since the last keyframe; nothing while stopped)
    v = np.load(f"{a.out}/wheel.npy")
    if len(v) < 10:  # no wheel odometry in the bags: speed / yaw rate from the poses
        g = np.load(f"{a.out}/gt.npy")
        dt_ = np.maximum(np.diff(g[:, 0]) * 1e-9, 1e-3)
        yaw = np.unwrap(2 * np.arctan2(g[:, 6], g[:, 7]))
        v = np.c_[
            g[1:, 0], np.linalg.norm(np.diff(g[:, 1:3], axis=0), axis=1) / dt_, np.zeros(len(dt_)), np.diff(yaw) / dt_
        ]
    dt = np.diff(v[:, 0]) / 1e9
    counts = {}
    for c in sorted(info):
        files, st = stamps_of(f"{a.out}/und_cam{c}")
        cum = np.interp(st, v[:, 0], np.r_[0, np.cumsum(np.abs(v[1:, 1]) * dt)])
        yaw = np.degrees(np.interp(st, v[:, 0], np.r_[0, np.cumsum(v[1:, 3] * dt)]))
        keep = [0]
        for i in range(1, len(files)):
            if cum[i] - cum[keep[-1]] >= a.kf_dist or abs(yaw[i] - yaw[keep[-1]]) >= a.kf_yaw:
                keep.append(i)
        if keep[-1] != len(files) - 1:
            keep.append(len(files) - 1)
        kd = f"{a.out}/kf_cam{c}"
        os.makedirs(kd, exist_ok=True)
        for f in os.listdir(kd):
            os.remove(f"{kd}/{f}")
        for i in keep:
            os.symlink(f"../und_cam{c}/{os.path.basename(files[i])}", f"{kd}/{os.path.basename(files[i])}")
        counts[c] = dict(frames=len(files), keyframes=len(keep))
    missing = sorted(set(cams) - set(info))
    # data checks: localization gaps / jumps (samples there fail QA in export.py), camera frame gaps
    g = np.load(f"{a.out}/gt.npy")
    dt = np.diff(g[:, 0]) * 1e-9
    v = np.linalg.norm(np.diff(g[:, 1:4], axis=0), axis=1) / np.maximum(dt, 1e-3)
    bad = np.flatnonzero((dt > a.max_gap) | (v > a.max_speed))
    cam_gaps = {}
    for c in sorted(info):
        st = stamps_of(f"{a.out}/und_cam{c}")[1]
        if len(st) > 2:
            d_ = np.diff(st) * 1e-9
            per = np.median(d_)
            n = int((d_ > 3 * per).sum())
            if n:
                cam_gaps[f"cam{c}"] = dict(gaps=n, longest_s=round(float(d_.max()), 2))
    issues = dict(
        pose_gaps=[[int(g[i, 0]), int(g[i + 1, 0])] for i in bad],
        camera_gaps=cam_gaps,
        missing_cameras=missing,
        no_tf_static=not tfb,
        errors=len(errors),
    )
    summary = json.dumps(
        dict(
            bags=sorted(a.bags),
            bag_ranges=bag_ranges(sorted(a.bags)),
            cameras=sorted(info),
            missing_cameras=missing,
            frames={str(k): v for k, v in counts.items()},
            tf_static=bool(tfb),
            poses=len(gt),
            duration_s=round((gt[-1][0] - gt[0][0]) * 1e-9, 1),
            issues=issues,
            errors=errors[:50],
        ),
        indent=1,
    )
    write(f"{a.out}/ingest.json", summary)
    if os.path.exists(f"{a.out}/pruned.json"):
        os.remove(f"{a.out}/pruned.json")
    print(f"cameras {sorted(info)} (missing {missing}), tf_static {'yes' if tfb else 'NO'}, {counts}")


if __name__ == "__main__":
    main()
