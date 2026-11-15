"""Training data of one scene (docs/DATASET.md) for an online multi-camera, multi-frame HD-map model.

Labels are stored once per scene in the map frame; samples carry poses and frames (hdlabel/bev.py turns them into
ego-frame labels of any range / resolution, as hdlabel/loader.py does):
  map.npz          cls (uint8 0.1 m raster, common.CLASSES, 0 = unobserved; arrows redrawn from their measured
                   outlines), height (road surface z, 0.2 m, NaN = none), lo, res, surf_res, origin
  vectors.geojson  3D vectors in the map frame (vectorize.py; arrows with directions and paint outlines)
  images/cam<N>/<stamp>.jpg   the undistorted frames the samples use (hard links into the work dir, no copy)
  frames.jsonl     one row per frame: id, cam, image, stamp (image header), pose_stamp (= stamp + time offset),
                   T_map_base (base_link at pose_stamp, snapped onto the surface), bag (source rosbag file)
  samples.jsonl    one row per sample: stamp, T_map_base, frames {cam: [frame ids: current, then history every
                   history_dt s back]}, qa, source (drive, bag, stamp)
  scene.json       cameras (K, size, T_base_cam, time offset + status), classes, default BEV spec, time range, history
Sample times: frames of the reference camera within the scene's time range, at least --every s apart (pose time).
QA per sample: observed share of the near region (x 0..30 m, |y| < 8 m) >= --min-observed, max |dt| of the current
frames <= --max-dt ms, no localization gap / jump (ingest.json issues), not rejected in the review file, and no camera
of the drive with a failed time offset. Only samples with qa.ok should be trained on.
--debug DIR: DIR/preview_<k>.jpg (reference image | BEV raster with the vectors).
"""
import argparse, json, os, shutil, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.bev import SceneMap
from hdlabel.common import CLASSES, Extrinsic, PoseInterp, bag_of, draw_bev, stamps_of
from hdlabel.mapproj import arrow_cls, load_arrows, snap_to_ground

p = argparse.ArgumentParser()
p.add_argument("--labels", required=True, help="project_labels.py output dir (ground_map.npz, arrows.json)")
p.add_argument("--vectors", required=True); p.add_argument("--data", required=True); p.add_argument("--calib", required=True)
p.add_argument("--out", required=True); p.add_argument("--debug")
p.add_argument("--t-range", type=int, nargs=2, required=True, metavar=("T0", "T1"))
p.add_argument("--ref", type=int, required=True, help="camera whose frames define the sample times")
p.add_argument("--every", type=float, default=0.5); p.add_argument("--history", type=float, default=2.0)
p.add_argument("--history-dt", type=float, default=0.5, help="spacing of the history frames (s)")
p.add_argument("--x", type=float, nargs=2, default=(-20.0, 60.0)); p.add_argument("--y", type=float, nargs=2, default=(-30.0, 30.0))
p.add_argument("--res", type=float, default=0.2, help="default BEV spec written to scene.json (the loader may use any)")
p.add_argument("--min-observed", type=float, default=0.5); p.add_argument("--max-dt", type=float, default=60.0)
p.add_argument("--review"); p.add_argument("--previews", type=int, default=3)
a = p.parse_args()
shutil.rmtree(a.out, ignore_errors=True); os.makedirs(a.out)

# ---- labels in the map frame
m = np.load(f"{a.labels}/ground_map.npz")
cls = arrow_cls(m["cls"], load_arrows(a.labels), m["lo"], float(m["res"]), m["origin"])
np.savez_compressed(f"{a.out}/map.npz", cls=cls, height=m["height"].astype(np.float32), lo=m["lo"], res=m["res"],
                    surf_res=m["surf_res"], origin=m["origin"])
shutil.copy(a.vectors, f"{a.out}/vectors.geojson")
smap = SceneMap(f"{a.out}/map.npz", f"{a.out}/vectors.geojson")

# ---- poses, cameras, calibration
ing = json.load(open(f"{a.data}/ingest.json"))
poser = PoseInterp(f"{a.data}/gt.npy"); snap = snap_to_ground(poser, f"{a.labels}/ground_map.npz")
cams = sorted(int(os.path.basename(f)[3:-5]) for f in os.listdir(a.calib) if f.startswith("cam") and f.endswith(".json"))
cams = [c for c in cams if os.path.isdir(f"{a.data}/und_cam{c}")]
ext = {c: Extrinsic.load(f"{a.calib}/cam{c}.json") for c in cams}
off_status = {c: json.load(open(f"{a.calib}/cam{c}.json")).get("offset_status", "unknown") for c in cams}
calib_ok = "failed" not in off_status.values()
und = {c: stamps_of(f"{a.data}/und_cam{c}") for c in cams}
pose_t = {c: und[c][1] + np.int64(round(ext[c].offset * 1e9)) for c in cams}
review = json.load(open(a.review)) if a.review and os.path.exists(a.review) else {}
rejected = set(review.get("reject_samples", []))
bad_t = [tuple(g) for g in (ing.get("issues") or {}).get("pose_gaps", [])]                # localization gaps / jumps

t0, t1 = a.t_range; cand = pose_t[a.ref][(pose_t[a.ref] >= t0) & (pose_t[a.ref] <= t1)]
samples = [cand[0]] if len(cand) else []
for t in cand[1:]:
    if (t - samples[-1]) * 1e-9 >= a.every - 1e-3: samples.append(t)
K = int(round(a.history / a.history_dt))

# ---- frames used: current (nearest pose time) + history (latest at or before t - k * dt, +60 ms)
frame_id, frames = {}, []
def use(c, j):
    if (c, j) not in frame_id:
        frame_id[(c, j)] = len(frames); frames.append((c, j))
    return frame_id[(c, j)]
rows = []
for t in samples:
    fr, dts = {}, {}
    for c in cams:
        ts = pose_t[c]
        if not len(ts): continue
        j0 = int(np.argmin(np.abs(ts - t))); ids = [use(c, j0)]; dts[c] = round((ts[j0] - t) * 1e-6, 1)
        for k in range(1, K + 1):
            ids.append(use(c, max(0, int(np.searchsorted(ts, t - k * a.history_dt * 1e9 + 0.06e9)) - 1)))
        fr[str(c)] = ids
    rows.append((t, fr, dts))

with open(f"{a.out}/frames.jsonl", "w") as f:
    T_all = {c: poser(pose_t[c]) for c in cams}
    for i, (c, j) in enumerate(frames):
        src = und[c][0][j]; dst = f"images/cam{c}/{os.path.basename(src)}"; os.makedirs(f"{a.out}/images/cam{c}", exist_ok=True)
        try: os.link(os.path.realpath(src), f"{a.out}/{dst}")
        except OSError: shutil.copy(src, f"{a.out}/{dst}")                 # other file system
        f.write(json.dumps(dict(id=i, cam=c, image=dst, stamp=int(und[c][1][j]), pose_stamp=int(pose_t[c][j]),
                                T_map_base=np.round(T_all[c][j], 6).tolist(), bag=bag_of(ing, int(und[c][1][j])))) + "\n")

# ---- samples
n_ok = 0
prev_k = set(np.linspace(0, len(rows) - 1, a.previews).astype(int)) if a.debug and a.previews and rows else set()
with open(f"{a.out}/samples.jsonl", "w") as out:
    for k, (t, fr, dts) in enumerate(rows):
        T = poser([t])[0]
        c_near, _ = smap.crop(T, (0.0, 30.0), (-8.0, 8.0), 0.2); obs = float((c_near > 0).mean())
        dtm = max(abs(v) for v in dts.values()); gap = any(g0 - 1e9 <= t <= g1 + 1e9 for g0, g1 in bad_t)
        ok = bool(calib_ok and obs >= a.min_observed and dtm <= a.max_dt and not gap and not review.get("reject") and int(t) not in rejected)
        n_ok += ok
        out.write(json.dumps(dict(stamp=int(t), T_map_base=np.round(T, 6).tolist(), frames=fr, dt_ms={str(c): v for c, v in dts.items()},
                                  qa=dict(observed_near=round(obs, 3), max_dt_ms=dtm, pose_ok=not gap, calib_ok=calib_ok, ok=ok),
                                  source=dict(drive=os.path.basename(a.data.rstrip("/")), bag=bag_of(ing, int(t)), stamp=int(t)))) + "\n")
        if k in prev_k:                                                    # reference image | BEV with vectors
            c_, _ = smap.crop(T, a.x, a.y, a.res); bev = draw_bev(c_, smap.vectors(T, a.x, a.y), a.x, a.y, a.res, scale=2)
            rc, rj = frames[fr[str(a.ref)][0]]; img = cv2.imread(und[rc][0][rj])
            img = cv2.resize(img, (round(img.shape[1] * bev.shape[0] / img.shape[0]), bev.shape[0]))
            os.makedirs(a.debug, exist_ok=True); cv2.imwrite(f"{a.debug}/preview_{k:04d}.jpg", np.hstack([img, bev]), [cv2.IMWRITE_JPEG_QUALITY, 85])

cam_meta = {}
for c in cams:
    Kj = json.load(open(f"{a.data}/und_cam{c}/K.json"))
    cam_meta[str(c)] = dict(K=Kj["K"], size=Kj["size"], T_base_cam=ext[c].T.tolist(), time_offset_s=ext[c].offset, time_offset_status=off_status[c])
json.dump(dict(format=2, drive=os.path.basename(a.data.rstrip("/")), bags=[os.path.basename(b) for b in ing.get("bags", [])],
               t_range=[int(t0), int(t1)], every_s=a.every, history=dict(frames=K, dt_s=a.history_dt), cameras=cam_meta, ref_camera=a.ref,
               classes=CLASSES, bev=dict(x=a.x, y=a.y, res=a.res, frame="base_link (snapped onto the map surface)",
                                         layout="row 0 = far (x max), col 0 = left (y max)"),
               map=dict(res=float(m["res"]), surf_res=float(m["surf_res"]), origin=m["origin"].tolist(), shape=list(cls.shape)),
               ground_snap=snap, review=bool(review), samples=len(rows), qa_ok=n_ok, frames=len(frames)),
          open(f"{a.out}/scene.json", "w"), indent=1)
print(f"{len(rows)} samples ({n_ok} pass QA), {len(frames)} frames -> {a.out}")
