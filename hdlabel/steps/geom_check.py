"""Geometric check of a scene's labels against the LiDAR map (independent of the cameras), plus debug PCDs.

  curb position   curbs are height steps in the LiDAR map: every 1 m along each curb vector, the raw map points across
                  it (+-2 m, 0.25 m bins, 20th percentile height) must change by > 7 cm over 0.5 m within 0.3 m of the
                  vector; reports the share of profiles that do and the median offset. Systematic label shifts
                  (calibration, time offset, localization) show up here without using any camera.
  height          vector z vs. the LiDAR ground points (map_surface.py debug PCD) within 0.3 m: median / p95 |dz|
Writes OUT/geom_check.json ("summary" = one line), OUT/geom_check.jpg (label map dimmed, curb profiles blue = LiDAR step
found, red = not), OUT/labels.pcd (label map cells, x y z rgb label: markings / crosswalks / curbs at 0.1 m, other classes at 0.3 m)
and OUT/vectors.pcd (vector elements sampled every 5 cm, x y z rgb kind). View with any PCD viewer next to
OUT/map_ground.pcd (LiDAR ground points used for the surface).
"""
import argparse, json, os, sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.bev import _dens
from hdlabel.common import CLASSES, PALETTE, VEC_COLOR
from hdlabel.mapproj import arrow_cls, load_arrows
from hdlabel.surface import map_files, read_pcd, write_pcd

p = argparse.ArgumentParser()
p.add_argument("--labels", required=True); p.add_argument("--vectors", required=True)
p.add_argument("--ground-pcd", help="map_surface.py debug PCD (LiDAR ground points)"); p.add_argument("--out", required=True)
p.add_argument("--map", required=True, help="point cloud map (tiles dir or .pcd)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
m = np.load(f"{a.labels}/ground_map.npz"); cls = arrow_cls(m["cls"], load_arrows(a.labels), m["lo"], float(m["res"]), m["origin"])
H, lo, res, sres, origin = m["height"], m["lo"], float(m["res"]), float(m["surf_res"]), m["origin"]
feats = json.load(open(a.vectors))["features"]
res_ = {}

# ---- curb position vs. LiDAR height steps (raw map points: the surface is smoothed, a 15 cm curb vanishes in it)
Q = np.concatenate([read_pcd(f_) for f_ in map_files(a.map, np.array([origin[:2] + lo + np.array(H.shape) * sres / 2]),
                                                       float(np.linalg.norm(np.array(H.shape) * sres) / 2))]).astype(np.float64)
bb0, bb1 = origin[:2] + lo, origin[:2] + lo + np.array(H.shape) * sres; Q = Q[np.all((Q[:, :2] > bb0) & (Q[:, :2] < bb1), 1)]
qtree = cKDTree(Q[:, :2]); offs, curbs = [], []
for f in feats:
    if f["properties"]["kind"] != "curb": continue
    P, _ = _dens(f["geometry"]["coordinates"], 0.2)
    for i in range(5, len(P) - 5, 5):                                  # every 1 m: profile across the curb
        t = P[i + 5, :2] - P[i - 5, :2]; t /= np.linalg.norm(t) + 1e-9; nrm = np.array([-t[1], t[0]])
        idx = qtree.query_ball_point(P[i, :2], 2.2); d = Q[idx, :2] - P[i, :2]; u, v = d @ t, d @ nrm; z = Q[idx, 2]
        k = (np.abs(u) < 0.75) & (np.abs(v) < 2.0) & (np.abs(z - P[i, 2]) < 0.6); u, v, z = u[k], v[k], z[k]
        bins = np.arange(-2.0, 2.01, 0.25); zb = np.full(len(bins) - 1, np.nan)
        for b in range(len(bins) - 1):
            kk = (v >= bins[b]) & (v < bins[b + 1])
            if kk.sum() >= 3: zb[b] = np.percentile(z[kk], 20)
        jump = np.abs(zb[2:] - zb[:-2]); pos = bins[1:-2] + 0.25                 # height change over 0.5 m, at the bin edge
        ok = np.isfinite(jump) & (jump > 0.07)
        if np.isfinite(zb).sum() < 8: continue                                    # too few map points to judge
        offs.append(float(np.min(np.abs(pos[ok]))) if ok.any() else np.inf); curbs.append((P[i], offs[-1]))
offs = np.array(offs)
res_["curb"] = dict(profiles=int(len(offs)), step_within_0_3m=round(float((offs <= 0.3).mean()), 3) if len(offs) else None,
                    median_offset_m=round(float(np.median(np.minimum(offs, 2.0))), 3) if len(offs) else None)

# ---- vector heights vs. LiDAR ground points
if a.ground_pcd and os.path.exists(a.ground_pcd):
    G = read_pcd(a.ground_pcd).astype(np.float64); tree = cKDTree(G[:, :2]); dz = []
    for f in feats:
        if f["properties"]["kind"] == "drivable_area": continue
        g = f["geometry"]; c = g["coordinates"] if g["type"] == "LineString" else g["coordinates"][0]
        P, _ = _dens(c, 0.5); d, i = tree.query(P[:, :2], distance_upper_bound=0.3); k = np.isfinite(d)
        dz += (P[k, 2] - G[i[k], 2]).tolist()
    dz = np.abs(np.array(dz))
    res_["height"] = dict(points=int(len(dz)), median_cm=round(float(np.median(dz)) * 100, 1) if len(dz) else None,
                          p95_cm=round(float(np.percentile(dz, 95)) * 100, 1) if len(dz) else None)
res_["summary"] = (f"curbs with a LiDAR height step within 0.3 m: {res_['curb']['step_within_0_3m']} of {res_['curb']['profiles']} profiles"
                   + (f"; vector z vs LiDAR ground: median {res_['height']['median_cm']} cm, p95 {res_['height']['p95_cm']} cm" if "height" in res_ else ""))
json.dump(res_, open(f"{a.out}/geom_check.json", "w"), indent=1)

# ---- image: label classes dimmed, curb profiles blue = LiDAR step within 0.3 m, red = not
img = np.ascontiguousarray(np.rot90(PALETTE[cls][..., ::-1] // 3)); Wc = cls.shape[1]
for P_, o in curbs:
    i, j = ((P_[:2] - origin[:2] - lo) / res).astype(int)
    if 0 <= i < cls.shape[0] and 0 <= j < Wc: cv2.circle(img, (int(i), int(Wc - 1 - j)), 4, (255, 80, 0) if o <= 0.3 else (0, 0, 255), -1)
cv2.imwrite(f"{a.out}/geom_check.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---- PCDs: label cells and vectors in 3D (map frame)
fine = np.isin(cls, [CLASSES.index(c) for c in ("marking", "crosswalk", "curb", "manhole")])
coarse = (cls > 0) & ~fine
sub = np.zeros_like(coarse); sub[::3, ::3] = coarse[::3, ::3]
ii, jj = np.nonzero(fine | sub); xy = np.c_[ii, jj] * res + lo + res / 2
z = H[np.minimum((ii * res / sres).astype(int), H.shape[0] - 1), np.minimum((jj * res / sres).astype(int), H.shape[1] - 1)]
k = np.isfinite(z); c_ = cls[ii, jj][k]
write_pcd(f"{a.out}/labels.pcd", np.c_[xy[k] + origin[:2], z[k] + origin[2]], dict(rgb=PALETTE[c_], label=c_))
KIND = ["lane_line", "curb", "road_boundary", "crosswalk", "marking_area", "arrow", "drivable_area"]
pts, col, kind = [], [], []
for f in feats:
    pr, g = f["properties"], f["geometry"]
    if pr["kind"] == "drivable_area": continue
    for c in ([g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]) + [o for o in pr.get("outlines", [])]:
        P, _ = _dens(c, 0.05); pts.append(P); b, g_, r = VEC_COLOR.get(pr["kind"], (0, 0, 255)); col.append(np.tile([r, g_, b], (len(P), 1)))
        kind.append(np.full(len(P), KIND.index(pr["kind"])))
if pts: write_pcd(f"{a.out}/vectors.pcd", np.concatenate(pts), dict(rgb=np.concatenate(col), kind=np.concatenate(kind)))
print(res_["summary"])
