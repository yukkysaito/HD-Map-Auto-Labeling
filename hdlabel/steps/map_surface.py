"""3D road surface of the ego road level from the LiDAR point cloud map (hdlabel/surface.py: layers + connectivity, so
bridges / elevated roads / decks are separated from the road the trajectories are on). Only the point cloud map is
used (the lanelet2 / vector map next to it is never read). No flat-ground assumption.

The grid covers --bbox; --gt trajectories (one or more drives, already restricted to the scene's level by the caller)
seed the level and define the corridor. QA: base_link height above the surface along each trajectory (~0 when
localization z and the map agree).
Writes OUT/map_ground.npz (height, height_w, lo, surf_res, origin), OUT/map_ground.json (QA + layer statistics);
with --debug DIR: DIR/surface_height.png and DIR/map_ground.pcd (ground points used, x y z intensity, map frame).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.surface import Grid, build, map_files, read_pcd, write_pcd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True, help="pointcloud_map.pcd directory (tiles) or a single .pcd file")
    p.add_argument("--gt", required=True, nargs="+", help="gt.npy of one or more drives")
    p.add_argument(
        "--t-range", type=float, nargs=2, help="use only this time range of the FIRST trajectory (ns) as its seed"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--debug")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="grid extent (map frame)",
    )
    p.add_argument("--surf-res", type=float, default=0.2)
    p.add_argument("--corridor", type=float, default=40.0)
    p.add_argument(
        "--step", type=float, default=0.25, help="max height step between neighbouring 0.5 m cells of one level (m)"
    )
    a = p.parse_args(argv)

    bb = np.array(a.bbox)
    inside = lambda g: np.all((g[:, :2] > bb[:2] - a.corridor) & (g[:, :2] < bb[2:] + a.corridor), 1)
    trajs, names = [], []
    for k, f in enumerate(a.gt):
        g = np.load(f)
        if k == 0 and a.t_range:
            g = g[(g[:, 0] >= a.t_range[0]) & (g[:, 0] <= a.t_range[1])]
        g = g[inside(g[:, 1:3])][:, 1:4]
        if len(g) > 1:
            trajs.append(g)
            names.append(f)
    if not trajs:
        raise SystemExit("no trajectory inside the box")
    origin = np.r_[np.round((bb[:2] + bb[2:]) / 2 / 100) * 100, np.round(np.median(np.concatenate(trajs)[:, 2]))]
    files = map_files(a.map, np.concatenate(trajs)[:, :2], a.corridor)
    P = np.concatenate([read_pcd(f, ("x", "y", "z", "intensity")) for f in files]).astype(np.float64)
    P[:, :3] -= origin
    lo = np.floor((bb[:2] - origin[:2]) / a.surf_res) * a.surf_res
    grid = Grid(lo, np.ceil((bb[2:] - origin[:2] - lo) / a.surf_res).astype(int), a.surf_res, origin)
    S = build(P, [t - origin for t in trajs], grid, a.corridor, a.step)
    H = S["height"]
    print(f"{len(files)} tiles, {S['info']}, height {np.nanmin(H) + origin[2]:.2f} .. {np.nanmax(H) + origin[2]:.2f} m")

    qa = {}
    for f, t in zip(names, trajs):
        d_ = (t[:, 2] - origin[2]) - grid.at(H, t - origin, cval=np.nan)
        d_ = d_[np.isfinite(d_)]
        if len(d_):
            qa[f] = dict(
                median_cm=round(float(np.median(d_)) * 100, 1),
                p05_cm=round(float(np.percentile(d_, 5)) * 100, 1),
                p95_cm=round(float(np.percentile(d_, 95)) * 100, 1),
                n=int(len(d_)),
            )
        print(f"base_link above the surface ({f}): {qa.get(f)}")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(
        f"{a.out}/map_ground.npz",
        height=H.astype(np.float32),
        height_w=S["height_w"].astype(np.float32),
        lo=lo,
        surf_res=a.surf_res,
        origin=origin,
    )
    json.dump(
        dict(
            tiles=len(files),
            origin=origin.tolist(),
            lo=lo.tolist(),
            surf_res=a.surf_res,
            shape=list(grid.shape),
            surface=S["info"],
            base_link_above_ground=qa,
        ),
        open(f"{a.out}/map_ground.json", "w"),
        indent=1,
    )
    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
        hn = np.rot90(np.nan_to_num((H - np.nanmin(H)) / max(np.nanmax(H) - np.nanmin(H), 1e-6), nan=0))
        cv2.imwrite(
            f"{a.debug}/surface_height.png",
            cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * np.isfinite(np.rot90(H))[..., None],
        )
        G = P[S["ground"]]
        v = np.floor(G[:, :3] / 0.1).astype(np.int64)  # 0.1 m voxels (mean), keeps the PCD small
        _, inv = np.unique(v, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = np.bincount(inv)
        M = np.stack([np.bincount(inv, G[:, k]) / n for k in range(4)], 1)
        write_pcd(f"{a.debug}/map_ground.pcd", M[:, :3] + origin, dict(intensity=M[:, 3].astype(np.float32)))


if __name__ == "__main__":
    main()
