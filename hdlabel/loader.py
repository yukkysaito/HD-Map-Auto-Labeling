"""Reference PyTorch dataset for an online multi-camera, multi-frame HD-map model (docs/DATASET.md; no model here).

Each item = one sample (qa.ok only) of a scene of output/<name>/dataset/:
  images      float [T, C, 3, H, W]  T frames (current + history, export's history spacing), C cameras, RGB 0..1, resized
  cam_mask    bool  [C]              camera present in this drive (missing ones are zeros)
  K           float [C, 3, 3]        intrinsics for the resized images
  T_base_cam  float [C, 4, 4]        camera -> base_link (official calibration)
  T_cur_frame float [T, C, 4, 4]     base_link at each frame's pose time -> base_link at the sample time (ego motion;
                                     T_cur_frame @ T_base_cam lifts frame features into the current ego frame)
  frame_dt    float [T, C]           frame pose time - sample time (s, <= 0 for history)
  bev_cls     long  [nx, ny]         raster labels (common.CLASSES, 0 = unobserved -> ignore in the loss), cropped from the
                                     scene's map-frame raster for `bev` (x, y range in m, res; default: dataset.json)
  bev_height  float [nx, ny]         road surface z in base_link (NaN = none)
  vec_pts     float [V, P, 3]        vector instances as P equally spaced points (base_link); vec_cls long [V] (index into
                                     VECTOR_CLASSES), vec_mask bool [V] (padding)
  vec_dir     float [V, 3]           arrows: allowed directions, multi-hot over ARROW_DIRECTIONS; arrow points run tail -> tip
The full-detail elements (lane line painted segments, arrow outlines, polygon rings) are in item_vectors(i).
split: "train" / "val" (by location, dataset.json); scenes: restrict to these ids. Items are in time order per scene
(val scenes are exported at the model cycle, so iterating a scene replays the drive cycle by cycle: eval/online_eval.py).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from .bev import SceneMap, resample

# MapTR-style vector classes: lane_line -> divider, curb + observed road_boundary -> boundary, crosswalk -> ped_crossing,
# arrow (lane direction marking, its directions in vec_dir) -> arrow, stop_line -> stop_line
VECTOR_CLASSES = ["divider", "boundary", "ped_crossing", "arrow", "stop_line"]
ARROW_DIRECTIONS = ["straight", "left", "right"]


def vec_class(e):
    k = e["kind"]
    if k == "lane_line":
        return 0
    if k == "curb" or (k == "road_boundary" and e.get("observed")):
        return 1
    if k == "crosswalk":
        return 2
    if k == "arrow":
        return 3
    if k == "stop_line":
        return 4
    return None


class HDMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_json: str,
        cams: list[int] | None = None,
        split: str | None = "train",
        history: int = 3,
        image_size=(704, 448),
        bev: dict | None = None,
        max_vectors: int = 300,
        npts: int = 20,
        scenes: list[str] | None = None,
    ):
        idx = json.load(open(dataset_json))
        root = os.path.dirname(os.path.abspath(dataset_json))
        cams = cams or idx["cameras"]  # default: every camera of the dataset
        self.cams, self.T, self.size, self.max_v, self.npts = cams, history, image_size, max_vectors, npts
        self.bev = dict(bev or idx["bev"])
        self.items, self.scenes, self._maps = [], {}, {}
        for s in idx["scenes"]:
            if (split and s.get("split") != split) or (scenes and s["scene"] not in scenes):
                continue
            d = os.path.join(root, s["path"])
            scene = json.load(open(f"{d}/scene.json"))
            frames = [json.loads(l) for l in open(f"{d}/frames.jsonl")]
            assert scene["history"]["frames"] + 1 >= history, (
                f"{s['scene']}: exported with {scene['history']['frames']} history frames"
            )
            self.scenes[s["scene"]] = dict(dir=d, scene=scene, frames=frames)
            self.items += [(s["scene"], r) for r in map(json.loads, open(f"{d}/samples.jsonl")) if r["qa"]["ok"]]

    def __len__(self):
        return len(self.items)

    def scene_map(self, sid) -> SceneMap:
        if sid not in self._maps:  # per worker process
            d = self.scenes[sid]["dir"]
            self._maps[sid] = SceneMap(f"{d}/map.npz", f"{d}/vectors.geojson")
        return self._maps[sid]

    def item_vectors(self, i) -> list[dict]:
        """all vector elements of item i in base_link, clipped to the BEV range, full detail (bev.SceneMap.vectors)."""
        sid, r = self.items[i]
        return self.scene_map(sid).vectors(np.array(r["T_map_base"]), self.bev["x"], self.bev["y"])

    def __getitem__(self, i):
        sid, r = self.items[i]
        sc = self.scenes[sid]
        W, H = self.size
        C = len(self.cams)
        T_cur = np.array(r["T_map_base"])
        T_cur_inv = np.linalg.inv(T_cur)
        t0 = r["stamp"]
        images = np.zeros((self.T, C, 3, H, W), np.float32)
        Tcf = np.tile(np.eye(4), (self.T, C, 1, 1))
        fdt = np.zeros((self.T, C))
        K = np.tile(np.eye(3), (C, 1, 1))
        Tbc = np.tile(np.eye(4), (C, 1, 1))
        mask = np.zeros(C, bool)
        for ci, c in enumerate(self.cams):
            meta = sc["scene"]["cameras"].get(str(c))
            ids = r["frames"].get(str(c))
            if meta is None or not ids:
                continue
            mask[ci] = True
            w0, h0 = meta["size"]
            K[ci] = np.diag([W / w0, H / h0, 1.0]) @ np.array(meta["K"])
            Tbc[ci] = np.array(meta["T_base_cam"])
            for k in range(self.T):
                f = sc["frames"][ids[k]]
                img = cv2.imread(os.path.join(sc["dir"], f["image"]))[..., ::-1]
                images[k, ci] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.0
                Tcf[k, ci] = T_cur_inv @ np.array(f["T_map_base"])
                fdt[k, ci] = (f["pose_stamp"] - t0) * 1e-9
        sm = self.scene_map(sid)
        cls, height = sm.crop(T_cur, self.bev["x"], self.bev["y"], self.bev["res"])
        pts = np.zeros((self.max_v, self.npts, 3), np.float32)
        vcls = np.zeros(self.max_v, np.int64)
        vmask = np.zeros(self.max_v, bool)
        vdir = np.zeros((self.max_v, len(ARROW_DIRECTIONS)), np.float32)
        n = 0
        for e in sm.vectors(T_cur, self.bev["x"], self.bev["y"]):
            k = vec_class(e)
            if k is None:
                continue
            if n >= self.max_v:
                raise ValueError(f"{sid} {r['stamp']}: more than max_vectors={self.max_v} vectors")
            pts[n] = resample(e["points"] if "points" in e else e["rings"][0], self.npts)
            vcls[n], vmask[n] = k, True
            for d in e.get("directions", []):
                vdir[n, ARROW_DIRECTIONS.index(d)] = 1.0
            n += 1
        f32 = lambda x: torch.from_numpy(np.asarray(x, np.float32))
        return dict(
            images=torch.from_numpy(images),
            cam_mask=torch.from_numpy(mask),
            K=f32(K),
            T_base_cam=f32(Tbc),
            T_cur_frame=f32(Tcf),
            frame_dt=f32(fdt),
            bev_cls=torch.from_numpy(cls.astype(np.int64)),
            bev_height=f32(height),
            vec_pts=torch.from_numpy(pts),
            vec_cls=torch.from_numpy(vcls),
            vec_mask=torch.from_numpy(vmask),
            vec_dir=torch.from_numpy(vdir),
            scene=sid,
            stamp=r["stamp"],
        )
