"""Reference PyTorch dataset for an online multi-camera, multi-frame HD-map model (docs/DATASET.md; no model here).

Each item = one sample (qa.ok only) of a scene of output/<name>/dataset/:
  images      float [T, C, 3, H, W]  T frames (current + history, export's history spacing), C cameras, RGB 0..1, resized
                                     (uint8 0..255 with uint8=True)
  cam_mask    bool  [C]              camera present in this drive (missing ones are zeros)
  K           float [C, 3, 3]        intrinsics for the resized images
  T_base_cam  float [C, 4, 4]        camera -> base_link (official calibration)
  T_cur_frame float [T, C, 4, 4]     base_link at each frame's pose time -> base_link at the sample time (ego motion;
                                     T_cur_frame @ T_base_cam lifts frame features into the current ego frame)
  frame_dt    float [T, C]           frame pose time - sample time (s, <= 0 for history)
  bev_cls     long  [nx, ny]         raster labels (common.CLASSES, 0 = unobserved -> ignore in the loss), cropped from the
                                     scene's map-frame raster for `bev` (x, y range in m, res; default: dataset.json)
  bev_height  float [nx, ny]         road surface z in base_link (NaN = none)
  bev_level   uint8 [nx, ny]         other road levels at the cell (<scene>/levels.npz, optional): bit 1 = a level above
                                     (overpass, elevated road, enclosure, tunnel ceiling), bit 2 = a level below; 0 without
  bev_level_dz half [2, nx, ny]      dz (m) of that level above (> 0, channel 0) / below (< 0, channel 1) the ego surface
                                     (NaN = none, all NaN without levels.npz)
  multilevel  float []               share of the crop's cells with another level (NaN without levels.npz); a sample is
                                     multilevel when it is >= MULTILEVEL
  vec_pts     float [V, P, 3]        vector instances as P equally spaced points (base_link); vec_cls long [V] (index into
                                     VECTOR_CLASSES), vec_mask bool [V] (padding)
  vec_dir     float [V, 3]           arrows: allowed directions, multi-hot over ARROW_DIRECTIONS; arrow points run tail -> tip
The full-detail elements (lane line painted segments, arrow outlines, polygon rings) are in item_vectors(i).
split: "train" / "val" / "test" (by location, dataset.json); scenes: restrict to these ids. Items are in time order per scene
(val scenes are exported at the model cycle, so iterating a scene replays the drive cycle by cycle: eval/online_eval.py).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from .bev import SceneMap, grid, resample
from .levels import Levels

# MapTR-style vector classes: lane_line -> divider, curb + observed road_boundary -> boundary, crosswalk -> ped_crossing,
# arrow (lane direction marking, its directions in vec_dir) -> arrow, stop_line -> stop_line
VECTOR_CLASSES = ["divider", "boundary", "ped_crossing", "arrow", "stop_line"]
ARROW_DIRECTIONS = ["straight", "left", "right"]
MULTILEVEL = 0.05  # share of the BEV crop over / under another level from which a sample counts as multilevel


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


def thin_still(rs, every_s, move_m=0.1):
    """samples (time order) without the near-duplicates of a standing vehicle: a sample is kept when it is move_m
    from the last kept one or every_s later."""
    out, last = [], None
    for r in rs:
        p, t = np.array(r["T_map_base"])[:2, 3], r["stamp"] * 1e-9
        if last is None or np.linalg.norm(p - last[0]) >= move_m or t - last[1] >= every_s:
            out.append(r)
            last = p, t
    return out


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
        fast_decode: bool = False,
        uint8: bool = False,
        still_every_s: float = 0.0,
    ):
        """fast_decode: JPEG decoding at 1/2, 1/4 or 1/8 scale (libjpeg DCT scaling) when the target size allows it -
        several times faster for long histories (T x C images per item), slightly different resampling.
        uint8: images as uint8 0..255 (4x less memory between loader workers; the model divides by 255).
        still_every_s: while the vehicle stands (< 0.1 m from the last kept sample), keep a sample only every this many
        seconds (0: keep all; waits at lights otherwise repeat one view many times)."""
        idx = json.load(open(dataset_json))
        self.fast_decode, self.uint8 = fast_decode, uint8
        root = os.path.dirname(os.path.abspath(dataset_json))
        cams = cams or idx["cameras"]  # default: every camera of the dataset
        self.cams, self.T, self.size, self.max_v, self.npts = cams, history, image_size, max_vectors, npts
        self.bev = dict(bev or idx["bev"])
        self.items, self.scenes, self._maps, self._levels = [], {}, {}, {}
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
            rs = [r for r in map(json.loads, open(f"{d}/samples.jsonl")) if r["qa"]["ok"]]
            self.items += [(s["scene"], r) for r in (thin_still(rs, still_every_s) if still_every_s else rs)]

    def __len__(self):
        return len(self.items)

    def scene_map(self, sid) -> SceneMap:
        if sid not in self._maps:  # per worker process
            d = self.scenes[sid]["dir"]
            self._maps[sid] = SceneMap(f"{d}/map.npz", f"{d}/vectors.geojson")
        return self._maps[sid]

    def level_crop(self, sid, T: np.ndarray) -> dict:
        """bev_level / bev_level_dz / multilevel (module doc) of the BEV crop of pose T in scene sid."""
        if sid not in self._levels:
            f = f"{self.scenes[sid]['dir']}/levels.npz"
            self._levels[sid] = Levels(f) if os.path.exists(f) else None
        lv, b = self._levels[sid], self.bev
        if lv is None:
            n = grid(b["x"], b["y"], b["res"])[0].shape
            dz = np.full((2, *n), np.nan, np.float16)
        else:
            dz = lv.crop_dz(self.scene_map(sid), T, b["x"], b["y"], b["res"])
        m = Levels.bits(dz)
        share = float((m > 0).mean()) if lv is not None else float("nan")
        return dict(bev_level=torch.from_numpy(m), bev_level_dz=torch.from_numpy(dz), multilevel=torch.tensor(share))

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
        images = np.zeros((self.T, C, 3, H, W), np.uint8 if self.uint8 else np.float32)
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
            flag = cv2.IMREAD_COLOR
            if self.fast_decode:
                for s_, fl in (
                    (8, cv2.IMREAD_REDUCED_COLOR_8),
                    (4, cv2.IMREAD_REDUCED_COLOR_4),
                    (2, cv2.IMREAD_REDUCED_COLOR_2),
                ):
                    if w0 // s_ >= W and h0 // s_ >= H:
                        flag = fl
                        break
            done = {}  # a frame repeated in the history (vehicle stopped) is read once
            for k in range(self.T):
                if ids[k] in done:
                    images[k, ci] = images[done[ids[k]], ci]
                else:
                    img = cv2.imread(os.path.join(sc["dir"], sc["frames"][ids[k]]["image"]), flag)[..., ::-1]
                    im = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                    images[k, ci] = im if self.uint8 else im / 255.0
                    done[ids[k]] = k
                f = sc["frames"][ids[k]]
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
            **self.level_crop(sid, T_cur),
            scene=sid,
            stamp=r["stamp"],
        )
