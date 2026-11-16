"""Evidence images for reviewing the extraction of a scene (by an AI agent or a person), written by review.py into
debug/scenes/<scene>/check/:

  items.json      the elements to decide (review_html.check_queue: low confidence + random spot check, minus the ones a
                  review already decided) and the map tiles to search for missed paint / curbs
  item_<n>.jpg    one element: ortho crop without | with its outline, two camera views without | with it
  tile_<n>.jpg    a 30 m map square around a sample position: ortho without | with all outlines
  sheet_<k>.jpg   several items stacked, to read many in one image

Colours as in review.html: road marking yellow, curb blue. The decision format is in docs/AGENT_REVIEW.md.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

COLOR = dict(marking=(0, 214, 255), curb=(255, 123, 61))  # BGR
PANEL = 256
ITEMS_PER_SHEET = 5
TILES = 4  # map squares per scene for the missed-paint search
TILE_M = 30.0


def _caption(img: np.ndarray, text: str, h: int = 24) -> np.ndarray:
    bar = np.full((h, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _rings(f: dict) -> list[np.ndarray]:
    g = f["geometry"]
    rings = g["coordinates"] if g["type"] == "Polygon" else [g["coordinates"]]
    return [np.asarray(r, float) for r in rings] + [np.asarray(o, float) for o in f["properties"].get("outlines", [])]


class Scene:
    """the exported scene (dataset/<scene>) + its ortho image, for cutting evidence out of both. Camera views come from
    the frames of the scene and of camera_dirs (other exported scenes: the labels of a scene also cover places that
    only other drives / neighbouring scenes saw)."""

    def __init__(self, dataset_dir: str, ortho_path: str, camera_dirs: list[str] = ()):
        self.sc = json.load(open(f"{dataset_dir}/scene.json"))
        m = np.load(f"{dataset_dir}/map.npz")
        self.lo, self.res, self.origin = np.asarray(m["lo"], float), float(m["res"]), np.asarray(m["origin"], float)
        self.ny = m["cls"].shape[1]
        self.ortho = cv2.imread(ortho_path)
        if self.ortho is None:
            raise FileNotFoundError(ortho_path)
        self.samples = [json.loads(line) for line in open(f"{dataset_dir}/samples.jsonl")]
        # every frame: (image path, camera key, K, size) and T_cam_map (map -> camera) for a vectorised projection
        self.frames, T = [], []
        for d in dict.fromkeys([dataset_dir, *camera_dirs]):
            if not os.path.exists(f"{d}/frames.jsonl"):
                continue
            cams = json.load(open(f"{d}/scene.json"))["cameras"]
            for line in open(f"{d}/frames.jsonl"):
                f = json.loads(line)
                cm = cams[str(f["cam"])]
                T_map_cam = np.asarray(f["T_map_base"], float) @ np.asarray(cm["T_base_cam"], float)
                T.append(np.linalg.inv(T_map_cam))
                self.frames.append(dict(image=f"{d}/{f['image']}", cam=str(f["cam"]), K=np.asarray(cm["K"]),
                                        size=cm["size"]))  # fmt: skip
        self.T_cam_map = np.array(T) if T else np.zeros((0, 4, 4))
        self.pos = -np.einsum("nji,nj->ni", self.T_cam_map[:, :3, :3], self.T_cam_map[:, :3, 3])  # camera centres

    def px(self, P: np.ndarray) -> np.ndarray:
        """map xy -> ortho pixel (col, row)."""
        q = (P[:, :2] - self.origin[:2] - self.lo) / self.res
        return np.c_[q[:, 0], self.ny - q[:, 1]]

    def ortho_crop(self, center, half_m: float, draw: list = (), size: int = PANEL) -> np.ndarray:
        """square crop of the ortho image around center (map xy), with polylines drawn: [(rings, BGR, closed)]."""
        c = self.px(np.asarray([center], float))[0]
        h = half_m / self.res
        x0, y0 = int(round(c[0] - h)), int(round(c[1] - h))
        n = int(round(2 * h))
        H, W = self.ortho.shape[:2]
        crop = np.zeros((n, n, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + n, W), min(y0 + n, H)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = self.ortho[sy0:sy1, sx0:sx1]
        k = size / n
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        for rings, col, closed in draw:
            for r in rings:
                q = np.round((self.px(r) - [x0, y0]) * k).astype(np.int32)
                cv2.polylines(crop, [q], closed, col, 2 if len(draw) == 1 else 1, cv2.LINE_AA)
        return crop

    def camera_views(self, P: np.ndarray, n: int = 2) -> list[int]:
        """up to n frames that see the points P (map xyz) well: 2.5-40 m ahead, inside the image, closest to 8 m and to
        the image centre; the second from another camera or 3 m away (another moment), so one occluded or blurred view
        does not hide the item."""
        if not len(self.frames):
            return []
        c = np.asarray(P, float).mean(0)
        near = np.flatnonzero(np.linalg.norm(self.pos - c, axis=1) < 45.0)
        pc = np.einsum("nij,j->ni", self.T_cam_map[near], np.r_[c, 1.0])[:, :3]
        cand = []
        for k, i in enumerate(near):
            fr = self.frames[i]
            K, (w, h), z = fr["K"], fr["size"], pc[k, 2]
            if not 2.5 < z < 40.0:
                continue
            u, v = (K @ pc[k])[:2] / z
            if 0.05 * w < u < 0.95 * w and 0.05 * h < v < 0.98 * h:
                cand.append((abs(z - 8.0) / 8.0 + abs(u / w - 0.5) + abs(v / h - 0.65), int(i)))
        out = []
        for _, i in sorted(cand):
            if all(self.frames[i]["cam"] != self.frames[j]["cam"] or np.linalg.norm(self.pos[i] - self.pos[j]) > 3.0
                   for j in out):  # fmt: skip
                out.append(i)
            if len(out) == n:
                break
        return out

    def camera_crop(self, i: int, P: np.ndarray, col, closed: bool, size: int = PANEL):
        """frame i cropped around the visible part of the points P (map xyz): (bare, with outline, text) or None."""
        fr = self.frames[i]
        K, (w, h) = fr["K"], fr["size"]
        q = (self.T_cam_map[i] @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        q = q[q[:, 2] > 0.5]
        uv = (K @ q.T).T
        uv = uv[:, :2] / uv[:, 2:]
        ins = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]  # crop around the visible part
        if not len(ins):
            return None
        (u0, v0), (u1, v1) = ins.min(0), ins.max(0)
        half = max(100.0, 0.75 * max(u1 - u0, v1 - v0))
        cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
        img = cv2.imread(fr["image"])
        if img is None:
            return None
        x0, y0, n = int(cu - half), int(cv_ - half), int(2 * half)
        pad = cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_CONSTANT)
        crop = pad[y0 + n : y0 + 2 * n, x0 + n : x0 + 2 * n]
        k = size / n
        bare = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        drawn = bare.copy()
        cv2.polylines(drawn, [np.round((uv - [x0, y0]) * k).astype(np.int32)], closed, col, 2, cv2.LINE_AA)
        dist = float(np.linalg.norm(self.pos[i] - P.mean(0)))
        return bare, drawn, f"cam{fr['cam']} {dist:.0f} m"


def item_image(sc: Scene, f: dict, group: str, n: int, reason: str) -> np.ndarray:
    """one row: ortho bare | ortho + outline | camera bare | camera + outline, with a caption."""
    col, closed = COLOR[group], f["geometry"]["type"] == "Polygon"
    rings = _rings(f)
    pts = np.vstack([r[:, :3] if r.shape[1] >= 3 else np.c_[r, np.zeros(len(r))] for r in rings])
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    ext = float((hi - lo).max())
    if ext > 26.0:  # a long curb / line: show 30 m around its middle
        mid = np.asarray(f["properties"].get("_at") or (lo + hi) / 2, float)
        center, half = mid, 15.0
        near = np.linalg.norm(pts[:, :2] - mid, axis=1) < 15.0
        cam_pts = pts[near] if near.sum() >= 2 else pts
    else:
        center, half, cam_pts = (lo + hi) / 2, max(4.0, ext / 2 + 2.0), pts
    a = _label(sc.ortho_crop(center, half), f"map {2 * half:.0f} m")
    b = _label(sc.ortho_crop(center, half, [(rings, col, closed)]), "map + outline")
    panels = [a, b]
    cam_pts = cam_pts if cam_pts.shape[1] == 3 else np.c_[cam_pts, np.zeros(len(cam_pts))]
    for i in sc.camera_views(cam_pts) + [None, None]:
        cam = sc.camera_crop(i, cam_pts, col, closed and ext <= 26.0) if i is not None else None
        if cam:
            panels += [_label(cam[0], cam[2]), _label(cam[1], "camera + outline")]
        elif len(panels) < 6:
            panels += [_label(np.zeros((PANEL, PANEL, 3), np.uint8), "no camera view")] * 2
        if len(panels) >= 6:
            break
    pr = f["properties"]
    conf = pr.get("confidence")
    text = (
        f"#{n}  {'road marking' if group == 'marking' else group}  {reason}"
        + (f"  conf {conf:.2f}" if conf is not None else "")
        + (f"  {pr['length']:.1f} m" if pr.get("length") else "")
    )
    return _caption(np.hstack(panels[:6]), text)


def tile_image(sc: Scene, features: list, groups: dict, center, n: int, size: int = 2 * PANEL) -> np.ndarray:
    draw = [(_rings(f), COLOR[groups[f["properties"]["kind"]]], f["geometry"]["type"] == "Polygon") for f in features]
    a = _label(sc.ortho_crop(center, TILE_M / 2, size=size), f"map {TILE_M:.0f} m")
    b = _label(sc.ortho_crop(center, TILE_M / 2, draw, size=size), "map + all outlines")
    return _caption(
        np.hstack([a, b]), f"tile {n}: is there paint without a yellow outline / a curb without a blue line?"
    )


def map_tiles(sc: Scene, seed: str, k: int = TILES) -> list[list[float]]:
    """up to k sample positions at least TILE_M apart (tiles do not overlap; a slow or stopped drive would otherwise put
    several tiles on one place), drawn reproducibly per scene."""
    xy = [np.asarray(s["T_map_base"], float)[:2, 3] for s in sc.samples]
    order = list(range(len(xy)))
    random.Random(seed).shuffle(order)
    pick = []
    for i in order:
        if all(np.abs(xy[i] - xy[j]).max() >= TILE_M for j in pick):
            pick.append(i)
        if len(pick) == k:
            break
    return [[round(float(v), 2) for v in xy[i]] for i in sorted(pick)]


def size_m(f: dict) -> float:
    """length of a line element, half the perimeter of an area: weights an element by how much map it covers."""
    g = f["geometry"]
    P = np.asarray(g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"], float)[:, :2]
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
    return L / 2 if g["type"] == "Polygon" else L


def write_pack(out: str, sc: Scene, features: list, items: list[dict], tiles: list, groups: dict) -> dict:
    """items: [{feature (geojson), group, at, reason}] -> out/check/; returns the items.json content."""
    d = f"{out}/check"
    os.makedirs(d, exist_ok=True)
    for old in os.listdir(d):
        if old.endswith((".jpg", ".json")):
            os.remove(f"{d}/{old}")
    rows, imgs = [], []
    for n, it in enumerate(items):
        f = dict(it["feature"], properties=dict(it["feature"]["properties"], _at=it["at"]))
        img = item_image(sc, f, it["group"], n, it["reason"])
        cv2.imwrite(f"{d}/item_{n:03d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        imgs.append(img)
        pr = it["feature"]["properties"]
        rows.append(dict(n=n, kind=pr["kind"], group=it["group"], at=it["at"], reason=it["reason"],
                         confidence=pr.get("confidence"), size_m=round(size_m(it["feature"]), 2), image=f"item_{n:03d}.jpg",
                         sheet=f"sheet_{n // ITEMS_PER_SHEET:02d}.jpg"))  # fmt: skip
    for k in range(0, len(imgs), ITEMS_PER_SHEET):
        cv2.imwrite(f"{d}/sheet_{k // ITEMS_PER_SHEET:02d}.jpg", np.vstack(imgs[k : k + ITEMS_PER_SHEET]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])  # fmt: skip
    trows = []
    for n, c in enumerate(tiles):
        cv2.imwrite(f"{d}/tile_{n}.jpg", tile_image(sc, features, groups, c, n), [cv2.IMWRITE_JPEG_QUALITY, 88])
        trows.append(dict(n=n, center=c, image=f"tile_{n}.jpg"))
    pack = dict(scene=os.path.basename(os.path.normpath(out)), items=rows, tiles=trows)
    json.dump(pack, open(f"{d}/items.json", "w"), indent=1)
    return pack
