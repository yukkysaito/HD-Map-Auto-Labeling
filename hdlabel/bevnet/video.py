"""Videos of a trained BEVSegNet on the val scenes, one video frame per dataset sample in time order: the camera
images with the predicted classes painted on the ground, the label raster, the prediction and the prediction on the
labelled cells only (forward = up). Cells the model abstains on or no camera sees within VIS_DEPTH are black.

usage: python -m hdlabel.bevnet.video output/<name>/debug/eval/bevnet<tag>.pt [--scenes S ...] [--out DIR]
       -> DIR/<model>_<scene>.mp4 (default DIR: <model dir>/videos)
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch

from hdlabel.common import CLASSES, PALETTE

from .data import BEVDataset, collate, downsample
from .metrics import VIS_DEPTH, Scores, predict, upsample
from .train import load_checkpoint

TILE = (352, 224)  # camera tile size in the video
COLS = 5


def bev_image(cls, scale):
    """class raster (row 0 = far, column 0 = left) -> RGB image, forward up."""
    return cv2.resize(PALETTE[cls], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def overlay(img, K, T_base_cam, cls, bev, cell, alpha=0.45, max_dist=60.0):
    """paint the class raster onto a camera image: each pixel's ray meets the ground plane (z = 0 in base_link) and
    takes the class of the cell it hits (unknown or beyond max_dist: unchanged)."""
    h, w = img.shape[:2]
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = T_base_cam[:3, :3] @ np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    o = T_base_cam[:3, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o[2] / d[2]
    gx, gy = o[0] + s * d[0], o[1] + s * d[1]
    row = np.floor((bev["x"][1] - gx) / cell).astype(np.int64)
    col = np.floor((bev["y"][1] - gy) / cell).astype(np.int64)
    ok = (s > 0) & (np.hypot(gx - o[0], gy - o[1]) < max_dist)
    ok &= (row >= 0) & (row < cls.shape[0]) & (col >= 0) & (col < cls.shape[1])
    c = np.zeros(u.size, np.int64)
    c[ok] = cls[row[ok], col[ok]]
    c = c.reshape(h, w)
    m = c > 0
    out = img.copy()
    out[m] = ((1 - alpha) * img[m] + alpha * PALETTE[c[m]]).astype(np.uint8)
    return out


def text(img, s, org, size=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)


def compose(x, y, pr, cams, bev, cell, scale, title):
    imgs = x["images"][0, 0].numpy()  # [C, 3, H, W] uint8, current frame
    K, Tbc = x["K"][0].numpy(), x["T_base_cam"][0].numpy()
    rows = []
    for r in range(0, len(cams), COLS):
        row = []
        for c in range(r, r + COLS):
            t = np.zeros((TILE[1], TILE[0], 3), np.uint8)
            if c < len(cams):
                t = imgs[c].transpose(1, 2, 0)
                if x["cam_mask"][0, c]:
                    t = overlay(t, K[c], Tbc[c], pr, bev, cell)
                t = cv2.resize(t, TILE, interpolation=cv2.INTER_AREA)
                text(t, f"cam {cams[c]}", (8, 22))
            row.append(t)
        rows.append(np.hstack(row))
    top = np.vstack(rows)
    panels = []
    ex, ey = int(bev["y"][1] / cell * scale), int(bev["x"][1] / cell * scale)  # ego (base_link origin)
    for im, name in ((y, "label"), (pr, "prediction"), (np.where(y > 0, pr, 0), "prediction (labelled cells only)")):
        im = bev_image(im, scale)
        cv2.arrowedLine(im, (ex, ey + 12), (ex, ey - 12), (255, 255, 255), 2, tipLength=0.4)
        text(im, name, (8, 24), 0.7)
        panels.append(np.pad(im, ((0, 0), (0, 8), (0, 0)), constant_values=40))
    legend = np.full((panels[0].shape[0], 200, 3), 40, np.uint8)
    for k in range(1, len(CLASSES)):
        cv2.rectangle(legend, (10, 20 + 28 * k), (34, 40 + 28 * k), PALETTE[k].tolist(), -1)
        text(legend, CLASSES[k], (42, 38 + 28 * k))
    bottom = np.hstack(panels + [legend])
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)), constant_values=40)
    head = np.full((36, w, 3), 40, np.uint8)
    text(head, title, (10, 25), 0.7)
    out = np.vstack([head, pad(top), np.full((8, w, 3), 40, np.uint8), pad(bottom)])
    return out[: out.shape[0] // 2 * 2, : out.shape[1] // 2 * 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="bevnet*.pt")
    p.add_argument("--dataset", help="dataset.json (default: output/<name>/dataset/dataset.json of the model)")
    p.add_argument("--scenes", nargs="+", help="val scene ids (default: every val scene)")
    p.add_argument("--out", help="output directory (default: <model dir>/videos)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps", type=float, default=4.0, help="video frames per second (samples are 0.5 s apart)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-mask", dest="mask", action="store_false", help="show the cells the model abstains on too")
    p.add_argument("--limit", type=int, default=0, help="samples per scene (0: all)")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    net, ck = load_checkpoint(a.model, a.device)
    cfg = ck["config"]
    ds_json = a.dataset or os.path.join(os.path.dirname(a.model), "..", "..", "dataset", "dataset.json")
    va = BEVDataset(ds_json, split="val", scenes=a.scenes, bev=cfg["bev"], image_size=tuple(cfg["image_size"]),
                    history=cfg.get("history", 1))  # fmt: skip
    cell = cfg["out_res"] or cfg["cell"]
    scale = max(1, 600 // int(round((cfg["bev"]["x"][1] - cfg["bev"]["x"][0]) / cell)))
    out = a.out or os.path.join(os.path.dirname(a.model), "videos")
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(a.model)[:-3]
    by_scene = {}
    for i, (sid, _) in enumerate(va.items):
        by_scene.setdefault(sid, []).append(i)
    for sid, idx in by_scene.items():
        idx = idx[: a.limit] if a.limit else idx
        path = f"{out}/{name}_{sid}.mp4"
        vw, t0, sc = None, time.time(), Scores(cfg["bev"]["res"])
        for j, i in enumerate(idx):
            x = collate([va[i]])
            with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == "cuda"):
                o = net(x)
            pr = predict(o)
            pk = predict(o, ck.get("known", False), net.vis_depth < VIS_DEPTH)
            sc.add(upsample(pr[0].cpu(), net.f), x["bev_cls"][0])
            y = downsample(x["bev_cls"], net.f)[0].numpy()
            t = (va.items[i][1]["stamp"] - va.items[idx[0]][1]["stamp"]) * 1e-9
            img = compose(x, y, (pk if a.mask else pr)[0].cpu().numpy(), ck["cams"], cfg["bev"], cell, scale,
                          f"{name}  scene {sid}  sample {j + 1}/{len(idx)}  t={t:.1f} s")  # fmt: skip
            if vw is None:
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (img.shape[1], img.shape[0]))
            vw.write(img[..., ::-1])
        vw.release()
        r = sc.result()
        print(
            f"{sid}: {len(idx)} samples, {time.time() - t0:.0f} s -> {path}  miou {r['miou']} iou {r['iou']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
