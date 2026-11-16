"""Semantic segmentation of all keyframes of a drive -> DATA/seg_cam<N>/<stamp>.png (uint8 Mapillary Vistas class id;
common.GROUPS maps them to label classes). Already segmented images are skipped.
--model: a Hugging Face Mask2Former checkpoint with Vistas classes (default), or `color_stub`: every pixel takes the
class of the nearest STUB_PALETTE colour (synthetic test / demo data only, see tests/synthetic.py).
Writes DATA/seg_done.json (images per camera) when every keyframe is segmented."""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import device, save_png

# color_stub: RGB of the synthetic world -> Vistas class id (13 road, 24 lane marking, 23 crosswalk, 2 curb,
# 15 sidewalk, 30 vegetation, 27 sky)
STUB_PALETTE = dict(road=(100, 100, 100), marking=(240, 240, 240), crosswalk=(235, 235, 120), curb=(170, 60, 60),
                    sidewalk=(190, 160, 120), grass=(40, 140, 40), sky=(150, 190, 250))  # fmt: skip
STUB_IDS = dict(road=13, marking=24, crosswalk=23, curb=2, sidewalk=15, grass=30, sky=27)


def color_stub(img: np.ndarray) -> np.ndarray:
    """RGB image -> Vistas ids of the nearest STUB_PALETTE colour."""
    pal = np.array(list(STUB_PALETTE.values()), np.float32)
    d = ((img[..., None, :].astype(np.float32) - pal) ** 2).sum(-1)
    return np.array([STUB_IDS[k] for k in STUB_PALETTE], np.uint8)[d.argmin(-1)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="data/<drive> with kf_cam*")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic")
    a = p.parse_args(argv)
    todo = {}
    for kd in sorted(glob.glob(f"{a.data}/kf_cam*")):
        out = kd.replace("/kf_cam", "/seg_cam")
        os.makedirs(out, exist_ok=True)
        files = [
            f for f in sorted(glob.glob(kd + "/*.jpg")) if not os.path.exists(f"{out}/{os.path.basename(f)[:-4]}.png")
        ]
        if files:
            todo[out] = files

    def done():
        n = {os.path.basename(d): len(glob.glob(f"{d}/*.png")) for d in sorted(glob.glob(f"{a.data}/seg_cam*"))}
        txt = json.dumps(dict(model=a.model, images=n))
        f = f"{a.data}/seg_done.json"
        if not os.path.exists(f) or open(f).read() != txt:  # unchanged: keep its time (nothing downstream is redone)
            open(f, "w").write(txt)
        print(f"all keyframes segmented: {n}")

    if not todo:
        done()
        return
    if a.model == "color_stub":
        for out, files in todo.items():
            for f in files:
                save_png(f"{out}/{os.path.basename(f)[:-4]}.png", color_stub(np.asarray(Image.open(f).convert("RGB"))))
        done()
        return
    warnings.filterwarnings("ignore")
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    proc = AutoImageProcessor.from_pretrained(a.model, use_fast=True)
    dev = device()
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    model = Mask2FormerForUniversalSegmentation.from_pretrained(a.model).to(dev, dtype).eval()
    for out, files in todo.items():
        json.dump(model.config.id2label, open(f"{out}/labels.json", "w"))
        for b in range(0, len(files), a.batch):
            ims = [Image.open(f).convert("RGB") for f in files[b : b + a.batch]]
            with torch.no_grad():
                inp = proc(images=ims, return_tensors="pt").to(dev)
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                segs = proc.post_process_semantic_segmentation(model(**inp), target_sizes=[im.size[::-1] for im in ims])
            for f, s in zip(files[b : b + a.batch], segs):
                save_png(
                    f"{out}/{os.path.basename(f)[:-4]}.png", s.cpu().numpy().astype(np.uint8)
                )  # complete or absent
        print(f"segmented {len(files)} images -> {out}", flush=True)
    done()


if __name__ == "__main__":
    main()
