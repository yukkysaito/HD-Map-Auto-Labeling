"""Semantic segmentation of all keyframes of a drive: Mask2Former (Mapillary Vistas) -> DATA/seg_cam<N>/<stamp>.png
(uint8 Vistas class id; common.GROUPS maps them to label classes). Already segmented images are skipped.
Writes DATA/seg_done.json (images per camera) when every keyframe is segmented."""
import argparse, glob, json, os, warnings

import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="data/<drive> with kf_cam*"); p.add_argument("--batch", type=int, default=4)
p.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic")
a = p.parse_args()
todo = {}
for kd in sorted(glob.glob(f"{a.data}/kf_cam*")):
    out = kd.replace("/kf_cam", "/seg_cam"); os.makedirs(out, exist_ok=True)
    files = [f for f in sorted(glob.glob(kd + "/*.jpg")) if not os.path.exists(f"{out}/{os.path.basename(f)[:-4]}.png")]
    if files: todo[out] = files
def done():
    n = {os.path.basename(d): len(glob.glob(f"{d}/*.png")) for d in sorted(glob.glob(f"{a.data}/seg_cam*"))}
    json.dump(dict(model=a.model, images=n), open(f"{a.data}/seg_done.json", "w")); print(f"all keyframes segmented: {n}")
if not todo: done(); raise SystemExit
warnings.filterwarnings("ignore")
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
proc = AutoImageProcessor.from_pretrained(a.model, use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(a.model).cuda().half().eval()
for out, files in todo.items():
    json.dump(model.config.id2label, open(f"{out}/labels.json", "w"))
    for b in range(0, len(files), a.batch):
        ims = [Image.open(f).convert("RGB") for f in files[b:b + a.batch]]
        with torch.no_grad():
            inp = proc(images=ims, return_tensors="pt").to("cuda"); inp["pixel_values"] = inp["pixel_values"].half()
            segs = proc.post_process_semantic_segmentation(model(**inp), target_sizes=[im.size[::-1] for im in ims])
        for f, s in zip(files[b:b + a.batch], segs):
            Image.fromarray(s.cpu().numpy().astype(np.uint8)).save(f"{out}/{os.path.basename(f)[:-4]}.png")
    print(f"segmented {len(files)} images -> {out}", flush=True)
done()
