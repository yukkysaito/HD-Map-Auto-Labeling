"""Acceptance check of an exported dataset (docs/DATASET.md), for the training side: needs only dataset/ itself.

usage: ./autolabel validate output/<name>/dataset/dataset.json [--out report.json] [--read-images]

Errors (exit code 1): unsupported format, missing / unreadable files, images whose size differs from the camera's,
broken references (sample -> frame -> image), non-monotonic sample times, non-rigid poses, pose jumps, vectors without
coordinates. Warnings: low QA pass rate, little road / observed area, lane-line vectors that mostly do not lie on
marking cells of the raster (labels and vectors disagree).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

SUPPORTED = (3,)


class Report:
    def __init__(self):
        self.errors, self.warnings, self.scenes = [], [], {}

    def error(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


def check_pose(T, where, rep):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return rep.error(where, "pose is not a finite 4x4 matrix")
    R = T[:3, :3]
    if (
        not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
        or abs(np.linalg.det(R) - 1) > 1e-3
        or not np.allclose(T[3], [0, 0, 0, 1])
    ):
        rep.error(where, "pose is not a rigid transform")


def lane_on_marking(m, vectors):
    """share of the PAINTED parts of lane-line / stop-line vectors (lane lines: their `segments`, not the gaps of dashed
    lines) that lie within 0.2 m of a marking cell of the raster."""
    from scipy.ndimage import binary_dilation

    mk = binary_dilation(m["cls"] == 2, iterations=2)
    lo, res, org = m["lo"], float(m["res"]), m["origin"]
    hit = n = 0
    for f in vectors:
        if f["properties"]["kind"] not in ("lane_line", "stop_line"):
            continue
        P = np.asarray(f["geometry"]["coordinates"], float)[:, :2]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        at = np.arange(0, s[-1] + 1e-9, 0.5)
        segs = f["properties"].get("segments")
        if segs:
            at = at[np.any([(at >= a) & (at <= b) for a, b in segs], 0)]
        q = np.stack([np.interp(at, s, P[:, k]) for k in (0, 1)], 1)
        ij = np.floor((q - org[:2] - lo) / res).astype(int)
        ok = np.all((ij >= 0) & (ij < mk.shape), 1)
        hit += int(mk[ij[ok, 0], ij[ok, 1]].sum())
        n += int(len(q))
    return hit / n if n else None


def check_scene(root, s, rep, read_images, max_speed=60.0):
    d = os.path.join(root, s["path"])
    w = s["scene"]
    for f in ("scene.json", "map.npz", "vectors.geojson", "samples.jsonl", "frames.jsonl"):
        if not os.path.exists(f"{d}/{f}"):
            return rep.error(w, f"missing {f}")
    sc = json.load(open(f"{d}/scene.json"))
    if sc.get("format") not in SUPPORTED:
        rep.error(w, f"scene format {sc.get('format')} not in {SUPPORTED}")
    frames = {}
    for k, line in enumerate(open(f"{d}/frames.jsonl")):
        fr = json.loads(line)
        frames[fr["id"]] = fr
        cam = sc["cameras"].get(str(fr["cam"]))
        if cam is None:
            rep.error(w, f"frame {fr['id']}: camera {fr['cam']} not in scene.json")
            continue
        img = f"{d}/{fr['image']}"
        if not os.path.exists(img):
            rep.error(w, f"frame {fr['id']}: missing image {fr['image']}")
            continue
        try:
            im = Image.open(img)
            if list(im.size) != list(cam["size"]):
                rep.error(w, f"frame {fr['id']}: image size {im.size} != camera size {cam['size']}")
            if read_images:
                im.load()
        except Exception as e:
            rep.error(w, f"frame {fr['id']}: unreadable image ({e})")
        check_pose(fr["T_map_base"], f"{w} frame {fr['id']}", rep)
    samples = [json.loads(line) for line in open(f"{d}/samples.jsonl")]
    st = np.array([x["stamp"] for x in samples], np.int64)
    if len(st) > 1 and np.any(np.diff(st) <= 0):
        rep.error(w, "sample stamps are not strictly increasing")
    for x in samples:
        check_pose(x["T_map_base"], f"{w} sample {x['stamp']}", rep)
        for c, ids in x["frames"].items():
            missing = [i for i in ids if i is not None and i not in frames]
            if missing:
                rep.error(w, f"sample {x['stamp']}: cam {c} references unknown frames {missing[:3]}")
    if len(samples) > 1:
        p = np.array([np.asarray(x["T_map_base"])[:3, 3] for x in samples])
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(st) * 1e-9, 1e-6)
        if np.any(v > max_speed):
            rep.error(w, f"pose jump between samples: {v.max():.0f} m/s")
    m = np.load(f"{d}/map.npz")
    cls = m["cls"]
    observed = float((cls > 0).mean())
    road = float((cls == 1).sum() / max((cls > 0).sum(), 1))
    vectors = json.load(open(f"{d}/vectors.geojson"))["features"]
    for k, f in enumerate(vectors):
        if not f["geometry"]["coordinates"]:
            rep.error(w, f"vector {k} ({f['properties'].get('kind')}) has no coordinates")
    on_mk = lane_on_marking(m, vectors)
    qa = s["qa_ok"] / max(s["samples"], 1)
    if qa < 0.5:
        rep.warn(w, f"only {qa:.0%} of the samples pass QA")
    if observed < 0.02 or road < 0.2:
        rep.warn(w, f"little labelled area (observed {observed:.1%}, road share {road:.0%})")
    if on_mk is not None and on_mk < 0.8:
        rep.warn(w, f"only {on_mk:.0%} of the lane / stop line vector points lie on marking cells")
    rep.scenes[w] = dict(samples=len(samples), frames=len(frames), qa_pass=round(qa, 3), observed=round(observed, 3),
                         road_share=round(road, 3), lines_on_marking=None if on_mk is None else round(on_mk, 3))  # fmt: skip


def validate(dataset_json: str, read_images: bool = False) -> Report:
    rep = Report()
    idx = json.load(open(dataset_json))
    root = os.path.dirname(os.path.abspath(dataset_json))
    if idx.get("format") not in SUPPORTED:
        rep.error("dataset.json", f"format {idx.get('format')} not in {SUPPORTED}")
        return rep
    for s in idx["scenes"]:
        try:
            check_scene(root, s, rep, read_images)
        except Exception as e:  # a broken scene is reported, the others are still checked
            rep.error(s["scene"], f"{type(e).__name__}: {e}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_json")
    p.add_argument("--out", help="write the report (json) here")
    p.add_argument("--read-images", action="store_true", help="decode every image (slow); default: header + size only")
    a = p.parse_args(argv)
    rep = validate(a.dataset_json, a.read_images)
    if a.out:
        res = dict(ok=not rep.errors, errors=rep.errors, warnings=rep.warnings, scenes=rep.scenes)
        json.dump(res, open(a.out, "w"), indent=1)
    for e in rep.errors[:30]:
        print("ERROR", e)
    for x in rep.warnings[:30]:
        print("warning", x)
    print(f"{'OK' if not rep.errors else 'FAILED'}: {len(rep.scenes)} scenes, {len(rep.errors)} errors, "
          f"{len(rep.warnings)} warnings" + (f" -> {a.out}" if a.out else ""))  # fmt: skip
    return 0 if not rep.errors else 1


if __name__ == "__main__":
    sys.exit(main())
