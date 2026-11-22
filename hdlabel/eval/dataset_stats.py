"""Dataset statistics for model design: per split, raster class shares (observed cells; for loss weights), share of
unobserved cells by distance ahead, vector instances per sample and kind, lane line styles, arrow directions, QA failures.
usage: dataset_stats.py output/<name>/dataset/dataset.json [--every 5] [--out FILE]"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.bev import SceneMap, grid
from hdlabel.common import CLASSES


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dataset_json")
    p.add_argument("--every", type=int, default=5, help="use every n-th sample")
    p.add_argument("--out", help="default: dataset_stats.json next to dataset.json")
    a = p.parse_args(argv)
    idx = json.load(open(a.dataset_json))
    root = os.path.dirname(os.path.abspath(a.dataset_json))
    bev = idx["bev"]
    out = {}
    xs = grid(bev["x"], bev["y"], bev["res"])[0][:, 0]
    for split in ("train", "val", "test"):
        cls_n = np.zeros(len(CLASSES))
        unobs_by_x = {}
        vec = Counter()
        styles = Counter()
        dirs = Counter()
        per_sample = []
        qa_fail = Counter()
        n = 0
        for s in (s for s in idx["scenes"] if s["split"] == split):
            d = os.path.join(root, s["path"])
            sm = SceneMap(f"{d}/map.npz", f"{d}/vectors.geojson")
            for k, line in enumerate(open(f"{d}/samples.jsonl")):
                r = json.loads(line)
                q = r["qa"]
                if not q["ok"]:
                    qa_fail[
                        "observed"
                        if q["observed_near"] < 0.5
                        else "pose"
                        if not q.get("pose_ok", True)
                        else "dt / calib / review"
                    ] += 1
                    continue
                if k % a.every:
                    continue
                T = np.array(r["T_map_base"])
                c, _ = sm.crop(T, bev["x"], bev["y"], bev["res"])
                els = sm.vectors(T, bev["x"], bev["y"])
                n += 1
                cls_n += np.bincount(c.ravel(), minlength=len(CLASSES))
                for x0 in range(int(bev["x"][0]), int(bev["x"][1]), 20):
                    rows = (xs >= x0) & (xs < x0 + 20)
                    unobs_by_x.setdefault(x0, []).append(float((c[rows] == 0).mean()))
                kinds = Counter(
                    e["kind"] + ("/" + ("observed" if e["observed"] else "unknown") if "observed" in e else "")
                    for e in els
                )
                vec.update(kinds)
                per_sample.append(sum(kinds.values()))
                styles.update(e["style"] for e in els if e["kind"] == "lane_line")
                dirs.update("+".join(e["directions"]) for e in els if e["kind"] == "arrow")
        obs = cls_n[1:].sum()
        out[split] = dict(
            samples_used=n,
            class_share_observed={c: round(float(v / max(obs, 1)), 4) for c, v in zip(CLASSES[1:], cls_n[1:])},
            unobserved_by_x={f"{x}..{x + 20} m": round(float(np.mean(v)), 3) for x, v in sorted(unobs_by_x.items())},
            vectors_per_sample=dict(
                mean=round(float(np.mean(per_sample)), 1) if per_sample else 0, max=int(max(per_sample, default=0))
            ),
            vector_kinds_per_sample={k: round(v / max(n, 1), 1) for k, v in vec.most_common()},
            lane_line_styles=dict(styles),
            arrow_directions=dict(dirs),
            qa_failed=dict(qa_fail),
        )
    json.dump(out, open(a.out or os.path.join(root, "dataset_stats.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
