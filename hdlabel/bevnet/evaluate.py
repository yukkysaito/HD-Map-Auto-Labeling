"""Score a trained BEVSegNet on a dataset (e.g. the same model on re-made labels, or on an unseen area).

usage: python -m hdlabel.bevnet.evaluate output/<name>/debug/eval/bevnet<tag>.pt [--dataset dataset.json]
       [--split val|train|all] [--val N] -> the scores as JSON on stdout.
       `--split all` scores every sample, e.g. of an area the model has never seen; `--still-every 2` scores a
       standing vehicle once per 2 s (waits at lights otherwise weigh one view many times)
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .data import BEVDataset
from .metrics import evaluate
from .train import load_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="bevnet*.pt")
    p.add_argument("--dataset", help="dataset.json (default: output/<name>/dataset/dataset.json of the model)")
    p.add_argument("--split", default="val", choices=["val", "test", "train", "all"])
    p.add_argument("--val", type=int, default=0, help="samples (0: all)")
    p.add_argument(
        "--still-every", type=float, default=0.0, help="one sample per N s while the vehicle stands (0: all)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    net, ck = load_checkpoint(a.model, a.device)
    cfg = ck["config"]
    ds_json = a.dataset or os.path.join(os.path.dirname(a.model), "..", "..", "dataset", "dataset.json")
    split = None if a.split == "all" else a.split
    va = BEVDataset(ds_json, split=split, bev=cfg["bev"], image_size=tuple(cfg["image_size"]),
                    history=cfg.get("history", 1), still_every_s=a.still_every)  # fmt: skip
    print(json.dumps(evaluate(net, va, a.val, ck.get("known", False))))


if __name__ == "__main__":
    main()
