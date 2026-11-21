"""Score a trained BEVSegNet on the val split of a dataset (e.g. the same model on re-made labels).

usage: python -m hdlabel.bevnet.evaluate output/<name>/debug/eval/bevnet<tag>.pt [--dataset dataset.json] [--val N]
       -> the val scores as JSON on stdout
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
    p.add_argument("--val", type=int, default=0, help="val samples (0: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    net, ck = load_checkpoint(a.model, a.device)
    cfg = ck["config"]
    ds_json = a.dataset or os.path.join(os.path.dirname(a.model), "..", "..", "dataset", "dataset.json")
    va = BEVDataset(ds_json, split="val", bev=cfg["bev"], image_size=tuple(cfg["image_size"]),
                    history=cfg.get("history", 1))  # fmt: skip
    print(json.dumps(evaluate(net, va, a.val, ck.get("known", False))))


if __name__ == "__main__":
    main()
