"""Per-cycle (online) evaluation protocol for the intended online HD-map model.

The val scenes are exported at the model cycle (dataset.val_every_s, default 0.1 s). Each val scene is replayed in
time order; at every cycle the model gets that cycle's input (multi-camera images, current + history frames with ego
motion, as hdlabel.loader.HDMapDataset returns them) and must return BEV class scores. A model may keep state across
cycles (temporal fusion); reset() is called at the start of each scene.
Measured: per-class IoU against the raster labels (unobserved cells ignored) and the model time per cycle (GPU
synchronised; data loading excluded) vs. the cycle budget (--budget-ms).

Model interface (an object):  reset() ; step(sample: dict of batched tensors) -> logits [num_classes, h, w] (any h, w
that divides the 400 x 300 BEV; labels are reduced to it by majority).  Default: the smoke-test network
(eval/train_smoke.py checkpoint, current frame only) - a placeholder to exercise the protocol, not a real model.
usage: online_eval.py output/<name>/dataset/dataset.json [--ckpt output/<name>/debug/eval/train_smoke_h3.pt] [--budget-ms 100]
       -> output/<name>/debug/eval/online_eval_<ckpt>.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import CLASSES
from hdlabel.eval.train_smoke import DEV, H, W, SmokeBEV, collate, down
from hdlabel.loader import HDMapDataset


class SmokeModel:
    """placeholder model: the smoke-test network, stateless."""
    def __init__(self, ckpt):
        c = torch.load(ckpt, map_location=DEV); self.history = c.get("history", 1)
        self.net = SmokeBEV(c["bev"], c["cell"], pretrained=False, history=self.history).to(DEV).eval()
        self.net.load_state_dict(c["state"]); self.cams = c["cams"]
    def reset(self): pass
    @torch.no_grad()
    def step(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16): return self.net(x)[0].float()


def main():
    p = argparse.ArgumentParser(); p.add_argument("dataset_json"); p.add_argument("--ckpt", help="default: debug/eval/train_smoke.pt")
    p.add_argument("--history", type=int, default=1, help="frames loaded per cycle (at least the model's)"); p.add_argument("--budget-ms", type=float, default=100.0)
    a = p.parse_args(); out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.dataset_json))), "debug", "eval")
    model = SmokeModel(a.ckpt or f"{out}/train_smoke.pt")
    idx = json.load(open(a.dataset_json)); NC = len(CLASSES); res = dict(scenes={}, budget_ms=a.budget_ms)
    I_all = torch.zeros(NC); U_all = torch.zeros(NC); times = []
    for s in [s for s in idx["scenes"] if s["split"] == "val"]:
        ds = HDMapDataset(a.dataset_json, model.cams, split="val", scenes=[s["scene"]], history=max(a.history, getattr(model, "history", 1)),
                          image_size=(W, H))
        if not len(ds): continue
        cyc = np.diff([r["stamp"] for _, r in ds.items]) * 1e-9
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)  # keeps time order
        model.reset(); I = torch.zeros(NC); U = torch.zeros(NC); st = []
        for x in loader:
            torch.cuda.synchronize(); t = time.perf_counter(); logits = model.step(x); torch.cuda.synchronize(); st.append((time.perf_counter() - t) * 1e3)
            f = x["bev_cls"].shape[-1] // logits.shape[-1]; y = down(x["bev_cls"], f)[0]; m = y > 0
            pr = logits[1:].argmax(0).cpu() + 1
            for c in range(1, NC): I[c] += ((pr == c) & (y == c) & m).sum(); U[c] += (((pr == c) | (y == c)) & m).sum()
        I_all += I; U_all += U; times += st[1:]
        res["scenes"][s["scene"]] = dict(cycles=len(ds), cycle_s=round(float(np.median(cyc)), 3) if len(cyc) else None,
                                         miou=round(float((I[1:NC - 1] / U[1:NC - 1].clamp(min=1)).mean()), 3),
                                         model_ms_p50=round(float(np.median(st)), 1), model_ms_p95=round(float(np.percentile(st, 95)), 1))
        print(s["scene"], res["scenes"][s["scene"]], flush=True)
    res["iou"] = {CLASSES[c]: round(float(I_all[c] / U_all[c].clamp(min=1)), 3) for c in range(1, NC - 1)}
    res["model_ms_p50"] = round(float(np.median(times)), 1); res["model_ms_p95"] = round(float(np.percentile(times, 95)), 1)
    res["within_budget"] = bool(res["model_ms_p95"] <= a.budget_ms)
    res["ckpt"] = os.path.basename(a.ckpt or "train_smoke.pt")
    json.dump(res, open(f"{out}/online_eval_{os.path.splitext(res['ckpt'])[0]}.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "scenes"}, indent=1))


if __name__ == "__main__":
    main()
