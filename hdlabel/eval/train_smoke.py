"""Smoke test of the dataset for the intended model (NOT the model): a tiny multi-camera BEV network is trained for a
few minutes on the train split and evaluated on the val split, to show that the labels are learnable from the camera
inputs with the given calibration / poses (a broken format, wrong extrinsics or misaligned labels would not train).

Network: ResNet18 features per camera (stride 8) -> every BEV cell (0.8 m) is projected into each camera on the
base_link plane z = 0 and the features are sampled and averaged over the cameras that see it -> 3 conv layers ->
class logits; cross entropy on the labels downsampled to 0.8 m (unobserved = ignore). With --history K the same is done
for the K - 1 earlier frames (0.5 s apart) through their ego motion (T_cur_frame) and the K BEV feature maps are
concatenated: a minimal multi-frame, multi-camera baseline (temporal fusion by ego-motion alignment).
Reports val IoU per class vs. a prior baseline (the most frequent train class per BEV cell).
Overfit test (do the labels carry learnable shape / position?): --scenes <one scene> --eval-split train --cell 0.4.
usage: train_smoke.py output/<name>/dataset/dataset.json [--steps 600]
       -> output/<name>/debug/eval/train_smoke.json|pt, train_smoke_examples.png (label | prediction for 3 train samples)
"""
import argparse, json, os, sys, time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel.common import CLASSES, PALETTE
from hdlabel.loader import HDMapDataset

NC = len(CLASSES); W, H = 352, 224; DEV = "cuda"


def collate(b): return {k: torch.stack([e[k] for e in b]) for k in b[0] if torch.is_tensor(b[0][k])}


def down(cls, f):
    """labels to the coarse grid (f x f blocks): majority of the observed classes (0 if none observed)."""
    b, h, w = cls.shape; c = cls.reshape(b, h // f, f, w // f, f).permute(0, 1, 3, 2, 4).reshape(b, h // f, w // f, f * f)
    oh = F.one_hot(c, NC)[..., 1:].sum(-2); return torch.where(oh.sum(-1) > 0, oh.argmax(-1) + 1, torch.zeros_like(oh[..., 0]))


class SmokeBEV(nn.Module):
    def __init__(self, bev: dict, cell: float = 0.8, pretrained: bool = True, history: int = 1):
        super().__init__()
        try: r = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
        except Exception: r = torchvision.models.resnet18(weights=None)
        self.enc = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)          # stride 8, 128 ch
        self.history = history
        self.head = nn.Sequential(nn.Conv2d(128 * history, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, NC, 1))
        xs = np.arange(bev["x"][1] - cell / 2, bev["x"][0], -cell); ys = np.arange(bev["y"][1] - cell / 2, bev["y"][0], -cell)
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("P", torch.tensor(np.stack([GX, GY, np.zeros_like(GX), np.ones_like(GX)], -1), dtype=torch.float32))
        self.f = int(round(cell / bev["res"]))

    def forward(self, x):
        return self.head(torch.cat([self.lift(x, k) for k in range(self.history)], 1))

    def lift(self, x, k):
        """frame k (0 = current) of all cameras -> BEV features of the current ego frame."""
        dev = self.P.device; img = x["images"][:, k].to(dev); B, C = img.shape[:2]
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
        feat = self.enc(((img - mean) / std).flatten(0, 1)); fh, fw = feat.shape[-2:]; feat = feat.view(B, C, -1, fh, fw)
        with torch.autocast("cuda", enabled=False):                                          # geometry in float32
            T = torch.linalg.inv(x["T_cur_frame"][:, k].to(dev) @ x["T_base_cam"].to(dev))       # current base -> camera at frame k
            q = torch.einsum("bcij,xyj->bcxyi", T, self.P)[..., :3]; z = q[..., 2:3]
            uv = torch.einsum("bcij,bcxyj->bcxyi", x["K"].to(dev), q)[..., :2] / z.clamp(min=0.1)
            g = torch.stack([uv[..., 0] / W * 2 - 1, uv[..., 1] / H * 2 - 1], -1)
        ok = (z[..., 0] > 0.5) & (g.abs() <= 1).all(-1) & x["cam_mask"].to(dev)[:, :, None, None]
        s = F.grid_sample(feat.flatten(0, 1), g.flatten(0, 1).to(feat.dtype), align_corners=False).view(B, C, -1, *g.shape[2:4])
        return (s * ok[:, :, None]).sum(1) / ok.sum(1).clamp(min=1)[:, None]


def main():
    p = argparse.ArgumentParser(); p.add_argument("dataset_json"); p.add_argument("--steps", type=int, default=600)
    p.add_argument("--cams", type=int, nargs="+", help="default: every camera of the dataset"); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--val", type=int, default=150, help="val samples to evaluate"); p.add_argument("--cell", type=float, default=0.8)
    p.add_argument("--history", type=int, default=1, help="frames per input (current + K-1 earlier, 0.5 s apart)")
    p.add_argument("--scenes", nargs="+", help="train on these scenes only (with --eval-split train: an overfit test)")
    p.add_argument("--eval-split", default="val", help="split to evaluate on; 'train' = how well the labels are reproduced")
    p.add_argument("--bev", type=float, nargs=5, metavar=("X0", "X1", "Y0", "Y1", "RES"), help="BEV crop (default: dataset.json)")
    p.add_argument("--tag", default="", help="suffix of the output files")
    p.add_argument("--init", help="continue from this train_smoke .pt (same --cell / --history / --bev)")
    a = p.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    bev_ = dict(x=a.bev[:2], y=a.bev[2:4], res=a.bev[4]) if a.bev else None
    mk = lambda split: HDMapDataset(a.dataset_json, a.cams, split=split, history=a.history, image_size=(W, H), bev=bev_, scenes=a.scenes)
    tr, va = mk("train"), mk(a.eval_split)
    bev = tr.bev
    net = SmokeBEV(bev, a.cell, history=a.history).to(DEV); f = net.f; opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    if a.init: net.load_state_dict(torch.load(a.init, map_location=DEV)["state"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=8, collate_fn=collate, drop_last=True, persistent_workers=True)
    # class weights (inverse sqrt frequency) and the prior baseline from the train labels
    cnt = torch.zeros(NC); prior = 0
    for i in np.linspace(0, len(tr) - 1, 200).astype(int):
        y = down(tr[i]["bev_cls"][None], f)[0]; cnt += torch.bincount(y.ravel(), minlength=NC); prior = prior + F.one_hot(y, NC).permute(2, 0, 1)
    wts = (cnt.sum() / cnt.clamp(min=1)).sqrt(); wts[0] = 0; wts = (wts / wts[1:].mean()).to(DEV); prior_cls = prior[1:].argmax(0) + 1
    t0, step, losses = time.time(), 0, []
    while step < a.steps:
        for x in dl:
            y = down(x["bev_cls"], f).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16): out = net(x)
            loss = F.cross_entropy(out.float(), y, weight=wts, ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(float(loss)); step += 1
            if step % 100 == 0: print(f"step {step} loss {np.mean(losses[-100:]):.3f} ({time.time() - t0:.0f} s)", flush=True)
            if step >= a.steps: break
    net.eval(); inter = torch.zeros(NC); union = torch.zeros(NC); pi = torch.zeros(NC); pu = torch.zeros(NC)
    with torch.no_grad():
        for i in np.linspace(0, len(va) - 1, min(a.val, len(va))).astype(int):
            x = collate([va[i]]); y = down(x["bev_cls"], f)[0]; m = y > 0
            pr = net(x)[0, 1:].argmax(0).cpu() + 1
            for c in range(1, NC):
                for pred, I, U in ((pr, inter, union), (prior_cls, pi, pu)):
                    I[c] += ((pred == c) & (y == c) & m).sum(); U[c] += (((pred == c) | (y == c)) & m).sum()
    res = dict(steps=a.steps, train_samples=len(tr), val_samples=len(va), minutes=round((time.time() - t0) / 60, 1),
               loss_first100=round(float(np.mean(losses[:100])), 3), loss_last100=round(float(np.mean(losses[-100:])), 3),
               val_iou={CLASSES[c]: dict(model=round(float(inter[c] / union[c].clamp(min=1)), 3), prior=round(float(pi[c] / pu[c].clamp(min=1)), 3))
                        for c in range(1, NC - 1)})
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.dataset_json))), "debug", "eval"); os.makedirs(out, exist_ok=True); tag = ("" if a.history == 1 else f"_h{a.history}") + a.tag; res["history"] = a.history; res["scenes"] = a.scenes; res["eval_split"] = a.eval_split; res["init"] = a.init
    json.dump(res, open(f"{out}/train_smoke{tag}.json", "w"), indent=1)
    torch.save(dict(state=net.state_dict(), cams=tr.cams, cell=a.cell, bev=bev, history=a.history), f"{out}/train_smoke{tag}.pt")
    tiles = []                                                  # label | prediction (observed cells) for 3 train samples
    with torch.no_grad():
        for i in np.linspace(0, len(tr) - 1, 5).astype(int)[1:4]:
            x = collate([tr[i]]); y = down(x["bev_cls"], f)[0].numpy(); pr = net(x)[0, 1:].argmax(0).cpu().numpy() + 1
            tiles.append(np.hstack([PALETTE[y], np.full((len(y), 4, 3), 255, np.uint8), PALETTE[np.where(y > 0, pr, 0)]]))
    cv2.imwrite(f"{out}/train_smoke{tag}_examples.png", cv2.resize(np.vstack(tiles)[..., ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
