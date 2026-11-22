"""Train and evaluate BEVSegNet on an exported dataset.

usage: python -m hdlabel.bevnet.train output/<name>/dataset/dataset.json [more dataset.json ...] [--steps 6000]
       [--tag NAME]
       -> output/<name>/debug/eval/bevnet<tag>.json (args, curve, val scores), .pt (checkpoint without the frozen
          ViT weights), _examples.png (label | prediction for 3 val samples). <name> is the first dataset's.

Several datasets (e.g. areas) train together; val scores are given per dataset (`val`: the first one,
`val_by_dataset`: all). All datasets must have the same cameras.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from hdlabel.common import PALETTE

from .data import NC, BEVDataset, collate, downsample
from .losses import class_weights, seg_loss
from .metrics import VIS_DEPTH, evaluate, predict
from .model import BEVSegNet


def save_checkpoint(net, path, **extra):
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("enc.vit.")}  # frozen: from the hub
    torch.save(dict(config=net.config, state=state, **extra), path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    net = BEVSegNet(**ck["config"])
    missing, unexpected = net.load_state_dict(ck["state"], strict=False)
    assert not unexpected and all(k.startswith("enc.vit.") for k in missing), (missing, unexpected)
    return net.to(device).eval(), ck


def examples(net, ds, known, path, n=3):
    tiles = []
    for i in np.linspace(0, len(ds) - 1, n + 2).astype(int)[1:-1]:
        x = collate([ds[i]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        y = downsample(x["bev_cls"], net.f)[0].numpy()
        pr = predict(out, known, net.vis_depth < VIS_DEPTH)[0].cpu().numpy()
        tiles.append(np.hstack([PALETTE[y], np.full((len(y), 4, 3), 255, np.uint8), PALETTE[pr]]))
    cv2.imwrite(path, np.vstack(tiles)[..., ::-1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_json", nargs="+")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=1000, help="score 100 val samples every N steps (0: off)")
    p.add_argument("--val", type=int, default=0, help="val samples for the final scores (0: all)")
    p.add_argument("--bev-x", type=float, nargs=2, default=[-20.0, 60.0], help="BEV x range (m)")
    p.add_argument("--bev-y", type=float, nargs=2, default=[-25.6, 25.6], help="BEV y range (m)")
    p.add_argument("--label-res", type=float, default=0.1, help="label raster cell (m)")
    p.add_argument("--cell", type=float, default=0.4, help="context (BEV U-Net) cell (m)")
    p.add_argument("--out-res", type=float, default=0.1, help="output cell (m); 0: the context cell")
    p.add_argument("--image-size", type=int, nargs=2, default=[704, 448])
    p.add_argument("--heights", type=float, nargs="+", default=[-0.5, 0.0, 0.5, 1.0, 2.0], help="lifting heights (m)")
    p.add_argument("--history", type=int, default=1, help="frames per camera: current + earlier (export spacing)")
    p.add_argument("--vit", default="facebook/dinov2-base", help="frozen Hugging Face ViT ('' for none)")
    p.add_argument("--rot", type=float, default=22.5, help="BEV rotation augmentation, +- degrees")
    p.add_argument("--no-flip", dest="flip", action="store_false", help="no BEV mirror augmentation")
    p.add_argument("--jitter", type=float, default=0.2, help="image brightness / contrast jitter")
    p.add_argument("--lovasz", type=float, default=0.5, help="weight of the Lovasz-softmax loss")
    p.add_argument("--known", type=float, default=0.5, help="weight of the abstention (cell unlabelled) loss")
    p.add_argument("--edge", type=float, default=0.0, help="extra cross-entropy weight of cells near class boundaries")
    p.add_argument("--aux", type=float, default=0.5, help="weight of the loss on the context logits")
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"

    bev = dict(json.load(open(a.dataset_json[0]))["bev"], x=a.bev_x, y=a.bev_y, res=a.label_res)
    kw = dict(bev=bev, image_size=tuple(a.image_size), history=a.history)
    trs = [BEVDataset(d, split="train", rot_deg=a.rot, flip=a.flip, **kw) for d in a.dataset_json]
    vas = [BEVDataset(d, split="val", **kw) for d in a.dataset_json]
    assert all(t.cams == trs[0].cams for t in trs), "datasets with different cameras"
    tr, va = trs[0], vas[0]
    train_set = torch.utils.data.ConcatDataset(trs) if len(trs) > 1 else tr
    net = BEVSegNet(
        bev, len(tr.cams), a.image_size, a.cell, a.out_res or None, a.heights, a.vit or None, a.jitter, a.history
    )
    net = net.to(dev)
    known = a.known > 0

    cnt = torch.zeros(NC)
    for t in trs:  # class frequencies of each dataset in proportion to its size
        for i in np.linspace(0, len(t) - 1, max(1, round(200 * len(t) / len(train_set)))).astype(int):
            sid, r = t.items[i]
            lab = t.scene_map(sid).crop(np.array(r["T_map_base"]), bev["x"], bev["y"], bev["res"])[0]
            cnt += torch.bincount(torch.from_numpy(lab.astype(np.int64)).ravel(), minlength=NC)
    wts = class_weights(cnt).to(dev)

    cnn = [q for n, q in net.named_parameters() if q.requires_grad and n.startswith(("enc.stem", "enc.body"))]
    rest = [q for n, q in net.named_parameters() if q.requires_grad and not n.startswith(("enc.stem", "enc.body"))]
    opt = torch.optim.AdamW([dict(params=cnn, lr=a.lr / 3), dict(params=rest, lr=a.lr)], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr / 3, a.lr], total_steps=a.steps, pct_start=0.05)
    dl = torch.utils.data.DataLoader(train_set, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=2, pin_memory=True)  # fmt: skip
    t0, step, losses, curve = time.time(), 0, [], []
    while step < a.steps:
        for x in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            loss = seg_loss(out, downsample(x["bev_cls"], net.f).to(dev), wts, a.lovasz, a.known, a.edge)
            if net.ctx_logits is not None and a.aux:
                loss = loss + a.aux * seg_loss(net.ctx_logits, downsample(x["bev_cls"], net.fc).to(dev), wts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                gb = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"step {step} loss {np.mean(losses[-50:]):.3f} ({time.time() - t0:.0f} s, {gb:.1f} GB)", flush=True
                )
            if a.eval_every and step % a.eval_every == 0 and step < a.steps:
                s = evaluate(net, va, 100, known)
                curve.append(dict(step=step, miou=s["miou"], boundary_f=s["boundary_f"]))
                print("  val100", curve[-1], flush=True)
            if step >= a.steps:
                break

    res = dict(model="bevnet", args=vars(a), train_samples=len(train_set), val_samples=len(va),
               minutes_train=round((time.time() - t0) / 60, 1), loss_last50=round(float(np.mean(losses[-50:])), 3),
               curve=curve)  # fmt: skip
    res["val"] = evaluate(net, va, a.val, known)
    if len(vas) > 1:
        res["val_by_dataset"] = {d: res["val"] if v is va else evaluate(net, v, a.val, known)
                                 for d, v in zip(a.dataset_json, vas)}  # fmt: skip
    res["minutes_total"] = round((time.time() - t0) / 60, 1)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.dataset_json[0]))), "debug", "eval")
    os.makedirs(out, exist_ok=True)
    name = f"{out}/bevnet{a.tag}"
    json.dump(res, open(f"{name}.json", "w"), indent=1)
    save_checkpoint(net, f"{name}.pt", known=known, cams=tr.cams)
    net.eval()
    examples(net, va, known, f"{name}_examples.png")
    print(json.dumps(res["val"]))


if __name__ == "__main__":
    main()
