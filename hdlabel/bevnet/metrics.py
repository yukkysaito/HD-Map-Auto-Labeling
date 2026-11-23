"""Prediction and evaluation against the label raster, always at the label resolution (comparable across models with
different output cells)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from hdlabel.common import CLASSES

from .data import NC, collate
from .losses import SURFACE, block_mean

VIS_DEPTH = 50.0  # m: cells no camera sees closer than this are unknown
THIN = ["marking", "crosswalk", "curb"]  # classes whose geometry is scored by the boundary F-score


def predict(out, known=False, vis=None):
    """class map [B, X, Y] from the logits [B, NC, X, Y]: argmax over the classes; 0 (unknown) where the known head
    (channel 0) says the cell is unlabelled, and outside vis (bool [B, x, y], any grid; e.g. vis_depth < VIS_DEPTH)."""
    pr = out[:, 1:].float().argmax(1) + 1
    if known:
        pr[out[:, 0].float() > 0] = 0
    if vis is not None:
        pr[~(F.interpolate(vis[:, None].float(), size=pr.shape[-2:], mode="nearest")[:, 0] > 0)] = 0
    return pr


def upsample(pr, f):
    return pr.repeat_interleave(f, -2).repeat_interleave(f, -1) if f > 1 else pr


class Scores:
    """accumulates over samples: IoU per class over the labelled cells; boundary F-score of the THIN classes (a
    predicted cell is correct if a label cell of its class lies within tol, and vice versa) and of the edge of the
    drivable surface (road_edge: SURFACE cells next to other labelled classes); IoU of the predicted known region (not
    abstained) with the labelled region."""

    def __init__(self, res: float, tol: float = 0.2):
        self.r = max(1, int(round(tol / res)))
        self.inter, self.union = torch.zeros(NC), torch.zeros(NC)
        self.bf = {c: torch.zeros(4) for c in THIN + ["road_edge"]}  # tp_pred, n_pred, tp_label, n_label
        self.surface = [CLASSES.index(c) for c in SURFACE]
        self.known = torch.zeros(2)

    def dilate(self, m):
        k = 2 * self.r + 1
        return F.max_pool2d(m[None, None].float(), k, 1, self.r)[0, 0] > 0

    @staticmethod
    def edge(surf, m):
        """surface cells with a labelled non-surface 4-neighbour (edges against unlabelled cells do not count)."""
        other = F.pad((~surf & m)[None, None].float(), (1, 1, 1, 1))[0, 0]
        nb = (other[:-2, 1:-1] + other[2:, 1:-1] + other[1:-1, :-2] + other[1:-1, 2:]) > 0
        return surf & m & nb

    def add(self, pr, y, pk=None):
        """pr, y [X, Y] class rasters at the label resolution; pk: prediction with abstention (0 = unknown)."""
        m = y > 0
        for c in range(1, NC):
            p, t = (pr == c) & m, y == c
            self.inter[c] += (p & t).sum()
            self.union[c] += (p | t).sum()
        for name in THIN:
            c = CLASSES.index(name)
            p, t = (pr == c) & m, y == c
            self.bf[name] += torch.tensor([(p & self.dilate(t)).sum(), p.sum(), (t & self.dilate(p)).sum(), t.sum()])
        p, t = (
            self.edge(torch.isin(pr, torch.tensor(self.surface)), m),
            self.edge(torch.isin(y, torch.tensor(self.surface)), m),
        )
        self.bf["road_edge"] += torch.tensor([(p & self.dilate(t)).sum(), p.sum(), (t & self.dilate(p)).sum(), t.sum()])
        if pk is not None:
            k = pk > 0
            self.known += torch.tensor([(k & m).sum(), (k | m).sum()])

    def result(self) -> dict:
        iou = {CLASSES[c]: round(float(self.inter[c] / self.union[c].clamp(min=1)), 3) for c in range(1, NC - 1)}
        bf = {}
        for k, (tp, n_p, tl, n_l) in self.bf.items():
            p, r = float(tp / max(n_p, 1)), float(tl / max(n_l, 1))
            bf[k] = round(2 * p * r / max(p + r, 1e-9), 3)
        return dict(iou=iou, miou=round(float(np.mean(list(iou.values()))), 4), boundary_f=bf,
                    known_iou=round(float(self.known[0] / self.known[1].clamp(min=1)), 3))  # fmt: skip


@torch.no_grad()
def evaluate(net, ds, n: int | None = None, known: bool = False, by_level: float | None = None) -> dict:
    """Scores of net on (up to n evenly spaced samples of) ds. by_level: also score separately (key `by_level`) the
    multilevel samples (share of the crop over / under another level >= by_level, loader.MULTILEVEL), the single-level
    ones and the labelled cells under an overhead level (bit 1 of bev_level) or above a lower one (bit 2); needs the
    scenes' levels.npz (samples without it are counted in `no_levels`)."""
    was = net.training
    net.eval()
    sc = Scores(ds.bev["res"])
    groups = LevelScores(ds.bev["res"], by_level) if by_level is not None else None
    herr = torch.zeros(2)  # sum |error|, cells (road surface height, if the model has the head)
    idx = np.arange(len(ds)) if not n or n >= len(ds) else np.linspace(0, len(ds) - 1, n).astype(int)
    for i in idx:
        x = collate([ds[i]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        y = x["bev_cls"][0]
        pr = upsample(predict(out)[0].cpu(), net.f)
        pk = upsample(predict(out, known, net.vis_depth < VIS_DEPTH)[0].cpu(), net.f)
        sc.add(pr, y, pk)
        if groups is not None:
            groups.add(pr, y, pk, x["bev_level"][0], float(x["multilevel"][0]))
        if "height" in net.aux:
            t = block_mean(x["bev_height"], net.fc)[0]
            ok = torch.isfinite(t)
            herr += torch.tensor([float((net.aux["height"][0].cpu()[ok] - t[ok]).abs().sum()), float(ok.sum())])
    net.train(was)
    r = sc.result()
    if herr[1] > 0:
        r["height_mae"] = round(float(herr[0] / herr[1]), 3)
    if groups is not None:
        r["by_level"] = groups.result()
    return r


class LevelScores:
    """Scores split by road levels: per sample (multilevel / single_level by the crop share `thr`) and per cell
    (overhead: labelled cells with another level above; above_lower: with another level below)."""

    def __init__(self, res: float, thr: float):
        self.thr, self.no_levels = thr, 0
        self.s = {k: Scores(res) for k in ("multilevel", "single_level", "overhead", "above_lower")}
        self.n = dict.fromkeys(self.s, 0)

    def add(self, pr, y, pk, level, share: float):
        if not np.isfinite(share):
            self.no_levels += 1
            return
        k = "multilevel" if share >= self.thr else "single_level"
        self.s[k].add(pr, y, pk)
        self.n[k] += 1
        for k, bit in (("overhead", 1), ("above_lower", 2)):
            m = (level & bit) > 0
            if (m & (y > 0)).any():  # only those cells are labelled: the rest is not scored
                z = torch.zeros_like(y)
                self.s[k].add(pr, torch.where(m, y, z), torch.where(m, pk, z.to(pk.dtype)))
                self.n[k] += 1

    def result(self) -> dict:
        out = {k: dict(self.s[k].result(), samples=self.n[k]) for k in self.s if self.n[k]}
        return dict(out, threshold=self.thr, no_levels=self.no_levels)
