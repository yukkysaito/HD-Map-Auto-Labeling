"""Losses on the labelled cells (label 0 = unobserved is ignored)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hdlabel.common import CLASSES

SURFACE = ["road", "marking", "crosswalk"]  # the drivable surface; its edge is the road boundary


def lovasz_softmax(logits, y, classes):
    """Lovasz-softmax (Berman et al., 2018): a surrogate of the per-class IoU. logits [B, NC, X, Y], y [B, X, Y]."""
    p = logits.float().softmax(1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    y = y.reshape(-1)
    p, y = p[y > 0], y[y > 0]
    losses = []
    for c in classes:
        fg = (y == c).float()
        if fg.sum() == 0:
            continue
        err, order = torch.sort((fg - p[:, c]).abs(), descending=True)
        fg = fg[order]
        inter = fg.sum() - fg.cumsum(0)
        union = fg.sum() + (1 - fg).cumsum(0)
        jac = 1 - inter / union
        jac[1:] = jac[1:] - jac[:-1]
        losses.append(torch.dot(err, jac))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def class_weights(counts: torch.Tensor) -> torch.Tensor:
    """CE weights from label counts per class: inverse square root of the frequency, mean 1, 0 for unobserved."""
    w = (counts.sum() / counts.clamp(min=1)).sqrt()
    w[0] = 0
    return w / w[1:].mean()


def boost_weights(w: torch.Tensor, boost: dict[str, float]) -> torch.Tensor:
    """class weights w multiplied by boost {class name: factor} (e.g. marking=2, sidewalk=0.5)."""
    w = w.clone()
    for name, f in boost.items():
        w[CLASSES.index(name)] *= f
    return w


def surface_map(y):
    """[B, X, Y] 1 on the drivable surface (SURFACE classes), 2 on the other labelled cells, 0 unlabelled."""
    s = torch.isin(y, torch.tensor([CLASSES.index(c) for c in SURFACE], device=y.device))
    return torch.where(y > 0, 2 - s.long(), 0)


def surface_nll(out, y):
    """[B, X, Y] -log P(drivable surface) on SURFACE cells, -log P(not) on the other labelled cells, 0 unlabelled; the
    probabilities are summed over the classes (channels 1..), so confusions inside either side cost nothing."""
    lp = out[:, 1:].float().log_softmax(1)
    idx = torch.tensor([CLASSES.index(c) - 1 for c in SURFACE], device=out.device)
    on = torch.zeros(lp.shape[1], dtype=torch.bool, device=out.device)
    on[idx] = True
    s = surface_map(y)
    nll = -torch.where(s == 1, lp[:, on].logsumexp(1), lp[:, ~on].logsumexp(1))
    return torch.where(s > 0, nll, 0)


def near_boundary(y, r: int):
    """[B, X, Y] bool: labelled cells within r cells of a cell of another labelled class."""
    yf = y.float()[:, None]
    hi = F.max_pool2d(yf, 2 * r + 1, 1, r)
    lo = -F.max_pool2d(torch.where(yf > 0, -yf, -torch.inf), 2 * r + 1, 1, r)  # min over the labelled cells
    return ((hi > lo) & (lo > 0))[:, 0] & (y > 0)


def block_mean(v: torch.Tensor, f: int) -> torch.Tensor:
    """[B, X, Y] values (NaN = none) -> [B, X/f, Y/f] mean of the finite values per f x f block (NaN if none)."""
    b, h, w = v.shape
    v = v.reshape(b, h // f, f, w // f, f)
    ok = torch.isfinite(v)
    n = ok.sum((2, 4))
    return torch.where(n > 0, torch.where(ok, v, 0).sum((2, 4)) / n.clamp(min=1), torch.nan)


def height_loss(pred: torch.Tensor, height: torch.Tensor, f: int) -> torch.Tensor:
    """smooth L1 (m) of the predicted road surface height [B, X, Y] against the label height [B, X*f, Y*f] (NaN = no
    surface), over the cells that have one."""
    y = block_mean(height, f)
    ok = torch.isfinite(y)
    if not ok.any():
        return pred.sum() * 0
    return F.smooth_l1_loss(pred[ok], y[ok], beta=0.1)


def seg_loss(out, y, weight, lovasz=0.0, known=0.0, edge=0.0, edge_r=2, road_edge=0.0, road_edge_r=3, surface=0.0):
    """cross entropy (+ Lovasz) over the classes; with known > 0 channel 0 learns "cell unlabelled" (BCE). edge > 0
    weights the cross entropy of the cells near a class boundary (1 + edge): thin lines and edges are few cells.
    road_edge > 0 adds road_edge to that weight within road_edge_r cells of the edge of the drivable surface (SURFACE vs
    the other labelled classes). surface > 0 adds the drivable-or-not loss (surface_nll; mean over the labelled cells,
    with the same edge weights)."""
    w = None
    if edge or road_edge:
        w = torch.ones_like(y, dtype=torch.float)
        if edge:
            w = w + edge * near_boundary(y, edge_r).float()
        if road_edge:
            w = w + road_edge * near_boundary(surface_map(y), road_edge_r).float()
    if w is not None:
        ce = F.cross_entropy(out.float(), y, weight=weight, ignore_index=0, reduction="none")
        lab = (y > 0).float()
        loss = (ce * w).sum() / (weight[y] * w * lab).sum().clamp(min=1)
    else:
        loss = F.cross_entropy(out.float(), y, weight=weight, ignore_index=0)
    if lovasz:
        loss = loss + lovasz * lovasz_softmax(out, y, range(1, out.shape[1]))
    if surface:
        wl = (y > 0).float() * (w if w is not None else 1)
        loss = loss + surface * (surface_nll(out, y) * wl).sum() / wl.sum().clamp(min=1)
    if known:
        loss = loss + known * F.binary_cross_entropy_with_logits(out[:, 0].float(), (y == 0).float())
    return loss
