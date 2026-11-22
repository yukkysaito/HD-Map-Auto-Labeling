"""Losses on the labelled cells (label 0 = unobserved is ignored)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def seg_loss(out, y, weight, lovasz=0.0, known=0.0, edge=0.0, edge_r=2):
    """cross entropy (+ Lovasz) over the classes; with known > 0 channel 0 learns "cell unlabelled" (BCE). edge > 0
    weights the cross entropy of the cells near a class boundary (1 + edge): thin lines and edges are few cells."""
    if edge:
        ce = F.cross_entropy(out.float(), y, weight=weight, ignore_index=0, reduction="none")
        w = 1 + edge * near_boundary(y, edge_r).float()
        lab = (y > 0).float()
        loss = (ce * w).sum() / (weight[y] * w * lab).sum().clamp(min=1)
    else:
        loss = F.cross_entropy(out.float(), y, weight=weight, ignore_index=0)
    if lovasz:
        loss = loss + lovasz * lovasz_softmax(out, y, range(1, out.shape[1]))
    if known:
        loss = loss + known * F.binary_cross_entropy_with_logits(out[:, 0].float(), (y == 0).float())
    return loss
