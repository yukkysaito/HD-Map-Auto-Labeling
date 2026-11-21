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


def seg_loss(out, y, weight, lovasz=0.0, known=0.0):
    """cross entropy (+ Lovasz) over the classes; with known > 0 channel 0 learns "cell unlabelled" (BCE)."""
    loss = F.cross_entropy(out.float(), y, weight=weight, ignore_index=0)
    if lovasz:
        loss = loss + lovasz * lovasz_softmax(out, y, range(1, out.shape[1]))
    if known:
        loss = loss + known * F.binary_cross_entropy_with_logits(out[:, 0].float(), (y == 0).float())
    return loss
