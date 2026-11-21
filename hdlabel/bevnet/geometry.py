"""BEV grids and camera projection (all in the current base_link frame, metres)."""

from __future__ import annotations

import numpy as np
import torch


def grid(bev: dict, cell: float) -> torch.Tensor:
    """cell centres [X, Y, 4] (homogeneous, z = 0) of the BEV range bev = {x: [lo, hi], y: [lo, hi]} at `cell` m.
    Row 0 = far (x max), column 0 = left (y max), as bev.SceneMap.crop."""
    xs = np.arange(bev["x"][1] - cell / 2, bev["x"][0], -cell)
    ys = np.arange(bev["y"][1] - cell / 2, bev["y"][0], -cell)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return torch.tensor(np.stack([gx, gy, np.zeros_like(gx), np.ones_like(gx)], -1), dtype=torch.float32)


def project(P, T_cur_frame, T_base_cam, K, cam_mask, size):
    """points P [B, N, 4] -> grid_sample coordinates [B, C, T, N, 2], depth [B, C, T, N] and in-view mask.

    T_cur_frame [B, T, C, 4, 4] (frame base_link -> current base_link), T_base_cam [B, C, 4, 4], K [B, C, 3, 3] for
    images of size = (width, height), cam_mask [B, C]."""
    w, h = size
    T = torch.linalg.inv(T_cur_frame @ T_base_cam[:, None])  # current base_link -> camera at frame t
    q = torch.einsum("btcij,bnj->bctni", T, P)[..., :3]
    z = q[..., 2]
    uv = torch.einsum("bcij,bctnj->bctni", K, q)[..., :2] / z.clamp(min=0.1)[..., None]
    g = torch.stack([uv[..., 0] / w * 2 - 1, uv[..., 1] / h * 2 - 1], -1)
    ok = (z > 0.5) & (g.abs() <= 1).all(-1) & cam_mask[:, :, None, None]
    return g, z, ok


def lift(points, heights, A):
    """grid points [X, Y, 4] at each height -> [B, len(heights) * X * Y, 4] in base_link (A [B, 4, 4]: virtual ego
    frame of the augmentation -> base_link)."""
    P = points.flatten(0, 1)
    P = torch.cat([P + torch.tensor([0.0, 0.0, z, 0.0], device=P.device) for z in heights])
    return torch.einsum("bij,nj->bni", A, P)
