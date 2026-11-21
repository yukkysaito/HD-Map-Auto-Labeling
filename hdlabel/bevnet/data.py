"""Training data: the current frame of every camera and the BEV label raster, with BEV augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from hdlabel.common import CLASSES
from hdlabel.loader import HDMapDataset

NC = len(CLASSES)


class BEVDataset(HDMapDataset):
    """HDMapDataset (current frame only) with random BEV rotation (+-rot_deg about z) and mirror (y -> -y).

    The labels are cropped in a virtual ego frame; the model gets A (virtual frame -> base_link) and samples the
    images there, so images and poses stay untouched."""

    def __init__(self, *args, rot_deg: float = 0.0, flip: bool = False, **kw):
        kw.setdefault("history", 1)
        super().__init__(*args, fast_decode=True, uint8=True, **kw)
        self.rot, self.flip = rot_deg, flip

    def __getitem__(self, i):
        x = super().__getitem__(i)
        A = np.eye(4)
        if self.rot or self.flip:
            a = np.deg2rad(np.random.uniform(-self.rot, self.rot))
            A[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            if self.flip and np.random.rand() < 0.5:
                A = A @ np.diag([1.0, -1.0, 1.0, 1.0])
            sid, r = self.items[i]
            T = np.array(r["T_map_base"]) @ A
            cls, _ = self.scene_map(sid).crop(T, self.bev["x"], self.bev["y"], self.bev["res"])
            x["bev_cls"] = torch.from_numpy(cls.astype(np.int64))
        x["A"] = torch.from_numpy(A.astype(np.float32))
        return x


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([e[k] for e in batch]) for k in batch[0] if torch.is_tensor(batch[0][k])}


def downsample(cls: torch.Tensor, f: int) -> torch.Tensor:
    """label rasters [B, X, Y] to f x f blocks: the majority of the observed classes (0 if none is observed)."""
    if f == 1:
        return cls
    b, h, w = cls.shape
    c = cls.reshape(b, h // f, f, w // f, f).permute(0, 1, 3, 2, 4).reshape(b, h // f, w // f, f * f)
    oh = F.one_hot(c, NC)[..., 1:].sum(-2)
    return torch.where(oh.sum(-1) > 0, oh.argmax(-1) + 1, torch.zeros_like(oh[..., 0]))
