"""Pixel ray / road surface intersection (hdlabel/mapproj.Projector) on a synthetic sloped surface (GPU)."""

import numpy as np
import pytest
import torch

from hdlabel.mapproj import Projector

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ray(pj, C, target):
    C = torch.tensor(C, device=pj.dev, dtype=torch.float64)
    d = torch.tensor((np.array(target) - C.cpu().numpy())[None], device=pj.dev, dtype=torch.float64)
    P, conv, t = pj.intersect(C, d / d.norm())
    return P[0].cpu().numpy(), bool(conv[0] and t[0] > 0)  # what project() keeps


def test_ray_hits_sloped_surface(tmp_path):
    H = np.fromfunction(lambda i, j: 0.1 * ((i + 0.5) * 0.2 - 50), (500, 500)).astype(np.float32)  # 10 % uphill along x
    np.savez(
        tmp_path / "g.npz",
        height=H,
        height_w=np.ones_like(H),
        lo=np.array([-50.0, -50.0]),
        surf_res=0.2,
        origin=np.zeros(3),
    )
    pj = Projector(str(tmp_path / "g.npz"))
    for target in ([8.0, 1.0, 0.8], [-10.0, 4.0, -1.0], [3.0, -2.0, 0.3]):
        P, conv = ray(pj, [0.0, 0.0, 3.0], target)
        assert conv and np.allclose(P, target, atol=0.02)
    # grazing ray up the slope: no fixed point ahead; it must be flagged (not converged or behind) or be right
    P, conv = ray(pj, [0.0, 0.0, 1.5], [40.0, 0.0, 4.0 - 0.3])
    assert not conv or np.allclose(P, [40.0, 0.0, 4.0], atol=0.3)


def test_gather_fills_what_splat_leaves_holes_in(tmp_path):
    """a camera looking down a flat road whose image is all 'road' (segmentation): gather gives every cell it sees a
    vote, splat (pixel stride 2) leaves far cells empty; where both vote they agree."""
    from hdlabel.common import GROUPS

    H = np.zeros((300, 300), np.float32)
    np.savez(tmp_path / "g.npz", height=H, height_w=np.ones_like(H), lo=np.array([-30.0, -30.0]), surf_res=0.2,
             origin=np.zeros(3))  # fmt: skip
    pj = Projector(str(tmp_path / "g.npz"), device="cpu")
    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # optical axis along +x, image down = -z
    T[:3, 3] = [0, 0, 2.0]
    seg = np.full((480, 640), GROUPS["road"][0], np.uint8)
    accs = {}
    for mode in ("splat", "gather"):
        acc = pj.new_acc()
        if mode == "gather":
            pj.gather(T, K, seg, None, acc, rmax=25.0)
        else:
            pj.project(T, K, seg, None, acc, stride=2, rmax=25.0)
        accs[mode] = pj.dense(acc["votes"].sum(1))
    seen_g, seen_s = accs["gather"] > 0, accs["splat"] > 0
    assert seen_s.sum() > 1000 and (seen_s & ~seen_g).sum() < 0.01 * seen_s.sum()
    assert seen_g.sum() > 1.2 * seen_s.sum()  # the far cells between splatted pixels
