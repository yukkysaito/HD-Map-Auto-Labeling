"""hdlabel.bevnet on CPU: projection geometry, output shapes and a training step of a tiny network."""

import numpy as np
import torch

from hdlabel.bevnet.data import NC, downsample
from hdlabel.bevnet.geometry import grid, project
from hdlabel.bevnet.losses import near_boundary, seg_loss
from hdlabel.bevnet.metrics import Scores, predict
from hdlabel.bevnet.model import BEVSegNet

BEV = dict(x=[-4.0, 12.0], y=[-3.2, 3.2], res=0.1)
SIZE = (64, 48)


def camera():
    """one camera 2 m above base_link looking forward (optical axis +x, image down = -z)."""
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    T[2, 3] = 2.0
    K = torch.tensor([[40.0, 0, 32], [0, 40.0, 24], [0, 0, 1]])
    return T, K


def batch():
    T, K = camera()
    return dict(
        images=torch.randint(0, 255, (1, 1, 1, 3, SIZE[1], SIZE[0]), dtype=torch.uint8),
        T_cur_frame=torch.eye(4).expand(1, 1, 1, 4, 4).clone(),
        T_base_cam=T[None, None],
        K=K[None, None],
        cam_mask=torch.ones(1, 1, dtype=torch.bool),
        bev_cls=torch.randint(0, NC - 1, (1, 160, 64)),
    )


def test_grid_orientation():
    g = grid(BEV, 0.4)
    assert g.shape == (40, 16, 4)
    assert g[0, 0, 0] > g[-1, 0, 0] and g[0, 0, 1] > g[0, -1, 1]  # row 0 = far, column 0 = left


def test_project_point_ahead_hits_principal_row():
    T, K = camera()
    P = torch.tensor([[[10.0, 0.0, 2.0, 1.0], [10.0, 0.0, 0.0, 1.0], [-5.0, 0.0, 0.0, 1.0]]])
    g, z, ok = project(
        P, torch.eye(4)[None, None, None], T[None, None], K[None, None], torch.ones(1, 1, dtype=bool), SIZE
    )
    assert np.allclose(g[0, 0, 0, 0].numpy(), [0, 0], atol=1e-5)  # at camera height, straight ahead: image centre
    assert g[0, 0, 0, 1, 1] > 0 and ok[0, 0, 0, :2].all()  # on the ground ahead: below the centre
    assert not ok[0, 0, 0, 2]  # behind the camera
    assert np.isclose(z[0, 0, 0, 0].item(), 10.0)


def test_tiny_network_trains_one_step():
    torch.manual_seed(0)
    net = BEVSegNet(BEV, 1, SIZE, cell=0.4, out_res=0.1, heights=[0.0, 1.0], vit=None)
    x = batch()
    out = net(x)
    assert out.shape == (1, NC, 160, 64) and net.ctx_logits.shape == (1, NC, 40, 16)
    y = downsample(x["bev_cls"], net.f)
    loss = seg_loss(out, y, torch.ones(NC), lovasz=0.5, known=0.5)
    loss.backward()
    assert torch.isfinite(loss) and net.bev.inp[0][0].weight.grad.abs().sum() > 0
    sc = Scores(BEV["res"])
    sc.add(predict(out.detach())[0], x["bev_cls"][0])
    r = sc.result()
    assert 0 <= r["miou"] <= 1 and set(r["boundary_f"]) == {"marking", "crosswalk", "curb"}


def test_downsample_majority_ignores_unobserved():
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    y[0, :2, :2] = torch.tensor([[2, 2], [0, 3]])
    assert downsample(y, 2)[0].tolist() == [[2, 0], [0, 0]]


def test_near_boundary_marks_both_sides_of_a_class_edge_only():
    y = torch.zeros(1, 6, 6, dtype=torch.long)
    y[0, :, :3], y[0, :, 3:], y[0, 0] = 1, 2, 0
    b = near_boundary(y, 1)[0]
    assert b[1:, 2:4].all() and b.sum() == 10  # the unlabelled first row is no class edge
    out, w = torch.randn(1, NC, 6, 6), torch.ones(NC)
    w[0] = 0
    assert torch.isclose(seg_loss(out, y, w), seg_loss(out, y, w, edge=1e-9))  # same normalisation as plain CE


def test_temporal_fusion_trains_on_past_frames():
    torch.manual_seed(0)
    net = BEVSegNet(BEV, 1, SIZE, cell=0.4, out_res=0.1, heights=[0.0, 1.0], vit=None, history=2)
    x = batch()
    x["images"] = x["images"].repeat(1, 2, 1, 1, 1, 1)
    back = torch.eye(4)
    back[0, 3] = -1.0  # the earlier frame was taken 1 m behind
    x["T_cur_frame"] = torch.stack([torch.eye(4), back])[None, :, None]
    out = net(x)
    seg_loss(out, downsample(x["bev_cls"], net.f), torch.ones(NC)).backward()
    assert out.shape == (1, NC, 160, 64) and net.gru.zr.weight.grad.abs().sum() > 0


def test_height_head_is_an_auxiliary_task_on_the_shared_bev():
    from hdlabel.bevnet.losses import block_mean, height_loss

    torch.manual_seed(0)
    net = BEVSegNet(BEV, 1, SIZE, cell=0.4, out_res=0.1, heights=[0.0, 1.0], vit=None, height_head=True)
    x = batch()
    h = torch.full((1, 160, 64), float("nan"))
    h[:, :80] = 0.3  # a surface on half of the area, no surface (NaN) elsewhere
    out = net(x)
    assert out.shape == (1, NC, 160, 64) and net.aux["height"].shape == (1, 40, 16)
    assert torch.isnan(block_mean(h, 4)[0, 20:]).all() and torch.allclose(block_mean(h, 4)[0, :20], torch.tensor(0.3))
    loss = height_loss(net.aux["height"], h, net.fc)
    loss.backward()
    assert torch.isfinite(loss) and net.bev.inp[0][0].weight.grad.abs().sum() > 0  # reaches the shared BEV backbone
    assert BEVSegNet(BEV, 1, SIZE, heights=[0.0], vit=None).height_head is None  # off by default (map only)
