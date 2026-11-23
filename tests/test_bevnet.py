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
    loss = seg_loss(out, y, torch.ones(NC), lovasz=0.5, known=0.5, edge=1.0, road_edge=1.0, surface=0.5)
    loss.backward()
    assert torch.isfinite(loss) and net.bev.inp[0][0].weight.grad.abs().sum() > 0
    sc = Scores(BEV["res"])
    sc.add(predict(out.detach())[0], x["bev_cls"][0])
    r = sc.result()
    assert 0 <= r["miou"] <= 1 and set(r["boundary_f"]) == {"marking", "crosswalk", "curb", "road_edge"}


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


def test_priority_options_are_off_by_default_and_steer_the_loss():
    from hdlabel.bevnet.losses import SURFACE, boost_weights, surface_map, surface_nll
    from hdlabel.bevnet.train import parse_boost
    from hdlabel.common import CLASSES

    torch.manual_seed(0)
    road, side, mark = CLASSES.index("road"), CLASSES.index("sidewalk"), CLASSES.index("marking")
    y = torch.full((1, 8, 8), road)
    y[0, :, 5:], y[0, :, 2], y[0, 0] = side, mark, 0  # road | marking line | road | sidewalk; first row unlabelled
    out, w = torch.randn(1, NC, 8, 8, requires_grad=True), torch.rand(NC) + 0.5
    w[0] = 0
    plain = seg_loss(out, y, w, lovasz=0.5, known=0.5)
    off = seg_loss(out, y, w, lovasz=0.5, known=0.5, road_edge=0.0, surface=0.0)
    assert torch.equal(plain, off) and torch.equal(boost_weights(w, {}), w)  # bit-identical when off
    # the road edge band: road / marking cells next to the sidewalk, not the marking line inside the road
    s = surface_map(y)[0]
    assert s[0].eq(0).all() and s[1:, :5].eq(1).all() and s[1:, 5:].eq(2).all()
    band = near_boundary(surface_map(y), 1)[0]
    assert band[1:, 4:6].all() and band.sum() == 14
    assert not torch.isclose(seg_loss(out, y, w, road_edge=2.0), seg_loss(out, y, w))
    # class boost: only the named class weight changes
    b = boost_weights(w, parse_boost(["marking=2", "sidewalk=0.5"]))
    assert torch.isclose(b[mark], 2 * w[mark]) and torch.isclose(b[side], 0.5 * w[side]) and b[road] == w[road]
    # surface loss: confusions inside the drivable surface cost nothing
    good = torch.full((1, NC, 8, 8), -10.0)
    good[:, road] = torch.where(s > 1, -10.0, 10.0)[None]  # all drivable cells say road, even the marking line
    good[:, side] = torch.where(s > 1, 10.0, -10.0)[None]
    assert surface_nll(good, y).max() < 1e-6 and set(SURFACE) <= set(CLASSES)
    loss = seg_loss(out, y, w, road_edge=1.0, surface=1.0)
    loss.backward()
    assert torch.isfinite(loss) and out.grad.abs().sum() > 0


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


def test_road_edge_boundary_f():
    import torch

    from hdlabel.bevnet.metrics import Scores

    y = torch.full((50, 50), 5)  # sidewalk
    y[:, 10:30] = 1  # road
    y[:, 20] = 2  # a lane line inside: no road edge
    near, far = y.clone(), y.clone()
    near[:, 30:32] = 1  # edge 0.2 m off: within the tolerance
    far[:, 30:36] = 1
    bf = []
    for pr in (y, near, far):
        sc = Scores(0.1)
        sc.add(pr, y)
        bf.append(sc.result()["boundary_f"]["road_edge"])
    assert bf[0] == bf[1] == 1.0 and bf[2] == 0.5


def test_level_heads_are_off_by_default_droppable_and_leave_the_map_unchanged(tmp_path):
    from hdlabel.bevnet.losses import level_dz_loss, level_loss
    from hdlabel.bevnet.train import load_checkpoint, save_checkpoint

    kw = dict(cell=0.4, out_res=0.1, heights=[0.0, 1.0], vit=None, height_head=True)
    x = batch()
    torch.manual_seed(0)
    base = BEVSegNet(BEV, 1, SIZE, **kw).eval()
    torch.manual_seed(0)
    net = BEVSegNet(BEV, 1, SIZE, level_head=True, height_levels=True, **kw).eval()
    with torch.no_grad():
        o0, o1 = base(x), net(x)
    assert torch.equal(o0, o1) and torch.equal(base.aux["height"], net.aux["height"])  # same map, same height
    assert set(base.aux) == {"height"} and set(net.aux) == {"height", "level", "level_dz"}
    assert net.aux["level"].shape == net.aux["level_dz"].shape == (1, 2, 40, 16)
    plain = BEVSegNet(BEV, 1, SIZE, heights=[0.0], vit=None)
    assert plain.level_head is None and plain.height_levels is None and not plain.config["level_head"]

    # losses: the level head learns where levels.npz exists (multilevel finite), dz only where a level is
    net.train()
    net(x)
    level = torch.zeros(1, 160, 64, dtype=torch.uint8)
    level[:, :40], level[:, 100:] = 1, 2
    dz = torch.full((1, 2, 160, 64), float("nan"))
    dz[:, 0, :40], dz[:, 1, 100:] = 5.0, -7.0
    lv = level_loss(net.aux["level"], level, torch.tensor([0.4]), net.fc)
    hz = level_dz_loss(net.aux["level_dz"], dz, net.fc)
    (lv + hz).backward()
    assert torch.isfinite(lv) and torch.isfinite(hz) and net.bev.inp[0][0].weight.grad.abs().sum() > 0
    assert net.level_head[1].weight.grad.abs().sum() > 0 and net.height_levels.weight.grad.abs().sum() > 0
    assert level_loss(net.aux["level"], level, torch.tensor([float("nan")]), net.fc) == 0  # no levels.npz
    assert level_dz_loss(net.aux["level_dz"], torch.full_like(dz, float("nan")), net.fc) == 0

    # checkpoints: with the heads, dropped to map only (same map output), and from before the heads existed
    net.eval()
    with torch.no_grad():
        o1 = net(x)  # the training forward updated the batch norm statistics
    save_checkpoint(net, tmp_path / "heads.pt")
    loaded, _ = load_checkpoint(tmp_path / "heads.pt")
    save_checkpoint(net.drop_aux(), tmp_path / "map.pt")
    assert net.aux == {} and not net.config["level_head"] and not net.config["height_head"]
    map_only, _ = load_checkpoint(tmp_path / "map.pt")
    with torch.no_grad():
        assert torch.equal(loaded(x), o1) and set(loaded.aux) == {"height", "level", "level_dz"}
        assert torch.equal(map_only(x), o1) and map_only.aux == {}
    old = {k: v for k, v in plain.config.items() if k not in ("level_head", "height_levels")}
    torch.save(dict(config=old, state=plain.state_dict()), tmp_path / "old.pt")
    assert load_checkpoint(tmp_path / "old.pt")[0].level_head is None
