"""Other road levels (hdlabel/levels.py), the multilevel sample tag (loader) and the per-level scores (bevnet.metrics)
on the synthetic two-level map of test_surface: a ground road and a deck 5 m up along y (x in -8..8)."""

import numpy as np
import torch
from test_surface import synthetic_map

from hdlabel.bev import SceneMap
from hdlabel.levels import RES, Levels, other_levels

LO, SRES = np.array([-50.0, -50.0]), 0.2


def tree_crowns(rng, n=12):
    """deep, irregular point blobs 2.5..9 m up (street tree crowns) at x = 30."""
    pts = []
    for y in np.linspace(-40, 40, n):
        c = rng.normal(0, 1, (3000, 3)) * [2.5, 2.5, 1.8] + [30, y, 5.5]
        pts.append(c[c[:, 2] > 2.5])
    return np.concatenate(pts)


def cell(M, x, y):
    return M[int((x - LO[0]) / RES), int((y - LO[1]) / RES)]


def test_deck_above_the_ground_road_and_ground_below_the_deck():
    rng = np.random.default_rng(0)
    P = np.concatenate([synthetic_map(), tree_crowns(rng)])
    H = np.zeros((500, 500))  # ego level = the ground road everywhere
    over, under = other_levels(P, H, LO, SRES)
    assert abs(cell(over, 0, 0) - 5.0) < 0.2 and abs(cell(over, 0, 30) - 5.0) < 0.2  # the deck above
    assert np.isnan(cell(over, 20, 0)) and np.isnan(cell(over, 30, 0))  # open sky / tree crowns are no level
    assert np.isnan(under).all()
    Hd = np.full((500, 500), np.nan)  # ego level = the deck (x in -8..8)
    Hd[int((-8 + 50) / SRES) : int((8 + 50) / SRES)] = 5.0
    over, under = other_levels(P, Hd, LO, SRES)
    assert abs(cell(under, 0, 0) + 5.0) < 0.2  # the ground road below the deck
    assert np.isnan(cell(over, 0, 0))


def test_levels_crop_and_the_multilevel_tag(tmp_path):
    from hdlabel.loader import HDMapDataset

    over = np.full((100, 100), np.nan, np.float16)
    over[45:55] = 5.0  # a level above along y at map x in -5..5
    under = np.full((100, 100), np.nan, np.float16)
    under[:, 0:10] = -6.0
    np.savez(tmp_path / "levels.npz", over=over, under=under, lo=LO, res=RES, origin=np.zeros(3))
    np.savez(tmp_path / "map.npz", cls=np.zeros((1000, 1000), np.uint8), height=np.zeros((500, 500), np.float32),
             lo=LO, res=0.1, surf_res=SRES, origin=np.zeros(3))  # fmt: skip
    lv, sm = Levels(str(tmp_path / "levels.npz")), SceneMap(str(tmp_path / "map.npz"))
    T = np.eye(4)
    T[0, 3] = -20.0  # ego 20 m before the level above, looking along +x
    m = lv.crop(sm, T, (-10, 30), (-5, 5), 0.5)
    assert m.shape == (80, 20)
    ahead = lambda x: m[int((30 - x) / 0.5), 10]
    assert ahead(20.0) == 1 and ahead(0.0) == 0 and ahead(-5.0) == 0
    T[1, 3] = -45.0  # now beside the level below too
    assert (lv.crop(sm, T, (-10, 30), (-5, 5), 0.5) & 2).any()

    ds = HDMapDataset.__new__(HDMapDataset)  # only what level_crop uses
    ds.bev, ds._levels, ds._maps = dict(x=[-10, 30], y=[-5, 5], res=0.5), {}, {"a": sm}
    ds.scenes = {"a": dict(dir=str(tmp_path)), "b": dict(dir=str(tmp_path / "none"))}
    T[1, 3] = 0.0
    r = ds.level_crop("a", T)
    assert r["bev_level"].shape == (80, 20) and abs(float(r["multilevel"]) - 10 / 40) < 0.03
    r = ds.level_crop("b", T)  # no levels.npz: no levels known
    assert r["bev_level"].shape == (80, 20) and not r["bev_level"].any() and torch.isnan(r["multilevel"])


def test_level_scores_split_samples_and_cells():
    from hdlabel.bevnet.metrics import LevelScores

    y = torch.ones(20, 10, dtype=torch.long)  # all road
    pr = y.clone()
    pr[:10] = 5  # wrong (sidewalk) in the first half
    level = torch.zeros(20, 10, dtype=torch.uint8)
    level[:10] = 1  # ... which lies under an overpass
    g = LevelScores(0.1, 0.05)
    g.add(pr, y, pr, level, 0.5)
    g.add(y, y, y, torch.zeros_like(level), 0.0)
    g.add(y, y, y, level, float("nan"))  # scene without levels.npz
    r = g.result()
    assert r["multilevel"]["samples"] == 1 and r["single_level"]["samples"] == 1 and r["no_levels"] == 1
    assert r["single_level"]["iou"]["road"] == 1.0 and r["multilevel"]["iou"]["road"] == 0.5
    assert r["overhead"]["iou"]["road"] == 0.0 and "above_lower" not in r


def test_evaluate_reports_scores_by_level():
    from test_bevnet import BEV, SIZE, batch

    from hdlabel.bevnet.metrics import evaluate
    from hdlabel.bevnet.model import BEVSegNet

    class DS(list):
        bev = BEV

    torch.manual_seed(0)
    net = BEVSegNet(BEV, 1, SIZE, cell=0.4, out_res=0.1, heights=[0.0, 1.0], vit=None)
    items = []
    for share in (0.3, 0.0):
        x = {k: v[0] for k, v in batch().items()}
        x["bev_level"] = torch.zeros(160, 64, dtype=torch.uint8)
        x["bev_level"][:40] = 1 if share else 0
        items.append(dict(x, multilevel=torch.tensor(share)))
    r = evaluate(net, DS(items), by_level=0.05)
    assert r["by_level"]["multilevel"]["samples"] == 1 and r["by_level"]["single_level"]["samples"] == 1
    assert r["by_level"]["overhead"]["samples"] == 1 and "miou" in r["by_level"]["overhead"]
    assert "by_level" not in evaluate(net, DS(items))
