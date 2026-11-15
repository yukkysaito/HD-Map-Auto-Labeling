"""Ego-frame labels from map-frame scene labels (hdlabel/bev.py): pose, axes, clipping, slopes."""

import json

import numpy as np

from hdlabel.bev import SceneMap, grid, resample


def make_scene(tmp_path, slope=0.0):
    cls = np.zeros((400, 400), np.uint8)
    cls[:, :] = 1  # 40 x 40 m at 0.1 m, origin (1000, 2000, 5)
    cls[300:310, :] = 2  # a marking band at map x = 1000 - 20 + 30..31 m
    H = np.fromfunction(lambda i, j: slope * ((i + 0.5) * 0.2 - 20), (200, 200)).astype(np.float32)
    np.savez(
        tmp_path / "map.npz",
        cls=cls,
        height=H,
        lo=np.array([-20.0, -20.0]),
        res=0.1,
        surf_res=0.2,
        origin=np.array([1000.0, 2000.0, 5.0]),
    )
    feats = [
        dict(
            type="Feature",
            geometry=dict(type="LineString", coordinates=[[990, 1995, 5], [1010, 1995, 5]]),
            properties=dict(kind="lane_line", style="solid", segments=[[0, 20]]),
        ),
        dict(
            type="Feature",
            geometry=dict(type="LineString", coordinates=[[1000, 2005, 5], [1003, 2005, 5]]),
            properties=dict(
                kind="arrow", directions=["right"], outlines=[[[1000, 2004.8, 5], [1003, 2004.8, 5], [1003, 2005.2, 5]]]
            ),
        ),
    ]
    json.dump(dict(type="FeatureCollection", features=feats), open(tmp_path / "v.geojson", "w"))
    return SceneMap(str(tmp_path / "map.npz"), str(tmp_path / "v.geojson"))


def pose(x, y, yaw, z=5.0):
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:3, 3] = [x, y, z]
    return T


def test_raster_axes(tmp_path):
    sm = make_scene(tmp_path)
    # ego at map (1000, 2000) facing +y (yaw 90 deg): the marking band (map x 1010..1011) is on the ego's RIGHT (y = -10..-11)
    c, h = sm.crop(pose(1000, 2000, np.pi / 2), x=(-15, 15), y=(-15, 15), res=0.5)
    Xg, Yg = grid((-15, 15), (-15, 15), 0.5)
    band = c == 2
    assert band.any() and np.all((Yg[band] <= -9.5) & (Yg[band] >= -11.5))
    assert np.nanmax(np.abs(h)) < 1e-3  # flat road at base_link height
    assert Xg[0, 0] > Xg[-1, 0] and Yg[0, 0] > Yg[0, -1]  # row 0 = far, col 0 = left


def test_vectors_transform_and_clip(tmp_path):
    sm = make_scene(tmp_path)
    els = sm.vectors(pose(1000, 2000, 0.0), x=(-5, 5), y=(-10, 10))
    ll = [e for e in els if e["kind"] == "lane_line"][0]
    P = np.array(ll["points"])
    assert (
        np.allclose(P[:, 1], -5, atol=1e-6)
        and P[:, 0].min() >= -5
        and P[:, 0].max() <= 5
        and abs(ll["length"] - 10) < 0.2
    )
    ar = [e for e in els if e["kind"] == "arrow"][0]
    assert (
        ar["directions"] == ["right"]
        and np.allclose(ar["points"][0][:2], [0, 5])
        and np.allclose(ar["points"][-1][:2], [3, 5])
    )
    assert len(ar["outlines"]) == 1


def test_slope_height(tmp_path):
    sm = make_scene(tmp_path, slope=0.05)  # 5 % up along map x
    c, h = sm.crop(pose(1000, 2000, 0.0, z=5.0), x=(0, 10), y=(-1, 1), res=1.0)
    assert np.allclose(h[0].mean(), 0.05 * 9.5, atol=0.02)  # 9.5 m ahead: 0.475 m up


def test_resample():
    P = resample([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 5)
    assert np.allclose(P[2], [10, 0, 0]) and len(P) == 5
