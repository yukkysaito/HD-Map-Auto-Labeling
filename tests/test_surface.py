"""Multi-level road surface (hdlabel/surface.py) on a synthetic map: a ground road, an elevated road crossing over it
(deck 5 m up) and a ramp joining them far away. The surface must follow the level of the trajectory."""
import numpy as np

from hdlabel.surface import Grid, build


def synthetic_map(seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    def plane(x0, x1, y0, y1, z, n_per_m2=40):
        n = int((x1 - x0) * (y1 - y0) * n_per_m2)
        x = rng.uniform(x0, x1, n); y = rng.uniform(y0, y1, n); pts.append(np.c_[x, y, z(x, y) + rng.normal(0, 0.02, n)])
    plane(-60, 60, -60, 60, lambda x, y: 0 * x)                     # ground everywhere
    plane(-8, 8, -60, 60, lambda x, y: 5.0 + 0 * x)                 # deck: elevated road along y (x in -8..8), 5 m up
    plane(8, 60, 50, 58, lambda x, y: np.clip((60 - x) / 52 * 5.0, 0, 5))   # ramp at y 50..58: deck (x=8) down to ground (x=60)
    for x in (-8.0, 8.0):                                           # deck edges: parapet walls (vertical points)
        y = rng.uniform(-60, 60, 20000); pts.append(np.c_[np.full_like(y, x), y, rng.uniform(5, 6, 20000)])
    return np.concatenate(pts)


GRID = Grid(lo=(-50, -50), shape=(500, 500), res=0.2, origin=(0, 0, 0))
at = lambda S, x, y: S["height"][int((x + 50) / 0.2), int((y + 50) / 0.2)]


def test_ground_trajectory_under_deck():
    P = synthetic_map(); traj = np.c_[np.linspace(-45, 45, 200), np.zeros(200), np.zeros(200)]       # along x at y = 0, under the deck
    S = build(P, [traj], GRID, corridor=40.0)
    assert abs(at(S, 0, 0)) < 0.05, "under the deck the surface must be the ground road"
    assert abs(at(S, 0, 20)) < 0.05 and abs(at(S, 30, 0)) < 0.05
    assert S["info"]["multi_layer_cells"] > 1000


def test_deck_trajectory_excludes_road_below():
    P = synthetic_map(); traj = np.c_[np.zeros(200), np.linspace(-45, 45, 200), np.full(200, 5.0)]   # on the deck along y
    S = build(P, [traj], GRID, corridor=40.0)
    assert abs(at(S, 0, 0) - 5.0) < 0.05, "on the deck the surface must be the deck"
    assert abs(at(S, 0, -30) - 5.0) < 0.05
    # the ground road beside / below the bridge is another level: not part of this surface
    assert np.isnan(at(S, 25, 0)) and np.isnan(at(S, -25, -20))


def test_frames_on_another_level_are_excluded(tmp_path):
    """a drive passing UNDER the scene's road level (e.g. the scene is on a bridge): its poses are flagged off_level and its
    camera frames are not used (mapproj.snap_to_ground / Camera.on_level)."""
    from hdlabel.common import PoseInterp
    from hdlabel.mapproj import snap_to_ground
    H = np.full((500, 500), 5.0, np.float32)                                    # surface: a deck at z = 5 over 100 x 100 m
    np.savez(tmp_path / "g.npz", height=H, height_w=np.ones_like(H), lo=np.array([-50.0, -50.0]), surf_res=0.2, origin=np.zeros(3))
    t = np.arange(200) * 1e8; x = np.linspace(-40, 40, 200); z = np.where(np.abs(x) < 20, 0.0, 4.9)  # below the deck in the middle
    g = np.c_[t, x, np.zeros(200), z, np.zeros(200), np.zeros(200), np.zeros(200), np.ones(200)]
    np.save(tmp_path / "gt.npy", g); poser = PoseInterp(str(tmp_path / "gt.npy"))
    snap_to_ground(poser, str(tmp_path / "g.npz"))
    assert poser.off_level[np.abs(x) < 19].all() and not poser.off_level[np.abs(x) > 21].any()
