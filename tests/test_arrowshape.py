"""Arrow direction from the paint mask (hdlabel/arrowshape.py). Conventions (easy to get wrong): patch rows = +u (lane
axis), cols = +v = driver's left; templates have the tip at the last row and the driver's left at the last column."""

import cv2
import numpy as np
import pytest

from hdlabel.arrowshape import TEMPLATES, paint_masks, shape


def patch_with(t, flip_lr=False, flip_ud=False, size=(400, 190)):
    m = np.zeros(size, bool)
    t = t > 0
    if flip_lr:
        t = t[:, ::-1]
    if flip_ud:
        t = t[::-1]
    r0, c0 = (size[0] - t.shape[0]) // 2, (size[1] - t.shape[1]) // 2
    m[r0 : r0 + t.shape[0], c0 : c0 + t.shape[1]] = t
    return m


@pytest.mark.parametrize(
    "name,mirror,expect",
    [
        ("S", False, ["straight"]),
        ("R", False, ["right"]),
        ("R", True, ["left"]),
        ("SL", False, ["straight", "left"]),
        ("SL", True, ["straight", "right"]),
    ],
)
def test_directions(name, mirror, expect):
    s = shape(patch_with(TEMPLATES[name], flip_lr=mirror))
    assert s["ok"] and s["directions"] == expect and s["sign"] == 1


def test_reversed_tip():
    s = shape(patch_with(TEMPLATES["R"], flip_lr=True, flip_ud=True))  # rotated by 180 deg: still a right arrow
    assert s["ok"] and s["directions"] == ["right"] and s["sign"] == -1


def test_plain_line_is_not_an_arrow():
    m = np.zeros((400, 190), bool)
    m[80:320, 90:98] = True  # 4.8 m x 0.16 m lane line dash
    assert not shape(m)["ok"]


def test_paint_segmentation_finds_centre_arrow():
    img = np.full((400, 190, 3), 90, np.uint8)
    img[patch_with(TEMPLATES["SL"])] = 200
    img = cv2.GaussianBlur(img, (3, 3), 0) + np.random.default_rng(0).integers(0, 6, img.shape, dtype=np.uint8)
    ms = paint_masks(img)
    assert ms and shape(ms[0])["directions"] == ["straight", "left"]


def test_nan_median_matches_numpy():
    from hdlabel.arrowshape import nan_median

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 255, (5, 30, 20, 3)).astype(np.float32)
    x[rng.uniform(size=x.shape) < 0.4] = np.nan
    x[:, 0, 0] = np.nan
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = np.nan_to_num(np.nanmedian(x, 0))
    assert np.allclose(nan_median(x), ref)
