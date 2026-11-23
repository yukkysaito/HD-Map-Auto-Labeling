"""Paint refinement of the marking and crosswalk classes (hdlabel/paint.py)."""

import types

import numpy as np

from hdlabel.common import CLASSES
from hdlabel.paint import XWALK_DEFAULTS, refine, refine_crosswalk
from hdlabel.run import load_config, projection_args

ROAD, MARKING, CROSSWALK = CLASSES.index("road"), CLASSES.index("marking"), CLASSES.index("crosswalk")


def scene(sharp: bool):
    """road with a segmented marking blob (40 x 40 cells) over two painted strokes; blurred: no paint contrast."""
    cls = np.full((100, 100), ROAD, np.uint8)
    cls[30:70, 30:70] = MARKING
    rgb = np.full((100, 100, 3), 80, np.uint8)
    if sharp:
        rgb[35:65, 38:42] = rgb[35:65, 55:59] = 220
    return cls, rgb


def test_blob_follows_the_paint():
    out, st = refine(*scene(True), res=0.1)
    assert (out[35:65, 38:42] == MARKING).all() and (out[35:65, 55:59] == MARKING).all()
    assert (out[50, 45:52] == ROAD).all() and (out[31, 31] == ROAD)  # unpainted inside of the blob
    assert st["refined_cells"] > 0 and st["marking_after"] < st["marking_before"]


def test_blob_without_paint_contrast_is_kept():
    cls, rgb = scene(False)
    out, st = refine(cls, rgb, res=0.1)
    assert (out == cls).all() and st["refined_cells"] == 0


def test_a_long_line_refines_only_where_its_paint_shows():
    cls = np.full((40, 200), ROAD, np.uint8)
    cls[18:23, 10:190] = MARKING  # segmented line, 0.5 m wide (blob)
    rgb = np.full((40, 200, 3), 80, np.uint8)
    rgb[19:22, 10:100] = 220  # painted 0.3 m wide in the sharp half; the far half is blurred (no contrast)
    out, _ = refine(cls, rgb, res=0.1)
    assert (out[18, 20:90] == ROAD).all() and (out[20, 20:90] == MARKING).all()  # sharp: follows the paint
    assert (out[18:23, 120:190] == MARKING).all()  # blurred: kept as segmented


def crossing(stripe=(200, 200, 200), pavement=(80, 80, 80)):
    """road with a segmented crosswalk area (60 x 40 cells) over 0.4 m stripes 0.9 m apart."""
    cls = np.full((100, 100), ROAD, np.uint8)
    cls[20:80, 30:70] = CROSSWALK
    rgb = np.full((100, 100, 3), pavement, np.uint8)
    for r in range(22, 78, 9):
        rgb[r : r + 4, 32:68] = stripe
    return cls, rgb


def test_crosswalk_gaps_become_road():
    cls, rgb = crossing()
    out, st = refine_crosswalk(cls, rgb, res=0.1)
    assert (out[22:26, 35:65] == CROSSWALK).all() and (out[40:44, 35:65] == CROSSWALK).all()  # stripes
    assert (out[27:30, 35:65] == ROAD).all()  # gap between two stripes
    assert (out[cls != CROSSWALK] == cls[cls != CROSSWALK]).all()  # nothing else changes
    assert st["to_road"] > 0


def test_crosswalk_on_red_pavement():
    out, _ = refine_crosswalk(*crossing((225, 225, 225), (200, 60, 60)), res=0.1)  # little contrast in red
    assert (out[40:44, 35:65] == CROSSWALK).all() and (out[45:48, 35:65] == ROAD).all()


def test_crosswalk_without_paint_contrast_is_kept():
    cls, rgb = crossing(stripe=(85, 85, 85))  # blurred / far: the stripes barely show
    out, st = refine_crosswalk(cls, rgb, res=0.1)
    assert (out == cls).all() and st["to_road"] == 0


def test_crosswalk_worn_spot_in_a_stripe_is_filled():
    cls, rgb = crossing()
    rgb[40:43, 50:52] = 80  # 0.06 m2 of worn paint in a stripe
    out, _ = refine_crosswalk(cls, rgb, res=0.1)
    assert (out[40:43, 50:52] == CROSSWALK).all()


def test_config_default_off():
    cfg = load_config(None)
    area = types.SimpleNamespace(cfg=cfg)
    assert "--paint-crosswalk" not in projection_args(area)  # unchanged commands
    cfg["projection"]["paint"]["crosswalk"] = dict(enabled=True, contrast=40)
    args = projection_args(area)
    i = args.index("--paint-crosswalk")
    assert args[i + 1 : i + 1 + len(XWALK_DEFAULTS)] == [40, *list(XWALK_DEFAULTS.values())[1:]]
    assert "--paint" not in args
