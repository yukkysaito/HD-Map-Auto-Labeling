"""Paint refinement of the marking class (hdlabel/paint.py)."""

import numpy as np

from hdlabel.common import CLASSES
from hdlabel.paint import refine

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")


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
