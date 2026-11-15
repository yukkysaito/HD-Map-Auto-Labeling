"""stop line classification (common.stop_lines): wide paint across the traffic direction."""
import numpy as np

from hdlabel.common import stop_lines


def line(p0, p1, width, style="short"):
    xy = np.linspace(p0, p1, 400)                   # dense like the skeleton lines of vectorize.py
    return dict(xy=xy, length=float(np.linalg.norm(np.subtract(p1, p0))), width=width, style=style)


def test_stop_line_across_lanes():
    lanes = [line((0, y), (40, y), 0.15, "dashed") for y in (-3.5, 0.0, 3.5)]          # traffic along x
    cand = [line((30, -5), (30.3, 5), 0.45),        # stop bar across the lanes -> stop line
            line((30, -5), (30.3, 5), 0.15),        # thin line across (e.g. guide dashes) -> no
            line((10, 1.7), (20, 1.7), 0.45),       # wide line along the lanes -> no
            line((30, 6), (30, 8), 0.45)]           # across but too short -> no
    out = stop_lines(lanes + cand)
    assert out == {3}


def test_no_reference_lanes():
    assert stop_lines([line((0, 0), (0, 5), 0.45)]) == set()
