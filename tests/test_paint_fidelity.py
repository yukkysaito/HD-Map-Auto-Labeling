"""Paint fidelity of the marking class (hdlabel/eval/paint_fidelity.py) on synthetic rasters."""

import json

import cv2
import numpy as np

from hdlabel.common import CLASSES
from hdlabel.eval.paint_fidelity import area_mask, main, score

ROAD, MARKING = CLASSES.index("road"), CLASSES.index("marking")


def scene():
    """asphalt (80) with a painted lane line (rows 20-21, 0.2 m wide, 12 m long) and a text-like patch of three
    vertical strokes (0.3 m wide, 0.3 m apart) at rows 60-80; slight texture noise."""
    rng = np.random.default_rng(0)
    g = np.full((120, 160), 80.0) + rng.normal(0, 2, (120, 160))
    paint = np.zeros((120, 160), bool)
    paint[20:22, 20:140] = True
    for c in (60, 66, 72):
        paint[60:80, c : c + 3] = True
    g[paint] = 220
    rgb = np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, 2)
    return rgb, paint


def labels(mark):
    cls = np.full(mark.shape, ROAD, np.uint8)
    cls[mark] = MARKING
    return cls


def test_strokes_beat_a_blob():
    rgb, paint = scene()
    blob = paint.copy()
    blob[58:82, 58:77] = True  # the segmentation's rectangle around the text
    exact, coarse = score(labels(paint), rgb, 0.1), score(labels(blob), rgb, 0.1, ref=labels(paint))
    a, b = exact["area"], coarse["area"]
    assert a["boundary_f"] > 0.9 and b["boundary_f"] < 0.7
    assert a["dark"] < 0.1 and b["dark"] > 0.4
    assert a["paint_iou"] > 0.9 and b["paint_iou"] < 0.6
    assert a["contrast"] > 100 and b["contrast"] < a["contrast"] / 2
    assert exact["line"]["boundary_f"] > 0.9 and exact["line"]["cells"] == 240  # the lane line is a line


def test_too_thin_label_misses_paint():
    rgb, paint = scene()
    thin = np.zeros_like(paint)
    thin[20, 20:140] = True  # half of the line's width
    s = score(labels(thin), rgb, 0.1)["line"]
    assert s["missed"] > 0.4 and s["dark"] < 0.1 and s["boundary_recall"] < 0.6


def test_groups():
    _, paint = scene()
    area = area_mask(paint, 0.1, 2.0, 0.7)
    assert not area[20:22].any()  # lane line
    assert area[60:80, 60:75][paint[60:80, 60:75]].all()  # strokes merged into a text area


def test_no_marking_and_no_contrast():
    rgb, paint = scene()
    assert score(np.full(paint.shape, ROAD, np.uint8), rgb, 0.1)["all"] is None
    flat = np.full_like(rgb, 80)
    s = score(labels(paint), flat, 0.1)["all"]
    assert s["dark"] is None and s["low_contrast_pieces"] == s["pieces"] and s["boundary_recall"] is None


def test_cli(tmp_path):
    rgb, paint = scene()
    np.savez(tmp_path / "g.npz", cls=labels(paint), res=0.1)
    cv2.imwrite(str(tmp_path / "o.png"), np.rot90(rgb)[..., ::-1])  # as project_labels.py writes ortho.jpg
    main([str(tmp_path / "g.npz"), str(tmp_path / "o.png"), "--out", str(tmp_path / "s.json")])
    s = json.load(open(tmp_path / "s.json"))
    assert s["all"]["boundary_f"] > 0.9 and s["area"]["dark"] < 0.1
