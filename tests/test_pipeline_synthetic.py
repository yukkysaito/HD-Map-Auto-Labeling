"""End-to-end on synthetic data (tests/synthetic.py), on CPU: bag -> dataset + debug output, and the labels match the
synthetic world (stop line, straight arrow, crosswalk at their true positions)."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synthetic as syn

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("synthetic"))
    cfg = syn.make(d)
    env = dict(os.environ, HDLABEL_DEVICE="cpu", PYTHONPATH="")
    r = subprocess.run([sys.executable, f"{ROOT}/hdlabel/run.py", "run", cfg], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return f"{d}/synthetic"


def features(out, kind):
    fs = []
    for f in glob.glob(f"{out}/work/scenes/*/vectors/vectors.geojson"):
        fs += [x for x in json.load(open(f))["features"] if x["properties"]["kind"] == kind]
    return fs


def road(c):  # map -> road frame of tests/synthetic.py
    return np.asarray(c, float)[:, :2] - [syn.X0, syn.Y0]


def test_dataset(out):
    ds = json.load(open(f"{out}/dataset/dataset.json"))
    assert ds["format"] == 3 and ds["qa_ok"] > 0 and ds["qa_ok"] == ds["samples"]
    assert os.path.exists(f"{out}/debug/scenes/{ds['scenes'][0]['scene']}/sample_0000.jpg")


def test_stop_line(out):
    st = features(out, "stop_line")
    assert st
    P = road(st[0]["geometry"]["coordinates"])
    assert np.all(np.abs(P[:, 0] - syn.STOP_X) < 0.3) and P[:, 1].min() < -4.5 and P[:, 1].max() > -1.5


def test_arrow(out):
    ar = features(out, "arrow")
    assert ar and all(a["properties"]["directions"] == ["straight"] for a in ar)
    tail, tip = road(ar[0]["geometry"]["coordinates"])
    assert tip[0] > tail[0] and abs((tail[1] + tip[1]) / 2 + 4.0) < 0.3 and syn.ARROW_X - 1 < tail[0] < syn.ARROW_X + 1


def test_crosswalk(out):
    cw = features(out, "crosswalk")
    assert cw
    P = road(cw[0]["geometry"]["coordinates"][0])
    assert syn.CW_X - 0.5 < P[:, 0].min() and P[:, 0].max() < syn.CW_X + 4.5


def test_validate(out, tmp_path):
    """validate: the exported dataset passes; a copy with a missing image and a broken pose fails."""
    import shutil

    from hdlabel.validate import validate

    assert not validate(f"{out}/dataset/dataset.json").errors
    bad = tmp_path / "dataset"
    shutil.copytree(f"{out}/dataset", bad)  # hard links are copied as files
    scene = sorted(p for p in bad.iterdir() if p.is_dir())[0]
    os.remove(next(scene.glob("images/*/*.jpg")))
    lines = (scene / "samples.jsonl").read_text().splitlines()
    s = json.loads(lines[0])
    s["T_map_base"][0][0] = 2.0  # not a rotation
    (scene / "samples.jsonl").write_text("\n".join([json.dumps(s)] + lines[1:]) + "\n")
    errs = validate(str(bad / "dataset.json")).errors
    assert any("missing image" in e for e in errs) and any("rigid" in e for e in errs)


def test_review_fixes_apply(out):
    """fixes as review.html saves them (map coordinates) change the vectors on the next run: a lane line is dropped,
    the arrow directions are overwritten."""
    lane = max(features(out, "lane_line"), key=lambda f: f["properties"]["length"])
    P = np.asarray(lane["geometry"]["coordinates"])
    arrow = features(out, "arrow")[0]
    (x0, y0, _), (x1, y1, _) = arrow["geometry"]["coordinates"]
    root = os.path.dirname(out)
    os.makedirs(f"{root}/review", exist_ok=True)
    fixes = [
        dict(kind="lane_line", at=P[len(P) // 2][:2].tolist(), drop=True),
        dict(kind="arrow", at=[(x0 + x1) / 2, (y0 + y1) / 2], set=dict(directions=["straight", "left"], shape="SL")),
    ]
    json.dump(dict(reviewer="test", fixes=fixes), open(f"{root}/review/any_scene.json", "w"))
    env = dict(os.environ, HDLABEL_DEVICE="cpu", PYTHONPATH="")
    r = subprocess.run(
        [sys.executable, f"{ROOT}/hdlabel/run.py", "run", f"{root}/config.yaml"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:]
    lengths = [f["properties"]["length"] for f in features(out, "lane_line")]
    assert lane["properties"]["length"] not in lengths
    assert all(a["properties"]["directions"] == ["straight", "left"] for a in features(out, "arrow"))
    assert all(a["properties"].get("reviewed") for a in features(out, "arrow"))
