"""Pipeline control (hdlabel/run.py): resume / redo decisions and failure isolation of step()."""
import os

import numpy as np

import pytest
import yaml

from hdlabel import run


@pytest.fixture
def area(tmp_path):
    cfg = dict(name="t", pointcloud_map="/nonexistent/pointcloud_map.pcd", output=str(tmp_path / "out"),
               scenes=dict(length_m=100), dataset=dict(bev=dict(x=[0, 10], y=[-5, 5], res=0.5)))
    f = tmp_path / "a.yaml"; f.write_text(yaml.safe_dump(cfg)); return run.Area(str(f))


def script(tmp_path, body):
    p = tmp_path / "s.py"; p.write_text("import sys, time\n" + body); return str(p)


def test_skip_redo_and_failure(area, tmp_path):
    out = tmp_path / "o.txt"; cnt = tmp_path / "count.txt"
    s = script(tmp_path, f"open({str(cnt)!r}, 'a').write('x'); open(sys.argv[1], 'w').write(sys.argv[2])\n")
    run.step(area, "t1", s, [out, "a"], [str(out)])
    run.step(area, "t1", s, [out, "a"], [str(out)])                   # done: skipped
    assert cnt.read_text() == "x"
    run.step(area, "t1", s, [out, "b"], [str(out)])                   # arguments changed: redone
    assert cnt.read_text() == "xx" and out.read_text() == "b"
    inp = tmp_path / "in.txt"; inp.write_text("1"); os.utime(inp, (out.stat().st_mtime + 10,) * 2)
    run.step(area, "t1", s, [out, "b"], [str(out)], inputs=[str(inp)])   # input newer than output: redone
    assert cnt.read_text() == "xxx"
    bad = script(tmp_path, "sys.exit('boom')\n")
    with pytest.raises(run.StepError):
        run.step(area, "t2", bad, [], [str(tmp_path / "never.txt")])
    assert not os.path.exists(f"{area.work}/stamps/t2.json") and os.path.exists(f"{area.dbg}/logs/t2.log")


def test_status_record(area):
    area.record("label", "s1", False, log="x.log", note="boom"); area.record("label", "s2", True)
    assert area.status["label"]["s1"]["ok"] is False and area.status["label"]["s2"]["ok"] is True
    assert os.path.exists(f"{area.out}/status.json")


def test_label_sharing(area, monkeypatch):
    """repeated drives of a route: a scene with the same fused drives inside an earlier scene's box shares its labels."""
    drives = {"a_0": ["a", "b"], "b_0": ["b", "a"], "b_1": ["b", "a"], "c_0": ["c"]}
    monkeypatch.setattr(run, "scene_drives", lambda area_, s: drives[s["id"]])
    sc = [dict(id="a_0", bbox=[0, 0, 100, 100]), dict(id="b_0", bbox=[2, 1, 115, 99]),     # inside (20 m tolerance)
          dict(id="b_1", bbox=[50, 0, 200, 100]),                                         # elsewhere
          dict(id="c_0", bbox=[0, 0, 100, 100])]                                          # other drives
    assert run.label_owners(area, sc) == {"a_0": "a_0", "b_0": "a_0", "b_1": "b_1", "c_0": "c_0"}


def test_tile_runs_align_drives():
    """two drives over the same road, starting at different points: their scene cuts coincide (fixed map grid)."""
    road = lambda x0, x1: np.c_[np.arange(x0, x1, 0.5), np.zeros(len(np.arange(x0, x1, 0.5)))]
    cuts = []
    for xy in (road(10, 900), road(170, 1000)):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        cuts.append({c for c in (round(float(xy[r[0], 0])) for r in run.tile_runs(xy, s, 300)) if 170 < c < 900})   # common part
    assert cuts[0] == cuts[1] == {300, 600}


def test_config_layers(tmp_path):
    """configs/default.yaml <- project config <- command line; logs: a directory or an .mcap file is one drive."""
    (tmp_path / "logs" / "drive_2026-01-02-03-04-05").mkdir(parents=True)
    (tmp_path / "logs" / "drive_2026-01-02-03-04-05" / "a.mcap").write_bytes(b"")
    (tmp_path / "single.mcap").write_bytes(b"")
    f = tmp_path / "p.yaml"; f.write_text(yaml.safe_dump(dict(name="p", pointcloud_map="/x.pcd", scenes=dict(length_m=100))))
    c = run.load_config(str(f), logs=[str(tmp_path / "logs" / "*"), str(tmp_path / "single.mcap")], output=str(tmp_path / "out"))
    assert c["scenes"] == dict(length_m=100, context_m=150, corridor_m=40)     # merged with the defaults
    assert c["topics"]["pose"] and c["cameras"]                               # defaults kept
    area = run.Area(c)
    assert sorted(area.drives) == ["20260102-030405", "single"]
    assert area.map == "/x.pcd" and area.out == str(tmp_path / "out" / "p")
