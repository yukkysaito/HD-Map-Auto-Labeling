"""Pipeline control (hdlabel/run.py): resume / redo decisions and failure isolation of step()."""

import os

import numpy as np
import pytest
import yaml

from hdlabel import run


@pytest.fixture
def area(tmp_path):
    cfg = dict(
        name="t",
        pointcloud_map="/nonexistent/pointcloud_map.pcd",
        output=str(tmp_path / "out"),
        review_dir=str(tmp_path / "review"),  # not the repository's review/
        scenes=dict(length_m=100),
        dataset=dict(bev=dict(x=[0, 10], y=[-5, 5], res=0.5)),
    )
    f = tmp_path / "a.yaml"
    f.write_text(yaml.safe_dump(cfg))
    return run.Area(str(f))


def script(tmp_path, body):
    p = tmp_path / "s.py"
    p.write_text("import sys, time\n" + body)
    return str(p)


def test_skip_redo_and_failure(area, tmp_path):
    out = tmp_path / "o.txt"
    cnt = tmp_path / "count.txt"
    s = script(tmp_path, f"open({str(cnt)!r}, 'a').write('x'); open(sys.argv[1], 'w').write(sys.argv[2])\n")
    run.step(area, "t1", s, [out, "a"], [str(out)])
    run.step(area, "t1", s, [out, "a"], [str(out)])  # done: skipped
    assert cnt.read_text() == "x"
    run.step(area, "t1", s, [out, "b"], [str(out)])  # arguments changed: redone
    assert cnt.read_text() == "xx" and out.read_text() == "b"
    inp = tmp_path / "in.txt"
    inp.write_text("1")
    os.utime(inp, (out.stat().st_mtime + 10,) * 2)
    run.step(area, "t1", s, [out, "b"], [str(out)], inputs=[str(inp)])  # input newer than output: redone
    assert cnt.read_text() == "xxx"
    bad = script(tmp_path, "sys.exit('boom')\n")
    with pytest.raises(run.StepError):
        run.step(area, "t2", bad, [], [str(tmp_path / "never.txt")])
    assert not os.path.exists(f"{area.work}/stamps/t2.json") and os.path.exists(f"{area.dbg}/logs/t2.log")


def test_status_record(area):
    area.record("label", "s1", False, log="x.log", note="boom")
    area.record("label", "s2", True)
    assert area.status["label"]["s1"]["ok"] is False and area.status["label"]["s2"]["ok"] is True
    assert os.path.exists(f"{area.out}/status.json")


def test_label_sharing(area, monkeypatch):
    """repeated drives of a route: a scene with the same fused drives inside an earlier scene's box shares its labels."""
    drives = {"a_0": ["a", "b"], "b_0": ["b", "a"], "b_1": ["b", "a"], "c_0": ["c"]}
    monkeypatch.setattr(run, "scene_drives", lambda area_, s: drives[s["id"]])
    sc = [
        dict(id="a_0", bbox=[0, 0, 100, 100]),
        dict(id="b_0", bbox=[2, 1, 115, 99]),  # inside (20 m tolerance)
        dict(id="b_1", bbox=[50, 0, 200, 100]),  # elsewhere
        dict(id="c_0", bbox=[0, 0, 100, 100]),
    ]  # other drives
    assert run.label_owners(area, sc) == {"a_0": "a_0", "b_0": "a_0", "b_1": "b_1", "c_0": "c_0"}


def test_fixes_file_per_scene(area):
    """vectorize gets the review fixes inside its scene's box only: a fix elsewhere (another scene verified, a new
    drive's scenes) leaves the file, and so the scene's vectors and everything after them, untouched."""
    import json

    os.makedirs(area.review, exist_ok=True)
    fix = lambda x, y: dict(kind="lane_line", at=[x, y], action="delete")  # noqa: E731
    json.dump(dict(fixes=[fix(10, 10), fix(101, 50), dict(kind="arrow", add={})]), open(f"{area.review}/a.json", "w"))
    s = dict(id="a_000", bbox=[0, 0, 100, 100])
    f = run.fixes_file(area, s)
    assert [x["at"] for x in json.load(open(f))["fixes"]] == [[10, 10], [101, 50]]  # margin; "add": arrows.py
    t = os.path.getmtime(f)
    os.utime(f, (t - 10,) * 2)
    json.dump(dict(fixes=[fix(500, 500)]), open(f"{area.review}/b.json", "w"))  # another place
    assert run.fixes_file(area, s) == f and os.path.getmtime(f) == t - 10
    json.dump(dict(fixes=[fix(50, 50)]), open(f"{area.review}/b.json", "w"))  # this scene
    run.fixes_file(area, s)
    assert len(json.load(open(f))["fixes"]) == 3 and os.path.getmtime(f) > t - 10


def test_impact_is_a_dry_run(tmp_path, monkeypatch, capsys):
    """./autolabel impact changes nothing in the output tree (no folders, stamps, plans, fixes files, no removal of a
    shared scene's former labels) and still lists the steps that would run (vectorize: its fixes file would change)."""
    import json
    import sys

    out, rv = tmp_path / "out", tmp_path / "review"
    (tmp_path / "bags").mkdir()
    for d in ("a", "b"):
        (tmp_path / "bags" / f"{d}.mcap").write_bytes(b"")
    cfg = dict(name="t", pointcloud_map=str(tmp_path / "map.pcd"), output=str(out), review_dir=str(rv),
               scenes=dict(length_m=100), dataset=dict(bev=dict(x=[0, 10], y=[-5, 5], res=0.5)),
               drives={d: dict(bags=str(tmp_path / "bags" / f"{d}.mcap"), tf_from="a" if d == "b" else None)
                       for d in ("a", "b")})  # fmt: skip
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    W = out / "t" / "work"
    for d in ("a", "b"):
        (W / "drives" / d).mkdir(parents=True)
        (W / "drives" / d / "ingest.json").write_text(json.dumps(dict(cameras=[0])))
    (W / "drives" / "a" / "tf_static.json").write_text("{}")  # b's camera mounting would come from a (tf_from)
    sc = [dict(id=f"{d}_000", drive=d, bbox=[0, 0, 100, 100], t_range=[0, 1], route_t=[0, 1]) for d in ("a", "b")]
    (W / "scenes.json").write_text(json.dumps(dict(scenes=sc)))
    (W / "scenes" / "b_000" / "labels").mkdir(parents=True)  # labels of b_000 from before it shared a_000's
    (W / "scenes" / "b_000" / "labels" / "ground_map.npz").write_bytes(b"x")
    (W / "scenes" / "a_000").mkdir()
    rv.mkdir()
    (rv / "a_000.json").write_text(json.dumps(dict(fixes=[dict(kind="lane_line", at=[10, 10], action="delete")])))
    monkeypatch.setattr(run, "scene_drives", lambda area_, s: ["a", "b"])
    monkeypatch.setattr(run, "load_tf_static", lambda *a_: {"cam0": {}})

    def tree():
        return {str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in tmp_path.rglob("*")}

    before = tree()
    monkeypatch.setattr(sys, "argv", ["autolabel", "impact", str(tmp_path / "p.yaml")])
    run.main()
    assert run.DRY_RUN is None
    assert tree() == before
    got = capsys.readouterr().out
    assert "vectors_a_000" in got and "ingest_b" in got and "labels_b_000" not in got


def test_tile_runs_align_drives():
    """two drives over the same road, starting at different points: their scene cuts coincide (fixed map grid)."""
    road = lambda x0, x1: np.c_[np.arange(x0, x1, 0.5), np.zeros(len(np.arange(x0, x1, 0.5)))]
    cuts = []
    for xy in (road(10, 900), road(170, 1000)):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        cuts.append(
            {c for c in (round(float(xy[r[0], 0])) for r in run.tile_runs(xy, s, 300)) if 170 < c < 900}
        )  # common part
    assert cuts[0] == cuts[1] == {300, 600}


def test_config_layers(tmp_path):
    """configs/default.yaml <- project config <- command line; logs: a directory or an .mcap file is one drive."""
    (tmp_path / "logs" / "drive_2026-01-02-03-04-05").mkdir(parents=True)
    (tmp_path / "logs" / "drive_2026-01-02-03-04-05" / "a.mcap").write_bytes(b"")
    (tmp_path / "single.mcap").write_bytes(b"")
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(dict(name="p", pointcloud_map="/x.pcd", scenes=dict(length_m=100))))
    c = run.load_config(
        str(f), logs=[str(tmp_path / "logs" / "*"), str(tmp_path / "single.mcap")], output=str(tmp_path / "out")
    )
    assert c["scenes"] == dict(length_m=100, context_m=150, corridor_m=40)  # merged with the defaults
    assert c["topics"]["pose"] and c["cameras"]  # defaults kept
    area = run.Area(c)
    assert sorted(area.drives) == ["20260102-030405", "single"]
    assert area.map == "/x.pcd" and area.out == str(tmp_path / "out" / "p")


def test_code_change_redoes(area, tmp_path):
    """a changed step script is run again even with the same arguments (run.code_version)."""
    out = tmp_path / "o.txt"
    s = script(tmp_path, "open(sys.argv[1], 'w').write('v1')\n")
    run.step(area, "t3", s, [out], [str(out)])
    open(s, "w").write("import sys\nopen(sys.argv[1], 'w').write('v2')\n")
    run._CODE.clear()  # a new run of the pipeline hashes the code again
    run.step(area, "t3", s, [out], [str(out)])
    assert out.read_text() == "v2"


def test_cameras_replaced(tmp_path):
    """a project's camera list replaces the default one (no default cameras reported missing)."""
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(dict(name="p", cameras={0: {}}, scenes=dict(length_m=50))))
    c = run.load_config(str(f))
    assert list(c["cameras"]) == [0] and c["scenes"]["context_m"] == 150


def test_code_version_follows_imports():
    """the hash covers modules imported as `from hdlabel import x` and `from hdlabel.x import y` (transitively)."""
    deps = {os.path.basename(f) for f in run._hdlabel_imports(f"{run.ROOT}/hdlabel/steps/review.py")}
    assert {"review_html.py", "bev.py", "common.py", "mapproj.py"} <= deps


def test_class_map_env(tmp_path):
    """segmentation.class_map reaches the steps as $HDLABEL_CLASS_MAP and replaces the Vistas ids of common.GROUPS."""
    import json
    import subprocess
    import sys

    cm = dict(road=[1], marking=[2], crosswalk=[3], curb=[5], sidewalk=[6], offroad=[7, 8], vertical=[8])
    env = dict(os.environ, HDLABEL_CLASS_MAP=json.dumps(cm), PYTHONPATH=run.ROOT)
    code = "from hdlabel.mapproj import seg_lut; l = seg_lut(); print(l[1], l[2], l[7], l[8], l[13])"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["0", "1", "5", "6", "6"]  # road, marking, offroad; vertical and unknown ids -> other (never voted)


def test_split_by_location_with_fixed_test_tiles(tmp_path):
    np.save(tmp_path / "gt.npy", np.array([[0.0, 600.0, 100.0], [1.0, 620.0, 120.0]]))  # route centre in tile (2, 0)
    area = type("A", (), dict(data=lambda self, d: str(tmp_path), cfg=dict(dataset=dict(split=dict(tile_m=250)))))()
    s = dict(drive="d", t_range=[0.0, 1.0])
    assert run.scene_split(area, s)[1] == [2, 0] and run.scene_split(area, s)[0] in ("train", "val")
    area.cfg["dataset"]["split"]["test_tiles"] = [[2, 0]]
    assert run.scene_split(area, s) == ("test", [2, 0])
