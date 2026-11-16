"""Review sheets of one scene for a visual check by a reviewer (a person or Claude reading the images).

  OUT/sample_<k>.jpg   for a few samples: every camera image with the scene labels rendered into it (ray / surface
                       intersection; occluders are not modelled, so labels also paint over cars) + the BEV raster with
                       the sample vectors. Overlay colours: road green tint, marking red, crosswalk magenta, curb blue,
                       sidewalk yellow, manhole cyan; vertical structures / unobserved: none.
  OUT/map.jpg          ortho image of the scene with the vectors, element ids and a 10 m grid (map coordinates)
  OUT/elements.json    id -> kind, properties, a point on the element (map frame) for writing fixes
  OUT/arrows.jpg       (written by arrows.py) arrows found (sharp ground patch + outline, tip up, driver's left on the
                       left; "A<n> S+L" on map.jpg), for checking the directions
  OUT/elements.jpg     one ortho crop per not yet reviewed marking_area / crosswalk with id, area, crop size (for classifying)
  OUT/check/           evidence images + items.json of the elements to decide (hdlabel/evidence.py, docs/AGENT_REVIEW.md)
  OUT/review.html      guided review in a browser (hdlabel/review_html.py): check items, add missed ones, accept / reject
The reviewer writes review/<scene>.json (see docs/ANNOTATION.md); vectorize.py / export.py apply it.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hdlabel import evidence, review_html
from hdlabel.bev import SceneMap
from hdlabel.common import CLASSES, VEC_COLOR, draw_bev
from hdlabel.mapproj import Projector


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="export.py output of the scene")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True, help="the scene's debug dir")
    p.add_argument("--ortho", help="ortho.jpg of the labels (project_labels.py --debug; default OUT/ortho.jpg)")
    p.add_argument("--review", help="the scene's review json (its fixes are shown in review.html)")
    p.add_argument(
        "--others", nargs="*", default=[], help="the other review files: what they decided is not asked again"
    )
    p.add_argument("--suspects", help="model_suspects.json (agents/training.py): places to verify from the evidence")
    p.add_argument("--n", type=int, default=3, help="samples per scene")
    p.add_argument("--width", type=int, default=560)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    scene = json.load(open(f"{a.dataset}/scene.json"))
    rows = [json.loads(l) for l in open(f"{a.dataset}/samples.jsonl")]
    frames = [json.loads(l) for l in open(f"{a.dataset}/frames.jsonl")]
    sm = SceneMap(f"{a.dataset}/map.npz", f"{a.dataset}/vectors.geojson")
    m = np.load(f"{a.dataset}/map.npz")
    cls = m["cls"]
    pj = Projector(f"{a.labels}/ground_map.npz")
    VECTORS = f"{a.dataset}/vectors.geojson"
    OV = np.zeros((len(CLASSES), 3), np.uint8)
    W = np.zeros(len(CLASSES))
    for name, col, w in (
        ("road", (0, 160, 0), 0.25),
        ("marking", (255, 0, 0), 0.7),
        ("crosswalk", (255, 0, 255), 0.6),
        ("curb", (0, 80, 255), 0.7),
        ("sidewalk", (255, 220, 0), 0.4),
        ("manhole", (0, 255, 255), 0.6),
    ):
        OV[CLASSES.index(name)] = col
        W[CLASSES.index(name)] = w

    def overlay(img, labels):
        L = cv2.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        w = W[L][..., None]
        return (img * (1 - w) + OV[L][..., ::-1] * w).astype(np.uint8)

    def put(img, text, y=22):
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    bev = scene["bev"]
    for k in np.linspace(0, len(rows) - 1, min(a.n, len(rows))).astype(int):
        row = rows[k]
        tiles = []
        T_s = np.array(row["T_map_base"])
        for c, ids in row["frames"].items():
            f = frames[ids[0]]
            cm = scene["cameras"][c]
            T = np.array(f["T_map_base"]) @ np.array(cm["T_base_cam"])
            img = cv2.imread(f"{a.dataset}/{f['image']}")
            K = np.array(cm["K"])
            im = overlay(img, pj.render(T, K, cm["size"], cls, stride=4))
            im = cv2.resize(im, (a.width, round(im.shape[0] * a.width / im.shape[1])))
            put(im, f"cam{c}  dt {row['dt_ms'][c]:+.0f} ms")
            tiles.append(im)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
        b, _ = sm.crop(T_s, bev["x"], bev["y"], bev["res"])
        bv = draw_bev(b, sm.vectors(T_s, bev["x"], bev["y"]), bev["x"], bev["y"], bev["res"], scale=1)
        sc = grid.shape[0] / bv.shape[0]
        bv = cv2.resize(bv, (round(bv.shape[1] * sc), grid.shape[0]), interpolation=cv2.INTER_NEAREST)
        put(bv, f"sample {k}  stamp {row['stamp']}  qa {'ok' if row['qa']['ok'] else 'NG'}")
        cv2.imwrite(f"{a.out}/sample_{k:04d}.jpg", np.hstack([grid, bv]), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # map sheet: ortho + vectors + ids + 10 m grid
    ortho = cv2.imread(a.ortho or f"{a.out}/ortho.jpg")
    if ortho is None:
        raise SystemExit(f"no {a.ortho or a.out + '/ortho.jpg'} (project_labels.py --debug)")
    lo, res, origin = m["lo"], float(m["res"]), m["origin"]
    Wc = cls.shape[1]
    img = (ortho * 0.65).astype(np.uint8)
    ipx = lambda q: np.round(
        np.c_[(q[:, 0] - origin[0] - lo[0]) / res, Wc - (q[:, 1] - origin[1] - lo[1]) / res]
    ).astype(np.int32)
    for gx in np.arange(np.ceil((origin[0] + lo[0]) / 10) * 10, origin[0] + lo[0] + cls.shape[0] * res, 10):
        x = int((gx - origin[0] - lo[0]) / res)
        cv2.line(img, (x, 0), (x, img.shape[0]), (60, 60, 60), 1)
        if gx % 50 == 0:
            cv2.putText(img, f"{gx:.0f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for gy in np.arange(np.ceil((origin[1] + lo[1]) / 10) * 10, origin[1] + lo[1] + Wc * res, 10):
        y = int(Wc - (gy - origin[1] - lo[1]) / res)
        cv2.line(img, (0, y), (img.shape[1], y), (60, 60, 60), 1)
        if gy % 50 == 0:
            cv2.putText(img, f"{gy:.0f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    els = []
    for f in json.load(open(VECTORS))["features"]:
        pr, g = f["properties"], f["geometry"]
        kind = pr["kind"]
        if kind == "drivable_area":
            continue
        pts = np.array(g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"], float)[:, :2]
        col = VEC_COLOR.get(kind, (0, 0, 255) if pr.get("observed") else (128, 128, 128))
        if kind == "lane_line" and pr["style"] != "solid":
            col = (0, 255, 255)
        cv2.polylines(img, [ipx(pts)], g["type"] == "Polygon", col, 1, cv2.LINE_AA)
        if kind == "arrow":  # outline, tip, "A<arrows.jpg id> S+L"
            cv2.polylines(img, [ipx(np.array(o, float)[:, :2]) for o in pr["outlines"]], True, col, 1, cv2.LINE_AA)
            at = pts.mean(0)
            q = ipx(pts[-1:])[0]
            cv2.circle(img, (int(q[0]), int(q[1])), 3, col, -1)
            els.append(
                dict(
                    id=len(els),
                    kind=kind,
                    at=np.round(at, 2).tolist(),
                    arrow_id=pr["arrow_id"],
                    directions=pr["directions"],
                    score=pr["score"],
                )
            )
            q = ipx(at[None])[0]
            cv2.putText(
                img,
                f"A{pr['arrow_id']} " + "+".join(d[0].upper() for d in pr["directions"]),
                (int(q[0]) + 4, int(q[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )
        if kind in ("lane_line", "stop_line", "marking_area", "crosswalk"):
            at = pts[len(pts) // 2] if g["type"] == "LineString" else pts.mean(0)
            i = len(els)
            els.append(
                dict(
                    id=i,
                    kind=kind,
                    at=np.round(at, 2).tolist(),
                    **{k: v for k, v in pr.items() if k in ("style", "length", "type")},
                )
            )
            q = ipx(at[None])[0]
            cv2.putText(
                img,
                str(i),
                (int(q[0]) + 3, int(q[1]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    json.dump(els, open(f"{a.out}/elements.json", "w"), indent=0)

    # element sheet: one ortho crop per area element (marking_area, crosswalk) with its outline and id, for classifying them
    base = (ortho * 0.9).astype(np.uint8)
    tiles = []
    for f in json.load(open(VECTORS))["features"]:
        pr, g = f["properties"], f["geometry"]
        if pr["kind"] not in ("marking_area", "crosswalk") or pr.get("reviewed"):
            continue  # already fixed by a review
        pts = np.array(g["coordinates"][0], float)[:, :2]
        at = np.round(pts.mean(0), 2).tolist()
        i = next(e["id"] for e in els if e["kind"] == pr["kind"] and e["at"] == at)
        q = ipx(pts)
        c0, r0 = q.min(0) - 20
        c1, r1 = q.max(0) + 20
        s = max(c1 - c0, r1 - r0)
        cc, rc = (c0 + c1) // 2, (r0 + r1) // 2
        c0, r0 = max(0, cc - s // 2), max(0, rc - s // 2)
        crop = base[r0 : r0 + s, c0 : c0 + s].copy()
        if crop.size == 0:
            continue
        cv2.polylines(
            crop, [q - [c0, r0]], True, (0, 255, 0) if pr["kind"] == "marking_area" else (255, 0, 255), 1, cv2.LINE_AA
        )
        crop = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA if s > 200 else cv2.INTER_CUBIC)
        area_m2 = abs(cv2.contourArea(pts.astype(np.float32)))
        put(crop, f"{i} {'MA' if pr['kind'] == 'marking_area' else 'CW'} {area_m2:.0f}m2 {s * res:.0f}m", 16)
        tiles.append(crop)
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(
            f"{a.out}/elements.jpg",
            np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
    cv2.imwrite(f"{a.out}/map.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def sample_xy(f):  # ego position of sample_<k>.jpg (k = row index), for showing the camera views near an element
        k = int(os.path.basename(f)[7:-4])
        return [round(float(v), 2) for v in np.array(rows[k]["T_map_base"])[:2, 3]] if k < len(rows) else None

    features = json.load(open(VECTORS))["features"]
    review = json.load(open(a.review)) if a.review and os.path.exists(a.review) else None
    scene_id = os.path.basename(os.path.normpath(a.out))
    rels = review_html.elements(features)
    queue, reasons = review_html.write(
        f"{a.out}/review.html", scene_id, os.path.relpath(a.ortho or f"{a.out}/ortho.jpg", a.out), cls.shape,
        m["lo"], float(m["res"]), m["origin"], rels, review,
        [dict(file=os.path.basename(f), xy=sample_xy(f)) for f in sorted(glob.glob(f"{a.out}/sample_*.jpg"))],
        a.review,
    )  # fmt: skip

    # evidence pack (check/) for the reviewer: the queue minus what a review already decided; once a scene has been
    # reviewed, no new spot check / map tiles are drawn (they would change with every fix)
    reviewed = bool(review and "checked" in review)
    others = [json.load(open(f)) for f in a.others if os.path.exists(f) and f != a.review]
    items = [
        dict(feature=features[rels[i]["src"]], group=rels[i]["group"], at=rels[i]["at"], reason=reasons[i])
        for i in queue
        if not any(review_html.decided(r, rels[i]["kind"], rels[i]["at"]) for r in [review] + others)
        and not (reviewed and reasons[i] != "low confidence")
    ]
    siblings = sorted(
        os.path.dirname(f) for f in glob.glob(f"{os.path.dirname(os.path.normpath(a.dataset))}/*/frames.jsonl")
    )
    sc = evidence.Scene(a.dataset, a.ortho or f"{a.out}/ortho.jpg", siblings)
    shown = [f for f in features if review_html.GROUP.get(f["properties"]["kind"]) in evidence.COLOR]
    tiles = [] if reviewed else evidence.map_tiles(sc, scene_id)
    if a.suspects and os.path.exists(a.suspects):  # where the trained model disagrees with the labels
        lo, res, org = m["lo"], float(m["res"]), m["origin"]
        hi = lo + np.array(cls.shape) * res
        done_tiles = [t["center"] for r in [review] + others if r for t in r.get("tiles", [])]
        for sp in json.load(open(a.suspects)):
            p_ = np.array(sp["map_xy"]) - org[:2]
            if not np.all((p_ > lo) & (p_ < hi)):
                continue
            if sp["type"] == "model_only":  # possibly missed paint: a map tile around it
                if not any(np.abs(np.subtract(c, sp["map_xy"])).max() < 10 for c in done_tiles + tiles):
                    tiles.append([round(float(v), 2) for v in sp["map_xy"]])
                continue
            grp = "curb" if sp["cls"] == "curb" else "marking"
            near = [
                i for i, e in enumerate(rels) if e["group"] == grp and np.hypot(*np.subtract(e["at"], sp["map_xy"])) < 3
            ]
            for i in near[:1]:
                if not any(review_html.decided(r, rels[i]["kind"], rels[i]["at"]) for r in [review] + others if r) \
                        and not any(it["at"] == rels[i]["at"] for it in items):  # fmt: skip
                    items.append(dict(feature=features[rels[i]["src"]], group=grp, at=rels[i]["at"],
                                      reason=f"model disagreement ({sp['samples']} samples)"))  # fmt: skip
    evidence.write_pack(a.out, sc, shown, items, tiles, review_html.GROUP)
    print(f"review sheets -> {a.out} ({len(els)} numbered elements, review.html)")


if __name__ == "__main__":
    main()
