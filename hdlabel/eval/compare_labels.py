"""Evaluation: agreement of two ground label maps (project_labels.py ground_map.npz), without any HD map.

Used for label QA by consistency:
  - repeatability: the same area labelled from two independent drives,
  - cross-camera: labels from one camera vs. labels from the other cameras (camera-specific geometric bias),
  - method: e.g. camera-only path (extras/camera_only) vs. map path.
On cells observed in both maps (optionally only within --corridor m of a trajectory): per-class IoU, overall
agreement, and for thin structures (marking, crosswalk, curb) a boundary F-score with a --tol m tolerance
(precision: share of A's class cells within tol of B's, recall: vice versa) and the symmetric mean distance between
them (chamfer, each cell's distance capped at 1 m).
Writes OUT.json and OUT.png (agreement map: grey agree, red only A, blue only B for markings).
"""

import argparse
import json

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=0.2)
    p.add_argument("--gt", help="restrict to --corridor m around this trajectory")
    p.add_argument("--corridor", type=float, default=15.0)
    a = p.parse_args(argv)

    A, B = np.load(a.a), np.load(a.b)
    res = float(A["res"])
    assert abs(float(B["res"]) - res) < 1e-9
    groups = [str(g) for g in A["groups"]]
    assert groups == [str(g) for g in B["groups"]]
    # common grid: B's cells shifted into A's index space
    shift = np.rint((B["lo"] + B["origin"][:2] - A["lo"] - A["origin"][:2]) / res).astype(int)
    ca, cb = A["cls"], B["cls"]
    cb2 = np.zeros_like(ca)
    r0, c0 = max(0, shift[0]), max(0, shift[1])
    r1, c1 = min(ca.shape[0], shift[0] + cb.shape[0]), min(ca.shape[1], shift[1] + cb.shape[1])
    cb2[r0:r1, c0:c1] = cb[r0 - shift[0] : r1 - shift[0], c0 - shift[1] : c1 - shift[1]]
    cb = cb2
    both = (ca > 0) & (cb > 0)
    if a.gt:
        g = np.load(a.gt)[:, 1:3] - A["origin"][:2]
        ij = ((g - A["lo"]) / res).astype(int)
        m = np.zeros(ca.shape, bool)
        ok = np.all((ij >= 0) & (ij < ca.shape), 1)
        m[ij[ok, 0], ij[ok, 1]] = True
        both &= distance_transform_edt(~m) * res < a.corridor

    out = dict(
        a=a.a,
        b=a.b,
        common_cells_km2=round(float(both.sum()) * res**2 / 1e6, 4),
        only_a_km2=round(float(((ca > 0) & (cb == 0)).sum()) * res**2 / 1e6, 4),
        only_b_km2=round(float(((cb > 0) & (ca == 0)).sum()) * res**2 / 1e6, 4),
        agreement=round(float((ca[both] == cb[both]).mean()), 4),
        iou={},
        thin={},
    )
    for j, g in enumerate(groups):
        x, y = (ca == j + 1) & both, (cb == j + 1) & both
        if x.sum() + y.sum() > 100:
            out["iou"][g] = round(float((x & y).sum() / (x | y).sum()), 4)
    tol = a.tol / res
    for g in ("marking", "crosswalk", "curb"):
        j = groups.index(g) + 1
        x, y = (ca == j) & both, (cb == j) & both
        if x.sum() < 100 or y.sum() < 100:
            continue
        dy, dx = distance_transform_edt(~y), distance_transform_edt(~x)
        prec, rec = float((dy[x] <= tol).mean()), float((dx[y] <= tol).mean())
        out["thin"][g] = dict(
            precision=round(prec, 4),
            recall=round(rec, 4),
            f=round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            chamfer_cm=round(
                float((np.minimum(dy[x], 1.0 / res).mean() + np.minimum(dx[y], 1.0 / res).mean()) / 2) * res * 100, 1
            ),
        )
    # confusion (row: A, col: B) on common cells
    conf = np.zeros((len(groups), len(groups)), np.int64)
    np.add.at(conf, (ca[both] - 1, cb[both] - 1), 1)
    out["confusion"] = dict(labels=groups, rows_a_cols_b=conf.tolist())
    json.dump(out, open(f"{a.out}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=1))

    mk = groups.index("marking") + 1
    vis = np.zeros(ca.shape + (3,), np.uint8)
    vis[both] = 60
    x, y = (ca == mk) & both, (cb == mk) & both
    vis[x & y] = 255
    vis[x & ~y] = (40, 40, 230)
    vis[y & ~x] = (230, 120, 40)
    cv2.imwrite(f"{a.out}.png", np.rot90(vis))


if __name__ == "__main__":
    main()
