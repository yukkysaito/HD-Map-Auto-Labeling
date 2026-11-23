# Maintenance notes (for maintainers / reviewers)

## Code structure
| File | Role |
|---|---|
| `autolabel` | Entry point (`./autolabel <cmd> [config.yaml] [--map ...] [--logs ...]`, `./autolabel test`). Drops a ROS PYTHONPATH and runs in `.venv` (else `python3`) |
| `configs/default.yaml` | Defaults of every setting (topics, cameras, scenes, dataset, parallelism). Overridden by a project config (`configs/example.yaml`) and the command line |
| `hdlabel/run.py` | Pipeline control:<br>- parallel execution (GPU steps limited by a semaphore)<br>- resume and failure isolation (`status.json`)<br>- scene splitting, choice of drives on the same road level<br>- train / val split<br>- `dataset.json` |
| `hdlabel/common.py` | Class definitions (segmentation ids -> labels, overridable by `segmentation.class_map`), pose interpolation, extrinsics, camera mounting files (`load_tf_static`), stop line rule, device choice, source bag tracing, BEV debug drawing |
| `hdlabel/surface.py` | Point cloud map I/O (PCD) and the multi-level road surface (height layers + connectivity Dijkstra) |
| `hdlabel/mapproj.py` | Projection onto the surface (GPU): ray / height field intersection, voting, drawing labels into camera images, ground snapping and other-level detection |
| `hdlabel/arrowshape.py` | Arrow paint extraction and template matching (`hdlabel/assets/arrow_templates_<set>.npz`, `markings.arrow_templates`) |
| `hdlabel/bev.py` | Ego-frame BEV raster and vectors from the map-frame labels (shared by export, loader and review) |
| `hdlabel/loader.py` | Reference training Dataset (PyTorch) |
| `hdlabel/validate.py` | `./autolabel validate`: acceptance check of an exported dataset (training side) |
| `hdlabel/audit.py` | Acceptance: per-scene risk from the automatic checks and the review, ranking, random sample, the acceptance page (`debug/audit.json`, `audit.html`) |
| `hdlabel/evidence.py` | Evidence packs for the reviewer (`debug/scenes/<scene>/check/`: map and camera crops per item, map tiles) |
| `hdlabel/agent_review.py` | Reviewer decisions -> review files; a person's sign-off -> rejected scenes |
| `hdlabel/review_html.py` | Which elements are checked (low confidence + spot check), the manual review page (`review.html`) |
| `hdlabel/agents/` | Agent layer (docs/AGENTS.md): `llm` (providers, cache, ledger, budget), `verify` (verifier + scene states), `bench` (locked reference, metrics, gate, verifier accuracy), `experiment` (candidates in isolation), `registry` (`tools/registry.yaml`), `knowledge` (SQLite FTS), `orchestrator` (evolve), `training` (model loop), `cli`, `prompts/` |
| `tools/registry.yaml` | The tools, their variants, the active one and its adoption history |
| `hdlabel/steps/*.py` | The processing steps (each a `main(argv)`, runs standalone, see `-h`). Steps are connected only through files |
| `hdlabel/eval/` | `compare_labels` / `compare_vectors`: repeatability between two drives (`./autolabel qa`).<br>`paint_fidelity`: how well the marking class follows the paint in the ortho image (boundary F-score vs image edges, contrast, dark / missed paint; lines vs areas).<br>`dataset_stats`: dataset statistics.<br>`train_smoke` / `online_eval`: learnability check and per-cycle evaluation |
| `hdlabel/bevnet/` | Reference camera-only BEV segmentation network (docs/DATASET.md "Reference model"): `model`, `data`, `losses`, `metrics`, `train`, `evaluate` and `video` (`python -m hdlabel.bevnet.train` / `.evaluate` / `.video`) |
| `tests/` | Unit tests + `test_pipeline_synthetic.py`: the whole pipeline on synthetic data (`tests/synthetic.py`) on CPU, labels checked against the synthetic world (~1 min) |
| `pyproject.toml`, `.github/workflows/ci.yml`, `docker/` | Package, ruff / pytest settings, CI (lint + tests on CPU for every push), GPU image |

Conventions
- Label ids and the palette are defined only in `common.CLASSES` / `PALETTE`.
- Transforms to the ego frame only go through `bev.SceneMap` (no second implementation).
- When the data format changes, update `docs/DATASET.md` and the `format` of `dataset.json`.
- Code is formatted with `ruff format` (line length 120) and must pass `ruff check`; CI enforces it.
- Settings go to `configs/default.yaml`, not into code. No site, vehicle or log paths in the repository
  (`configs/local/` and `review/` are git-ignored).

### Resume / redo (`step()` in `run.py`)
A step is skipped when
- all its outputs exist
- the outputs are newer than the inputs
- its arguments and code equal those of the last successful run (`work/stamps/<tag>.json`; code = the step script,
  the `hdlabel` modules it imports, transitively, and `hdlabel/assets/`; see `run.code_version`).

When settings or inputs change (new drives, calibration, review fixes) only the affected steps are redone. A failed
step loses its stamp and runs again on the next `run`. A failed drive only affects its own scenes. Steps that may
rewrite an input of later steps write a file only when its content changes (e.g. `ingest.py`, the review files,
`work/fixes.json` = the fixes of all review files, which is what vectorize reads), so a rerun does not cascade. Paths
under the output root are replaced by `<OUT>` in the key, so experiments (`output/<name>/experiments/<id>/`) reuse the
stamps of unchanged steps. `./autolabel impact` lists what a change would redo without running anything.

## Checks after a change
1. `./autolabel test` passes (also with `HDLABEL_DEVICE=cpu`), `ruff check . && ruff format --check .` and
   `pyflakes hdlabel tests` are clean. CI runs the same.
2. End-to-end test (about 3 min): one ~1-minute bag plus two broken bags. Example config (e.g. `configs/local/e2e.yaml`,
   not versioned):
   ```yaml
   name: _e2e_test
   pointcloud_map: /path/to/pointcloud_map.pcd
   drives:
     good: {bags: /path/to/one_minute.mcap}
     truncated: {bags: output/_e2e_bags/truncated.mcap}   # head -c 420000000 <another bag> > output/_e2e_bags/truncated.mcap
     empty: {bags: output/_e2e_bags/empty.mcap}           # : > output/_e2e_bags/empty.mcap
   scenes: {length_m: 200, context_m: 60, corridor_m: 40}
   calib: {scenes: 1}
   ```
   Afterwards `rm -rf output/_e2e_test output/_e2e_bags`.
   - the good bag: training data and debug output are produced. It is short, so `coarse` time offsets are expected.
   - the truncated bag: ingested up to the break, recorded in `errors` of `ingest.json`.
   - the empty bag: only this drive fails; everything else finishes and `status` shows the failure and its log (exit code 1).
3. On a full set of your logs (`./autolabel run <config>`):
   - `status` shows no failure.
   - the curb / LiDAR step agreement in `debug/scenes/*/geom_check.json` has not dropped (baseline below).
   - arrow directions in `arrows.jpg` are right.
4. Repeatability: `./autolabel qa <config>`; the agreement between two drives has not dropped (`debug/qa/summary.json`).
5. Learnability: `.venv/bin/python hdlabel/eval/train_smoke.py output/<name>/dataset/dataset.json`; the val IoU of
   every class beats the prior. Always run it after changing the data format or coordinate frames.
6. Scale test (for large changes): list 1-minute bag files in `logs` (`["/path/to/logs/*/*.mcap"]`, one bag = one
   drive) and run with `debug: false`, `prune_frames: true`.

## Baselines (development sample data: 2 drives of the same route, ~30 min each, 10 cameras, 2026-11-15)
Processing, samples, time offsets and label sharing are from the 20-scene layout (300 m grid); the other rows were
measured with the earlier 12-scene layout (a cut every 300 m driven).
| Item | Value |
|---|---|
| Processing | every step succeeds (calib 2; of 20 scenes 9 labelled, 11 shared; export 20; debug 20). 35 min including calibration |
| Samples | 3960 of 4032 pass QA (train 1389, val 2571) |
| Time offsets | cam8 of one drive +1.0 s, the others 2-52 ms. Fine estimates for cam5 / cam6 / cam8, the others `reference`.<br>Changing the scenes used for the estimate moves `reference` cameras by up to 12 ms (the error of this method; ~12 cm at 10 m/s) |
| Curbs vs. LiDAR steps | agreement 0.71-0.82 (`geom_check.json`) |
| Vector height vs. LiDAR ground | median difference 0.7-1.5 cm, p95 7-8 cm |
| Repeatability between 2 drives (2 scenes) | agreement 0.93 / 0.94, lane marking F 0.92 / 0.92, crosswalk F 0.97 / 0.97, curb F 0.96 / 0.95, lane line vector point F 0.85 / 0.92 |
| Rerun repeatability | labels (GPU projection), vectors, time offsets and training data are byte-identical |
| Arrows | 134 in one drive, all checked in `arrows.jpg`; one false positive removed by review. Same count in both drives |
| Missed arrows (1 scene) | all 200 candidate patches checked (`--debug-candidates`): no vehicle arrow among the 165 rejected ones (bicycle lane chevrons are excluded on purpose). Remaining misses: arrows never appearing in the label map (e.g. hidden by a car the whole time) |
| Overpasses (real drives) | 8 horizontal structures (decks, elevated roads) 5.4-10.1 m above the trajectories were extracted from the map (5 directly above).<br>In every scene containing them the surface at the structure stays within 0.61 m (a sidewalk step) of the driven surface: the structures do not mix in |
| Overpasses (real map) | checked at a pedestrian deck 5.4 m directly above the trajectory (595 m²).<br>- real ground trajectory: the surface under the deck is the ground (8.7 m), the deck (13.9 m) does not mix in<br>- virtual trajectory on the deck: all 52,797 surface cells at 13.3-14.2 m, no cell at ground height<br>- a virtual drive at the same XY 6 m higher is excluded from fusion by `scene_drives` |
| Slopes / underpass (real map, virtual drives) | No logged drive climbs a slope (max grade 2.8%), so virtual drives were traced on sloped roads found in the point cloud map (z of a vehicle following the road surface), then the surface was built from them and cropped like a sample.<br>- underpass: 170 m, +2.5 m -> -5.6 m (~5%), passing under a crossing road; half of the cells near the route have another road level above in the LiDAR, none of them took it (0 cells > 1 m off the route level); surface vs. route median 2.9 cm, BEV crop height at the route points ahead vs. their true base_link z: median 3.2 cm, p95 21 cm<br>- rising road over the same junction: 194 m, 4.5% (p95); surface vs. route 1.6 cm, BEV 1.7 cm (p95 12 cm), no other level mixed in<br>Labels (camera side) on slopes are still unverified with real images |
| Stop lines | 8 distinct stop lines detected in the sample scenes, all 8 correct (checked on the ortho images). On the perpendicular-line candidates, the width / length / angle rule kept 7 of the 8 real stop lines and none of the 6 false ones (guide dashes, fragments) |
| Lane-line fragments | AI review of the evidence packs (10 scenes of one drive, 2026-11-16): 43 of 70 checked lane lines were not paint (shadows, joints, kerb edges, worn asphalt), nearly all short or low-confidence; 3 of 10 scenes failed the spot-check rule. `markings.fragments` (confidence < 0.5, or < 2.2 m and < 20 views) removes 42 of the 43 and 10 of the 27 correct ones: 40 % of the lane-line elements, 3.8 % of their length. Thresholds fitted on that review; re-check them with the next review |
| Learnability | `train_smoke.py --scenes <one scene> --eval-split train --cell 0.4 --steps 2000 --bev -20 60 -30 30 0.2` (6 min).<br>- loss 1.00 -> 0.15<br>- IoU on the training samples: road 0.94, crosswalk 0.92, sidewalk 0.92, offroad 0.88, curb 0.75, marking 0.59 (thin lines are coarse in 0.4 m cells)<br>- `train_smoke_overfit_examples.png`: crosswalks, curbs, lane lines and arrows appear at the same place with the same shape<br>- all train scenes (1382 samples): `--eval-split train --cell 0.4 --steps 8000`, then `--init <that .pt> --steps 10000` (2.5 h in total). loss 1.62 -> 0.60, IoU road 0.75, crosswalk 0.69, sidewalk 0.64, offroad 0.61, curb 0.52, marking 0.41; far above the prior for every class and still rising (the small network converges slowly) |
| BEV observed fraction (110-150 m ahead) | about 13% |
| Time (RTX 5070) | - calibration: 13 min with 10 cameras, 3 min with 3 cameras (200 frames per camera instead of 400: offsets within 4 ms, same confidence)<br>- label to debug (12 scenes): about 25 min; arrows 30-60 s per scene<br>- applying review fixes (vectorize and later): 6 min |

Scale test (2026-11-15): 17 one-minute bags, each as its own drive; no debug output, unused frames pruned.
- Result: one command, 85 min; ingest 17, calibration 17 and 18 scenes all succeeded.
- Samples: 1033 of 1789 pass QA. The failures are drives too short (1 min) to turn enough for the time offset estimate;
  they are excluded on purpose.
- Disk: pruning freed 12.6 GB. Debug output was only logs and statistics (656 KB).

## How the geometry is verified (not only by learnability)
- **Independent of the cameras** (`geom_check.py`)
  - is there a LiDAR height step at the curb vectors?
  - difference between vector heights and LiDAR ground points.
  - the point cloud map has zero intensity and is thinned to ~0.6 m spacing, so lane markings cannot be checked with
    LiDAR directly; curb steps are used instead.
- **Against the camera images** (`sample_*.jpg` of `review.py`): labels drawn back into every camera image; lane
  lines and curbs must lie on the real paint. An offset means a time offset, calibration or localization problem.
- **Repeatability** (`./autolabel qa`): labels of the same place made from another drive's images alone are compared.
- **3D** (`*.pcd`): LiDAR ground points, labels and vectors overlaid in a PCD viewer, to see heights and level mixing.

## Pitfalls (recurring)
- **Orientation conventions**
  - BEV: row 0 = ahead, column 0 = left.
  - Arrow patches: rows along the axis, columns to the left of the axis. Templates: tip at the last row, driver's left
    at the last column.
  - `arrows.jpg` is rotated: tip up, driver's left on the left.
  - Do not judge directions from camera images alone (perspective skews them).
- **Cell centres**: cell (i, j) is centred at `lo + (i + 0.5) * res`. Easy to get half a cell off when building
  height fields in tests.
- **Overpasses**
  - Surface layers are chosen by connectivity from the trajectory. Choosing by a height band picks up distant
    elevated roads; do not do it.
  - Only drives with points on the same level are fused (`scene_drives`).
  - Frames taken while the vehicle was on another level (localization z more than 1 m from the surface) are not used
    for projection or arrows (`Camera.on_level`).
  - A scene whose own route crosses itself vertically (spiral ramps) is not supported; use shorter scenes.
- **Arrow candidates**
  - Arrows touching zebra zones or red-and-white pavement become part of a big blob in the label map; wide parts are
    cut off before taking candidates.
  - In patches, saturated (e.g. red) pavement and the border of the rendered area are excluded from paint.
  - Only original candidates may spawn new candidates for arrows next to them in the patch. If spawned candidates may
    spawn too, neighbouring arrows keep re-adding each other with slightly shifted centres and the step never ends
    (one scene ran for an hour). `arrows.py` prints progress every 50 candidates.
  - Template matching scores all variants at half resolution first and re-scores the best 16 at full resolution.
    Half resolution alone mistakes a reversed right-turn arrow for a straight one.
  - Mirrored, reversed and scaled variants are generated from the templates.
  - New shapes (three-way, U-turn, ...) are cut from `arrow_cand_<n>.npz` of `arrows.py --debug DIR --debug-candidates`
    and added to `arrow_templates.npz` as `<class>_<name>`.
- **Time offset window**: use the scene itself (`t_range`). The route extended by 150 m for labelling (`route_t`) adds
  frames that do not turn and blunts the curve; cam6 / cam8 then fell below the acceptance threshold and the other
  cameras were aligned to cam5 alone, 10-30 ms too large (happened on 2026-11-14).
- **Time offsets**: short drives (1 min) do not turn enough and end `coarse`. Cameras whose estimate failed are not
  fused, and the samples of that drive fail QA.
- **Ray intersection**: rays grazing an uphill slope may not converge to a fixed point. Non-converged (`conv`) and
  backward (t <= 0) rays are dropped; covered by a test.
- **Repeatability of GPU sums**: `index_add_` with repeated indices adds in a different order on every GPU run. In
  float32 the last bits vary, votes change and labels changed by single pixels. Order-dependent sums are taken in
  float64 and cast back (`mapproj.project`). Time offsets and all other steps were checked to be byte-identical on rerun.
- **ROS environments**: a shell with ROS sourced puts ROS Python packages on PYTHONPATH, which breaks pytest and
  others. `./autolabel` drops PYTHONPATH.

## Known limitations / future work
- **Systematic segmentation errors**: zebra zones and red-and-white crosswalks become lane markings, etc. Fixed by
  review; the real fix is another model or fine-tuning (review results can serve as ground truth).
- **Missed arrows**: an arrow hidden by a car the whole time is never seen by a camera and cannot be detected; it
  stays as lane line fragments. Adding it with a review `add` (docs/ANNOTATION.md) measures it from a sharp patch.
  Verified by disabling automatic detection (acceptance score 0.99) and adding one known right-turn arrow: exactly that
  arrow came out as a right turn, tail and tip within 5 cm of the automatic detection.
- **Stop lines** are found only when the paint is a thin line (<= `--line-width`) across the lanes; very wide stop bars stay `marking_area`. Other area types (zebra, text, parking) are only set by review: geometry alone separated zebra zones at precision ~0.8 / recall ~0.7 on the reviewed elements, too low to set automatically (image-based classification would be needed).
- **Repeated drives of a route**: a scene reuses the labels of an earlier scene without computing them when
  - it fuses the same set of drives, and
  - its label box lies inside the earlier scene's box, with a tolerance of `scenes.share_tolerance_m` (default 20 m).

  Scenes are cut on a fixed map grid (`length_m` squares, `run.tile_runs`), so drives starting at different places
  get the same scenes on the same road. In the sample data 11 of 20 scenes were shared (including a drive returning
  to the same place). Samples (images, poses) are still made per drive. A scene whose box sticks out of the earlier
  box by more than the tolerance (e.g. a drive turning another way after the same cell) is not shared.
- **Slopes and elevated roads with real logs**: the geometry was checked on the real map with virtual drives (baselines), but no
  logged drive up a slope or on top of an elevated road exists yet, so labels from camera images there are unverified.
