# HD-Map-Auto-Labeling

Automatic generation of HD-map training data from rosbags.

From rosbags (cameras, localization, the vehicle's official calibration) and a LiDAR point cloud map, this tool builds
ground truth for online HD-map models that take multi-camera, multi-frame input:
- raster labels (road, lane markings, crosswalks, curbs, sidewalks, ...)
- 3D vectors (lane lines, stop lines, crosswalks, curbs, road boundaries, **turn arrows with their allowed directions**)

**No HD map (lanelet2 / vector map) is used.** No flat-ground assumption either: slopes, overpasses and roads under
elevated structures are handled.

The labelling tools are driven by AI agents (Claude, ChatGPT, ... interchangeable): an independent verifier checks every
scene against the sensor evidence and the labels are revised until they pass; recurring problems send the agents to
research better methods, implement them as new tool variants and adopt them only if a locked benchmark shows they are
better ([docs/AGENTS.md](docs/AGENTS.md)). A person only accepts the result.

## Requirements
- Input
  - rosbags (mcap, ROS 2): camera images (CompressedImage) + camera_info, localization (Odometry of base_link in the map
    frame), /tf_static. Vehicle speed (VelocityReport) is optional. Topic names and cameras are set in
    `configs/default.yaml` (defaults: Autoware names).
  - a LiDAR point cloud map (.pcd) in the same map frame as the localization.
- Environment: Linux, an NVIDIA GPU with CUDA (tested with 12 GB), Python >= 3.10. Setup: [env/README.md](env/README.md)
  (uv, pip or Docker). The segmentation model is downloaded from Hugging Face on first use; **check its license**
  (below).
- Road marking conventions default to **Japan** (arrow shapes, stop line width): see `markings` in `configs/default.yaml`.

## Quick start (synthetic data, CPU, ~1 min)
```bash
python tests/synthetic.py /tmp/hdlabel_demo        # a synthetic bag + point cloud map + config (no real data needed)
HDLABEL_DEVICE=cpu ./autolabel run /tmp/hdlabel_demo/config.yaml
./autolabel validate /tmp/hdlabel_demo/synthetic/dataset/dataset.json
# open /tmp/hdlabel_demo/synthetic/debug/audit.html and debug/scenes/*/review.html in a browser
```

## Usage
```bash
./autolabel run --map /data/pointcloud_map.pcd --logs "/data/logs/*" --name my_area   # bags -> training data + debug output
./autolabel status --name my_area                                                      # what is done / failed, QA, audit
./autolabel validate output/my_area/dataset/dataset.json                               # acceptance check of a dataset
./autolabel test                                                                       # tests (incl. a synthetic end-to-end run)
./autolabel evolve my_area.yaml                                                        # agents: next improvement (docs/AGENTS.md)
```
- `--logs`: globs. A directory (all `*.mcap` in it) or a single `.mcap` file is one drive.
- Topics: `topics` (templates with the camera id) and per camera `cameras.<id>.image / camera_info` in the config.
- Bags without /tf_static: pass the camera mounting with `--tf-static` (.json, .yaml static transforms, or a bag that has
  /tf_static; see `configs/default.yaml`). Hardware-synchronised cameras: `calib: {time_offset: 0}` skips the estimate.
- For repeated runs put the settings in a config: copy `configs/example.yaml`, edit it, `./autolabel run my_area.yaml`.
  Only the keys you write override `configs/default.yaml`; command-line options override both.
- **Adding logs**: put new logs where `logs` points and `run` again.
- **Resume / redo**: finished work is skipped; only what changed (inputs, settings) or failed is redone. A failing bag
  or scene does not stop the others (exit code 1, `status` lists the failures and their logs).
- **Subsets**: `--drive` / `--scene` restrict the work, `--force` redoes it.
- **Large runs**: `--no-debug` (or `debug: false`) disables debug output, `prune_frames: true` saves disk space,
  `jobs` sets the parallelism (progress and an estimate of the remaining time are printed per stage).
- **Verification and acceptance**: `run` verifies every scene with the verifier agent (`agents` / `verification` in the
  config; without a model the scenes stay `uncertain`). The agents run on a Claude / ChatGPT subscription through the
  `claude` and `codex` CLIs by default, or on API keys with `agents.plan: api` (docs/AGENTS.md). The training data lists only accepted scenes. A person
  accepts on `debug/audit.html` and `./autolabel signoff` (docs/ANNOTATION.md); `validate` checks a dataset on the
  training side (files, images, poses, references, labels vs. vectors).

## Output (`output/<name>/`, change with `--output` / `output`)
```
dataset/                    training data only (docs/DATASET.md); this directory is all a training machine needs
  dataset.json              entry point: classes, vector classes, default BEV, scenes (train / val split by location), source bags
  <scene>/map.npz           labels (map-frame raster, 0.1 m, + road surface height)
  <scene>/vectors.geojson   3D vectors (map frame)
  <scene>/samples.jsonl     samples: time, ego pose, current / past frames per camera, QA, source bag and time
  <scene>/frames.jsonl, images/cam<N>/  input images (undistorted) and their poses
debug/                      for checking (may be deleted / disabled)
  scenes/<scene>/           review sheets (labels drawn on the camera images, map, arrow list), previews,
                            PCDs (LiDAR ground points, labels, vectors), geometric check against the LiDAR map
  scenes/<scene>/check/     evidence pack for the reviewer (items.json, sheet_*.jpg, tile_*.jpg)
  scenes/<scene>/reviewed/  evidence of the decided items (shown on the acceptance page)
  scenes/<scene>/review.html  manual review in a browser (optional)
  scenes/<scene>/verification.json  state (accepted / needs_revision / rejected / uncertain) and issues
  audit.json|html           the acceptance page: readiness, risk ranking, review results, evidence, sign-off
  logs/, dataset_stats.json, qa/, eval/
work/                       intermediate data for resuming (per drive: images, poses, segmentation, calibration; per scene: surface, labels)
status.json                 success / failure, notes and log of every work item
agent/                      agent call cache and cost ledger;  experiments/<id>/  candidate tool runs (docs/AGENTS.md)
```
Reference loader for training: `hdlabel/loader.py`. The BEV range and resolution are chosen in the loader (labels are
stored once, in map coordinates).

## How it works (`hdlabel/steps/`)
| Step | What it does |
|---|---|
| `ingest.py` | Reads the bags once:<br>- images scaled (1/2) and undistorted<br>- localization, vehicle speed, /tf_static<br>- time range of every bag (to trace samples back to their bag)<br>- data checks: localization gaps / jumps, dropped camera frames; broken bags or images are used up to the error, which is recorded<br>- keyframes (every 0.5 m) |
| `segment.py` | Semantic segmentation of the keyframes (default: Mask2Former, Mapillary Vistas classes; any model via `segmentation` in the config). |
| scenes (`run.py`) | Drives are cut into scenes at the lines of a fixed 300 m map grid.<br>- labels cover the route +- 150 m and 40 m to the side (ground truth for the whole BEV ahead)<br>- scenes of other drives on the same route share the labels (computed once) |
| `time_offset.py` | Estimates each camera's image time offset: the offset that minimises the smear of lane markings while turning. Extrinsics are the official ones, unchanged. |
| `map_surface.py` | 3D road surface of the vehicle's road level from the point cloud map (`hdlabel/surface.py`):<br>- points are split into height layers<br>- only layers connected to the trajectory by small steps are kept<br>- so under an overpass the surface is the ground, on a bridge the deck; other levels never mix |
| `project_labels.py` | The segmentation of every camera of every drive is cast as rays onto the surface (GPU) and voted into 0.1 m cells.<br>- only drives on the same road level are fused<br>- frames taken while the vehicle was on another level are skipped |
| `arrows.py` | For every arrow a sharp 2 cm road patch is built from a nearby (2.5-11 m) camera image.<br>- matched against templates cut from real arrows (straight, right, straight + left; mirrored / reversed / scaled automatically) to get the heading and allowed directions (`hdlabel/arrowshape.py`)<br>- the raster is redrawn with the measured outline |
| `vectorize.py` | Vectors from the label map: lane lines (solid / dashed, painted segments), stop lines (wide lines across the lanes), crosswalks, curbs, road boundaries (real boundary or edge of the observed area), drivable area, arrows. Applies review fixes (`review/<name>/*.json`, map coordinates). |
| `export.py` | Writes the training data. Samples at the reference camera's times, every 0.5 s (`dataset.every_s`; val optionally denser, `val_every_s`).<br>QA fails a sample when: the near range is unobserved, time offset estimation failed, localization is missing, or a reviewer rejected it. |
| `review.py`, `geom_check.py` | Debug output (`docs/ANNOTATION.md`):<br>- sheets with labels drawn on the camera images, map, arrow list, review.html<br>- curbs vs. LiDAR height steps, vector heights vs. LiDAR ground (checks independent of the cameras)<br>- PCDs |

| Document | Contents |
|---|---|
| [docs/DATASET.md](docs/DATASET.md) | Training data format (frames, raster, vectors, multi-frame input, loader) |
| [docs/AGENTS.md](docs/AGENTS.md) | Agents: verification, data / tool / model loops, protected benchmark, registry, knowledge |
| [docs/AGENT_REVIEW.md](docs/AGENT_REVIEW.md) | Evidence packs, decision criteria and the manual decisions file |
| [docs/ANNOTATION.md](docs/ANNOTATION.md) | Acceptance by a person, manual review and review fix files |
| [docs/MAINTENANCE.md](docs/MAINTENANCE.md) | Code structure, tests and regression checks, known limitations, pitfalls |
| [env/README.md](env/README.md) | Python environment |

All settings are in `configs/default.yaml` (commented). Site-specific paths, calibration files and review results can
live in `configs/local/` and `review/`, which git ignores.

## License
Apache License 2.0 ([LICENSE](LICENSE)).

### Third-party models and data
The labels are made with a semantic segmentation model, so **the terms of that model (and of the data it was trained
on) may apply to the labels you generate**. Check them for your use; this is not legal advice.
- Default model `facebook/mask2former-swin-large-mapillary-vistas-semantic` (Hugging Face; license tag "other"). The
  Mask2Former code is MIT, but its model zoo states that the released weights are licensed under **CC BY-NC 4.0
  (non-commercial)**: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
- It is trained on Mapillary Vistas; check the dataset's own license at https://www.mapillary.com/dataset/vistas.
- To use another model, set `segmentation.model` (a Hugging Face Mask2Former checkpoint) and, if its classes are not
  Vistas', `segmentation.class_map` (its class ids per label group) in your config.
