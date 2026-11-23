# Training data format (format 5)

Format 5: `road_boundary` `observed: false` means "not a boundary to learn" (with an optional `reason`); the marking
and crosswalk raster classes follow the paint. Format 4 datasets are re-exported by `./autolabel run`.

Target model: multi-camera, multi-frame input; every cycle it outputs the HD map around the vehicle (200-300 m),
as a raster (BEV segmentation) or as vectors (MapTR style). Both kinds of ground truth come from the same labels.

**Labels are stored once per scene in map coordinates; samples only carry poses** (the same idea as the nuScenes map).
- Range, resolution and orientation are chosen by the loader at training time; no need to regenerate the data.
- Fine shapes are not limited by a resolution: lane line and arrow outlines are vectors (cm), the raster is 0.1 m.
- No BEV is stored per sample, so the data is small.
- The labels of a scene cover the route where samples are taken (inside one cell of the 300 m grid) +- 150 m and
  40 m to the side: ground truth reaches 150 m ahead.

## Directory (`output/<name>/dataset/`)
```
dataset.json                 entry point (below)
<scene>/
  scene.json                 cameras (K, size, T_base_cam, time offset and its confidence), default BEV, source bags, history settings
  map.npz                    cls (uint8, 0.1 m) / height (float32, 0.2 m road surface height, NaN = no road) / lo / res / surf_res / origin
  vectors.geojson            3D vectors (map frame, m; z = road surface height)
  samples.jsonl              one line per sample
  frames.jsonl               one line per input frame (only the images the samples use)
  levels.npz                 optional (config dataset.levels): other road levels above / below the ego level (below)
  images/cam<N>/<stamp>.jpg  undistorted images (hard links into work/: no extra space; copying makes real files)
```
All paths are relative to the scene directory: `dataset/` can be moved to another machine as is.

## Coordinate frames
- `map`: the Autoware map frame (m), the same as the point cloud map.
- `base_link`: the vehicle (x forward, y left, z up). **z is snapped to the map road surface** (the localization z is
  10-15 cm below the surface; corrected).
- Cameras: OpenCV convention (x right, y down, z forward). `T_base_cam` maps camera to base_link.
- Images: undistorted pinhole (`K`, `size`), half the original resolution.
- Time: ns. `stamp` is the image header time; `pose_stamp = stamp + time_offset_s` is the time the pose is taken at
  (with the estimated time offset).

## dataset.json
`format`, `name`, `cameras` (camera ids), `classes` (raster class names), `vector_classes`, `arrow_directions`,
`bev` (default crop: `x`, `y` [m], `res`), `split` (`tile_m`, `val_every`, sample counts),
`scenes` (`scene`, `drive`, `path`, `samples`, `qa_ok`, `split`, `tile`, `bags`, `t_range`).

## samples.jsonl (one line)
```json
{"stamp": 1793863477914273793,                 // sample time (ns, pose time)
 "T_map_base": [[...4x4...]],                  // ego pose at the sample time (snapped to the surface)
 "frames": {"8": [120, 98, 77, 56, 35], ...},  // per camera, ids in frames.jsonl: [current, 0.5 s earlier, 1.0 s earlier, ...]
 "dt_ms": {"8": 0.0, "5": -24.5, ...},         // pose time of the current frame - sample time
 "qa": {"observed_near": 0.93, "max_dt_ms": 48.0, "pose_ok": true, "calib_ok": true, "ok": true},
 "source": {"drive": "20260101-120000", "bag": "<source bag file name>", "stamp": 1793863477914273793}}
```
- Train only on samples with `qa.ok == true`. A sample fails when
  - less than 50% of the near range (30 m ahead, 8 m to each side) is observed
  - the current frame of a camera is more than 60 ms off
  - localization is missing or jumps there (detected by ingest)
  - time offset estimation failed for a camera of the drive
  - a reviewer rejected it
- History spacing and length are in `scene.json` `history` (default 0.5 s x 4 frames = 2 s).
- `source` traces the sample back to its rosbag file and time; every frame also has `bag`.

## frames.jsonl (one line)
`{"id": 120, "cam": 8, "image": "images/cam8/<stamp>.jpg", "stamp": ..., "pose_stamp": ..., "T_map_base": [[...]], "bag": "..."}`
- `T_map_base`: the ego pose at the frame's pose time.
- To bring features of a past frame into the current ego frame use
  `inv(T_map_base[sample]) @ T_map_base[frame] @ T_base_cam` (the loader's `T_cur_frame`).

## Raster (map.npz -> ego BEV)
Cropped with `SceneMap.crop(T_map_base, x, y, res)` in `hdlabel/bev.py`.
- Grid orientation: row 0 = ahead (largest x), column 0 = left (largest y).
- Cell value: the class where the base_link z axis through the cell meets the 3D road surface (correct on slopes).
- `height`: the base_link z of that point.

The classes are defined in `spec/labels.yaml` (`raster`); `dataset.json` `classes` lists them and `validate` checks
that they match.

| id | class | meaning |
|---|---|---|
| 0 | unobserved | seen by no camera / off the road surface (**excluded from the loss**) |
| 1 | road | roadway (manhole covers included) |
| 2 | marking | lane lines, stop lines, arrows etc. (road markings except crosswalks), following the paint seen in the fused ortho image (letters and hatching as strokes; where it is blurred, the segmentation's shape). Arrows are redrawn with the outline measured in a sharp single frame |
| 3 | crosswalk | crosswalk stripes (the gaps between them are road where the paint is seen; the crosswalk vector is the whole area) |
| 4 | curb | curb |
| 5 | sidewalk | sidewalk |
| 6 | offroad | vegetation, ground and other non-road surfaces |
| 7 | other | (does not occur on the road surface) |

`cls[i, j]` of `map.npz` is at map `x = origin.x + lo.x + (i + 0.5) * res`, `y = origin.y + lo.y + (j + 0.5) * res`.
**The raster only contains the road level the vehicle drives on**: the ground road when driving under an overpass,
the deck on a bridge; levels never mix. A road at another height that is connected to the ego level by drivable slopes
and does not lie over or under it (e.g. a frontage road beside a viaduct, joined by a ramp) is part of the ego level:
`height` tells it apart.

## Other road levels (levels.npz, optional)
Grade-separated places (overpasses, elevated roads and rails, noise enclosures, tunnel ceilings above the road; the
ground below a bridge or viaduct) from the LiDAR point cloud map (`hdlabel/levels.py`). Written by export with
`dataset.levels: true`, or afterwards for an existing dataset:
`python -m hdlabel.levels output/<name>/dataset/dataset.json --map <pointcloud_map.pcd>`. No new labels: other levels
stay unlabelled in `map.npz`; the file only tags where they are.
- `over` / `under` float16, 1 m cells (`res`) from `lo` / `origin` as `map.npz`: dz of the other level above / below
  the ego surface (m), NaN = none. A level is a thin, connected structure >= 150 m2 with >= 4 m clearance (trees,
  signs and poles are not).
- The loader returns `bev_level` (uint8 per BEV cell: bit 1 = a level above, bit 2 = below) and `multilevel` (share of
  the crop's cells with another level; NaN without the file). A sample is multilevel from `loader.MULTILEVEL` (5%).
- `python -m hdlabel.bevnet.evaluate ... --by-level` also scores multilevel and single-level samples and the labelled
  cells under / above another level separately.

## Vectors (vectors.geojson -> ego frame)
`SceneMap.vectors(T_map_base, x, y)` transforms them to base_link and crops them to the range.
- Lines are split where they leave / enter the range; polygons are intersected with it.
- Points are only thinned with a 5 cm tolerance; no shape is lost.

| kind | geometry | main attributes |
|---|---|---|
| `lane_line` | `points` line | one element per painted piece, as on the road: each dash of a dashed line is its own element. `line_id`: pieces of one line (collinear dashes) share it; `style` of that line: solid / dashed / short; `length` of the piece; `painted` (1.0) and `segments` (`[[0, length]]`) kept for the format |
| `arrow` | `points` line (tail -> tip) | `directions`: a set of `straight` / `left` / `right` (e.g. `["straight", "left"]`), `shape`: S / L / R / SL / SR, `outlines` (paint outline rings, 3D). For e.g. right-turn-only lanes |
| `stop_line` | `points` line | `width` (m; `vectors.geojson` also has `length`). A wide (~0.45 m) line across the lanes; before format 3 these were `lane_line` |
| `curb` | `points` line | |
| `curb_candidate` | `points` line | a curb line the curb filter (`markings.curb_filter`, active tool variant) judged not to be a road kerb; `curb_check` = the rules it failed. Not a boundary for training (the loader skips it); the raster keeps its cells as curb |
| `marking_candidate` | as its kind | a lane line / stop line / marking area / arrow the marking gate (`markings.marking_gate`, vectorize variant `marking_level`) found off the road level by the LiDAR ground height (ledge, raised sidewalk, plaza behind a kerb step); `candidate_of` = its kind, `marking_check` = the failed rules, `marking_level` = the measurements. Not a marking for training (the loader skips it) |
| `road_boundary` | `points` line | `observed`: true = reliable boundary (curb / sidewalk / off-road outside, a consistent edge), false = not a boundary to learn (**do not learn it as a boundary**); `reason` (only on false runs, with `markings.boundary_shape`): `edge_of_view` (edge of the observed area) or `no_consistent_edge` (ragged outline of a plaza / planter / wavy segmentation edge) |
| `crosswalk` | `rings` polygon | `type` (set by review, e.g. `red_white`) |
| `marking_area` | `rings` polygon | thick stop lines, text, zebra zones etc. `type` (set by review: zebra / stop_line / text / ...) |
| `drivable_area` | `rings` polygon (with holes) | |

Every marking (lane_line, stop_line, arrow, marking_area), crosswalk and curb element also has `confidence` (mean
share of its class in the camera votes of its cells, 0..1) and `observations` (median number of camera views of its
cells): use them to weight or filter elements in training.

Lane-line fragments that are mostly noise are not in the vectors: confidence < 0.5, or shorter than 2.2 m and seen by
fewer than 20 camera views (config `markings.fragments`). The raster (`map.npz`) is not filtered.

Mapping to MapTR-style classes (the loader's `VECTOR_CLASSES`):
- divider = `lane_line`
- boundary = `curb` + `road_boundary` (observed)
- ped_crossing = `crosswalk`
- arrow = `arrow` (point order = arrow heading, `vec_dir` = allowed directions)
- stop_line = `stop_line`

## Reference loader (`hdlabel/loader.py`)
```python
from hdlabel.loader import HDMapDataset

ds = HDMapDataset(
    "output/<name>/dataset/dataset.json",
    split="train",  # cams=[...] selects cameras and their order (default: all)
    history=3,
    image_size=(704, 448),
    bev=dict(x=[-50, 150], y=[-80, 80], res=0.2),
)
x = ds[0]  # images [T, C, 3, H, W], cam_mask, K, T_base_cam, T_cur_frame [T, C, 4, 4], frame_dt,
# bev_cls / bev_height [nx, ny], vec_pts [V, 20, 3] / vec_cls / vec_mask / vec_dir [V, 3],
# bev_level [nx, ny] / multilevel (other road levels, with levels.npz)
els = ds.item_vectors(0)  # vectors with all attributes (painted segments of lane lines, arrow outlines, polygons)
```
- Some drives lack cameras (not in the bag): their `cam_mask` is false and their images are 0.
- Without `bev` the default crop of `dataset.json` is used.
- `still_every_s=N` keeps one sample per N s while the vehicle stands (waits at lights repeat one view; 0: all).

## Train / val split
Drives pass the same places many times, so a random split would make validation too easy. The split is **by
location**:
- the map is cut into `split.tile_m` tiles (default 250 m); the tiles in `split.test_tiles` (`[[ix, iy], ...]`) are a
  fixed test split, kept out of training and model selection; every 5th other tile (fixed hash) is val.
- a scene's split is the tile of its route centre.
- val / test scenes are sampled like train (`dataset.every_s`) unless `dataset.val_every_s` is set, e.g. 0.1 s = the
  model cycle: then they can be replayed in time order to evaluate every cycle (`hdlabel/eval/online_eval.py`).

## Learnability check (`hdlabel/eval/train_smoke.py`; not model development)
A small BEV network checks that
- images, calibration, poses and labels correspond
- the ground truth can be learned from camera input.

Network: ResNet18 features sampled at the projection of each BEV cell (flat-ground assumption), trained on 0.8 m
cells. Val is compared with "the most frequent train class per cell". Results go to `output/<name>/debug/eval/`.

## Reference model (`hdlabel/bevnet`)
A camera-only BEV segmentation network that shows what the dataset supports (not tuned for deployment):
- input: the current frame of every camera (704 x 448); output: classes on a 0.1 m grid, 60 m ahead, 20 m behind,
  25.6 m to each side
- frozen DINOv2 + ResNet18 image features, lifted to a 0.4 m BEV grid by sampling every cell at several heights
  (SimpleBEV), a BEV U-Net, and a 0.1 m detail head on stride-2 image features
- trained on the labelled cells only; output channel 0 lets the model abstain where cells are never labelled.

```
python -m hdlabel.bevnet.train output/<name>/dataset/dataset.json --tag _mine   # ~70 min on a 12 GB GPU
python -m hdlabel.bevnet.video output/<name>/debug/eval/bevnet_mine.pt          # videos of the val scenes
python -m hdlabel.bevnet.evaluate bevnet_mine.pt --dataset <other>/dataset.json --split all   # e.g. an unseen area
```
Several `dataset.json` (e.g. areas) can be given to `train`; it then also reports the val scores of each.
Scores (`debug/eval/bevnet<tag>.json`) are at the label resolution: IoU per class and a boundary F-score (0.2 m
tolerance) for markings, crosswalks and curbs.
