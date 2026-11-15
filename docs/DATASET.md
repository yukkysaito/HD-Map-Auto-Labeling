# Training data format (format 2)

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

| id | class | meaning |
|---|---|---|
| 0 | unobserved | seen by no camera / off the road surface (**excluded from the loss**) |
| 1 | road | roadway |
| 2 | marking | lane lines, stop lines, arrows etc. (road markings except crosswalks). Arrows are redrawn with the outline measured in a sharp single frame |
| 3 | crosswalk | crosswalk |
| 4 | manhole | manhole |
| 5 | curb | curb |
| 6 | sidewalk | sidewalk |
| 7 | offroad | vegetation, ground and other non-road surfaces |
| 8 | other | (does not occur on the road surface) |

`cls[i, j]` of `map.npz` is at map `x = origin.x + lo.x + (i + 0.5) * res`, `y = origin.y + lo.y + (j + 0.5) * res`.
**The raster only contains the road level the vehicle drives on**: the ground road when driving under an overpass,
the deck on a bridge; levels never mix.

## Vectors (vectors.geojson -> ego frame)
`SceneMap.vectors(T_map_base, x, y)` transforms them to base_link and crops them to the range.
- Lines are split where they leave / enter the range; polygons are intersected with it.
- Points are only thinned with a 5 cm tolerance; no shape is lost.

| kind | geometry | main attributes |
|---|---|---|
| `lane_line` | `points` line | `style`: solid / dashed / short, `painted` (painted fraction), `segments` (painted intervals [s0, s1] m from the start of the visible part), `length` |
| `arrow` | `points` line (tail -> tip) | `directions`: a set of `straight` / `left` / `right` (e.g. `["straight", "left"]`), `shape`: S / L / R / SL / SR, `outlines` (paint outline rings, 3D). For e.g. right-turn-only lanes |
| `curb` | `points` line | |
| `road_boundary` | `points` line | `observed`: true = real boundary (curb / sidewalk / off-road outside), false = edge of the observed area (**do not learn it as a boundary**) |
| `crosswalk` | `rings` polygon | `type` (set by review, e.g. `red_white`) |
| `marking_area` | `rings` polygon | thick stop lines, text, zebra zones etc. `type` (set by review: zebra / stop_line / text / ...) |
| `drivable_area` | `rings` polygon (with holes) | |

Mapping to MapTR-style classes (the loader's `VECTOR_CLASSES`):
- divider = `lane_line`
- boundary = `curb` + `road_boundary` (observed)
- ped_crossing = `crosswalk`
- arrow = `arrow` (point order = arrow heading, `vec_dir` = allowed directions)

## Reference loader (`hdlabel/loader.py`)
```python
from hdlabel.loader import HDMapDataset
ds = HDMapDataset("output/<name>/dataset/dataset.json", split="train",      # cams=[...] selects cameras and their order (default: all)
                  history=3, image_size=(704, 448), bev=dict(x=[-50, 150], y=[-80, 80], res=0.2))
x = ds[0]   # images [T, C, 3, H, W], cam_mask, K, T_base_cam, T_cur_frame [T, C, 4, 4], frame_dt,
            # bev_cls / bev_height [nx, ny], vec_pts [V, 20, 3] / vec_cls / vec_mask / vec_dir [V, 3]
els = ds.item_vectors(0)   # vectors with all attributes (painted segments of lane lines, arrow outlines, polygons)
```
- Some drives lack cameras (not in the bag): their `cam_mask` is false and their images are 0.
- Without `bev` the default crop of `dataset.json` is used.

## Train / val split
Drives pass the same places many times, so a random split would make validation too easy. The split is **by
location**:
- the map is cut into `split.tile_m` tiles (default 250 m); every 5th tile (fixed hash) is val.
- a scene's split is the tile of its route centre.
- val scenes are sampled at the model cycle (`val_every_s`, 0.1 s): replay them in time order to evaluate every
  cycle (`hdlabel/eval/online_eval.py`).

## Learnability check (`hdlabel/eval/train_smoke.py`; not model development)
A small BEV network checks that
- images, calibration, poses and labels correspond
- the ground truth can be learned from camera input.

Network: ResNet18 features sampled at the projection of each BEV cell (flat-ground assumption), trained on 0.8 m
cells. Val is compared with "the most frequent train class per cell". Results go to `output/<name>/debug/eval/`.
