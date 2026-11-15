# Review (visual check and annotation fixes)

Most errors of the automatic labels come from the segmentation (image semantics), not from the geometry. Examples:
a whole zebra (channelizing) zone becomes "lane marking", a red-and-white crosswalk becomes "lane marking", raised
vegetation pulls the height. Agreement between two drives cannot measure this (both have the same error), so the
images are checked. The reviewer can be a person or a vision-capable LLM agent (it can read the sheets and write the
fix files).

## Procedure
1. `./autolabel run <config>` (or `./autolabel debug <config> [--scene ID ...]`)
   -> `output/<name>/debug/scenes/<scene>/`
   - `sample_<k>.jpg`: labels drawn on every camera image + the BEV (raster + vectors).
     Overlay colours: road green, lane marking red, crosswalk magenta, curb blue, sidewalk yellow, manhole cyan.
     Labels are drawn where rays meet the road surface, so they also appear on cars and buildings (occlusion is not
     modelled). That is not an error.
   - `map.jpg`: ortho image of the scene + vectors + element numbers + 10 m grid (map coordinates, labelled every 50 m).
   - `elements.jpg`: every area element (marking_area, crosswalk) cut out of the ortho image (number, area, crop width).
     **The fastest way to check element types** (tens of elements of a scene in one image).
   - `arrows.jpg`: arrows found automatically (sharp road patch from a nearby camera image + outline + S/L/R/SL/SR + score).
   - `elements.json`: element number -> kind, attributes, a point on the element (map coordinates).
   - `preview_<k>.jpg`: reference camera image + the BEV of the training data (raster + vectors; arrows with outline and directions).
   - `geom_check.jpg` / `.json`: geometric check against the LiDAR map (is there a LiDAR height step at the curb:
     blue = yes, red = no; curb ramps at crosswalks have no step, red is correct there; vector height vs. LiDAR ground).
   - `map_ground.pcd` (LiDAR ground points), `labels.pcd` (labels, rgb + label), `vectors.pcd` (vectors, rgb + kind):
     overlay them in a PCD viewer (e.g. CloudCompare) to check positions and heights in 3D.
2. What to look at
   - In the camera images: does the red (lane marking) lie on the real paint? (offset = time offset / calibration problem)
   - Are zebra zones, red-and-white crosswalks, text and arrows of the right kind?
   - Lines that do not exist (shadows, patched asphalt, tyre marks) or missing lines.
   - In the BEV: are lane lines correctly dashed / solid?
3. Write the result to `review/<name>/<scene>.json` (directory: config `review_dir`). Scenes without a file are unreviewed.
4. `./autolabel run <config>` applies the fixes (only vectorize and later steps are redone).

## review/<name>/<scene>.json
```json
{
  "reviewer": "alice",
  "date": "2026-01-01",
  "verdict": "ok",
  "notes": "curb on the cam10 side slightly too wide 20 m ahead, otherwise good",
  "reject": false,
  "reject_samples": [1793863477914273793],
  "fixes": [
    {"kind": "marking_area", "at": [1023.4, 2567.8], "set": {"type": "zebra"}},
    {"kind": "lane_line",    "at": [1050.1, 2580.2], "drop": true, "why": "shadow"},
    {"kind": "lane_line",    "at": [1060.0, 2590.0], "set": {"style": "dashed"}}
  ]
}
```
- `reject: true`: every sample of the scene gets `qa.ok = false` (excluded from training). `reject_samples`: single samples (stamps).
- `fixes`: `at` is in map coordinates (the `at` of `elements.json` can be used as is). The fix applies to the element
  of that kind nearest to `at` (within 1.5 m). `drop: true` deletes it, `set` overwrites attributes (`reviewed: true`
  is added). `"set": {"kind": "crosswalk"}` changes the kind itself (e.g. a red-and-white crosswalk segmented as
  lane marking).
- **Fixes are in map coordinates, so they also apply to other scenes (other drives) showing the same place** (all
  review files of the output are read). One review covers every later drive of the route. Vectors of another drive
  differ slightly, so another element of the same kind within 1.5 m may be hit instead.
- Example `type` values of `marking_area`: `zebra` (channelizing zone), `stop_line`, `arrow` (arrows, bicycle lane
  chevrons), `text` (road text), `parking` (parking bays), `double_line` (a yellow double line merged into an area),
  `diamond` (crosswalk ahead), `other` (unreadable fragments). Crosswalk `type`: `red_white`.
- Arrows (`kind: "arrow"`): check the directions in `arrows.jpg` (tip up, driver's left on the left; number n is
  `A<n>` in `map.jpg`). Fix a wrong one with
  `{"kind": "arrow", "at": [x, y], "set": {"directions": ["straight", "right"], "shape": "SR"}}`;
  drop non-arrows (e.g. bicycle lane arrows) with `"drop": true`.
  Missed arrows (e.g. hidden by a car) can be added:
  `{"kind": "arrow", "add": {"tail": [x, y], "tip": [x, y], "directions": ["right"]}}` (map coordinates, read from the
  10 m grid of `map.jpg`).
  - Added arrows are measured again (outline, length) from a sharp road patch around the given axis.
  - Without `directions` the directions come from template matching; with it, from the given value.
  - In `arrows.json` they are recorded with `auto: false`. The `at` of an arrow in `elements.json` is the midpoint of tail and tip.
- The label raster itself is not edited. If the raster is badly wrong, reject the scene / samples.

## Effort
A few minutes per scene (one `elements.jpg` + one or two `sample` images). Fixes are in map coordinates, so they stay
valid when the bags are ingested again and drive / scene names change. `elements.jpg` only shows elements not yet
reviewed.

## Policy
- Not every scene needs a review. Start with a few scenes per area (turns, intersections and slopes first) to find
  systematic errors, and fix those in the parameters / processing (add them to the known issues in `docs/MAINTENANCE.md`).
- Review files are manual work: keep them under version control (`output/` can be regenerated, reviews cannot). They
  are site data, so they do not belong in this repository (`review/` is git-ignored); point `review_dir` at your own
  repository.
