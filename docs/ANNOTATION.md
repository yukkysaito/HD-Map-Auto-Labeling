# Review and acceptance

Who does what:
- **Review (an AI agent)**: decides, item by item, whether the extraction is right (is an outline on real paint / a
  real curb). Its input is the evidence pack of every scene; its decisions become review files. Procedure and criteria:
  [AGENT_REVIEW.md](AGENT_REVIEW.md).
- **Acceptance (a person)**: `output/<name>/debug/audit.html`. It shows whether the batch is ready, the error rate the
  reviewer measured on a random spot check, missed paint, and the evidence of everything the reviewer deleted, found
  off or could not judge, plus random items it accepted (to check the reviewer). The person rejects scenes if needed and
  saves a sign-off (`./autolabel signoff <config> --signoff signoff.json`).

Most errors of the automatic labels come from the segmentation (image semantics), not from the geometry. Examples:
shadows or patched asphalt taken for paint, paint on kerbs, raised vegetation pulling the height. Agreement between two
drives cannot measure this (both have the same error), so the images are checked.

The rest of this page is the manual path: a person reviewing in a browser (`review.html`) or writing review files by
hand, e.g. to set the kind / type of an element, which the agent review does not judge.

## Procedure
0. Pick the scenes: `output/<name>/debug/audit.html` ranks every scene by the risk of the automatic checks (curb vs.
   LiDAR step, vector height, QA failures, time offsets, unobserved boundaries, low-confidence elements, two-drive
   agreement) and draws a reproducible random sample (`audit` in the config: `sample`, `seed`, `max_risk`). Review the
   scenes over `max_risk` and the random sample; the batch is accepted when none is over `max_risk` after the fixes.
1. `./autolabel run <config>` (or `./autolabel debug <config> [--scene ID ...]`)
   -> `output/<name>/debug/scenes/<scene>/`
   - `review.html` (**start here**): open it in a browser. The reviewer judges the **extraction only**: is the
     outline on real paint / a real curb, and does it follow the paint. What kind of marking it is (arrow directions,
     crosswalk, zebra ...) is not asked; those attributes stay as the pipeline set them. The page has three steps:
     1. **Check items**: every road marking / curb with confidence < 0.5, then a random spot check of 10 confident
        ones (fixed per scene; the low-confidence list alone cannot show confident errors). The map zooms to each one.
        Keys: `Space` correct, `X` no paint / curb here (deleted), `1` there, but the outline is clearly off
        (recorded), `←` / `→` previous / skip. Hold `H` to hide the lines and see the paint. A camera sheet is shown
        when one was taken within 40 m.
     2. **Find missing**: move along the road; `P` marks missed paint / curbs or any other problem with a note.
     3. **Finish**: your name, **Accept** or **Reject** (too wrong to fix here), which saves `<scene>.json`.
        Put it at the path shown on the page (step 3 below).
     Unsaved work is kept in the browser and restored when the page is opened again.
   - `sample_<k>.jpg`: labels drawn on every camera image + the BEV (raster + vectors).
     Overlay colours: road green, lane marking red, crosswalk magenta, curb blue, sidewalk yellow.
     Labels are drawn where rays meet the road surface, so they also appear on cars and buildings (occlusion is not
     modelled). That is not an error.
   - `map.jpg`: ortho image of the scene + vectors + element numbers + 10 m grid (map coordinates, labelled every 50 m).
   - `elements.jpg`: every area element (marking_area, crosswalk) cut out of the ortho image (number, area, crop width).
     Optional: for checking element types by hand (tens of elements of a scene in one image).
   - `arrows.jpg`: arrows found automatically (sharp road patch from a nearby camera image + outline + S/L/R/SL/SR + score).
   - `elements.json`: element number -> kind, attributes, a point on the element (map coordinates).
   - `preview_<k>.jpg`: reference camera image + the BEV of the training data (raster + vectors; arrows with outline and directions).
   - `geom_check.jpg` / `.json`: geometric check against the LiDAR map (is there a LiDAR height step at the curb:
     blue = yes, red = no; curb ramps at crosswalks have no step, red is correct there; vector height vs. LiDAR ground).
   - `map_ground.pcd` (LiDAR ground points), `labels.pcd` (labels, rgb + label), `vectors.pcd` (vectors, rgb + kind):
     overlay them in a PCD viewer (e.g. CloudCompare) to check positions and heights in 3D.
2. What to look at
   - Does the outline lie on the real paint? (an offset of everything = time offset / calibration problem: reject)
   - Outlines on things that are not paint (shadows, patched asphalt, tyre marks) or paint without an outline.
   - Curbs: is there a raised edge where the line is?
   - Optional, by hand in the json: the kind / type / arrow directions (`set` fixes below, `elements.jpg`,
     `arrows.jpg`). These attributes are a bonus of the base annotation; a training setup can drop them.
3. Put the result into `review/<name>/<scene>.json` (directory: config `review_dir`), saved by review.html or written
   by hand (format below). Scenes without a file are unreviewed.
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
    {"kind": "lane_line",    "at": [1060.0, 2590.0], "drop": true}
  ]
}
```
- Written by review.html in addition: `checked` (elements confirmed as correct, so a reopened page continues
  where it stopped) and `issues` (`{"at": [x, y], "note": "..."}`, with `kind` when it is about an element whose
  outline is off; for a person to follow up, not applied automatically). `verdict` is `ok` (accepted, nothing
  found), `fixed` (accepted with deletions / issues) or `bad` (rejected).
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
