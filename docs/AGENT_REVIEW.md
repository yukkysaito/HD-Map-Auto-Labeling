# Review by an AI agent

`./autolabel run` does this automatically with the verifier agent (docs/AGENTS.md; its criteria are
`hdlabel/agents/prompts/verifier.md`). This page describes the evidence pack and the decisions file, which an agent
session or a person can also write by hand.

The review of the extraction is done by an AI agent with image input (e.g. Claude Code); a person only accepts the
result on the acceptance page (`debug/audit.html`). The agent judges **the extraction only**: is an outline on real
paint / a real curb, and is it where the paint is. What kind of marking it is (arrow directions, crosswalk, zebra,
...) is not judged.

## Loop
```bash
./autolabel run <config>                     # labels + evidence packs (debug/scenes/<scene>/check/)
./autolabel review <config>                  # scenes waiting for a review
# per scene: read check/sheet_*.jpg and check/tile_*.jpg, write decisions.json (below)
./autolabel review <config> --scene <scene> --decisions decisions.json   # -> review/<name>/<scene>.json
./autolabel run <config>                     # applies the deletions, rebuilds the data and debug/audit.html
./autolabel review <config>                  # until nothing is waiting
```
The person opens `debug/audit.html`, looks at what the agent flagged, rejects scenes if needed and saves a sign-off;
`./autolabel signoff <config> --signoff signoff.json` records it (rejected scenes leave the training data on the next
run).

## Evidence pack (`debug/scenes/<scene>/check/`, written by `hdlabel/evidence.py`)
- `items.json`: the elements to decide.
  - Every road marking / curb with confidence < 0.5.
  - The first time a scene is reviewed, also 10 random confident ones (`random spot check`, fixed per scene). They
    estimate the error rate of what is not checked.
- `sheet_<k>.jpg`: 5 items each. Every row has 6 panels, all centred on the item:
  - `map`: the ortho image of the fused labels, north up, scale in the label (e.g. "map 8 m").
  - `map + outline`: the same with the item's outline (road marking yellow, curb blue).
  - `camN d m` + `camera + outline`, twice: the two camera frames that see the item best (from another camera or
    another moment, so that one blocked or blurred view does not hide it; distance in the label), without | with the
    outline projected onto the road surface. `no camera view` when no frame sees it.
- `tile_<n>.jpg`: 4 random 30 m map squares of the scene, without | with every outline, for finding missed paint.

## How to decide an item
| decision | when |
|---|---|
| `ok` | The outline lies on paint on the road surface (any kind: line, arrow, text, stripes, crosswalk) and follows it within about 0.3 m. For a curb: there is a raised edge (kerb) along the line. |
| `drop` | No paint there: shadow, asphalt patch or seam, tyre marks, wet patch, manhole, a joint between road and pavement, paint on a kerb face / wall / pole (not the road surface), a car. For a curb: no raised edge (flat transition, painted line only, wall of a building). |
| `off` | The paint / curb is there, but the outline is clearly beside it (more than ~0.5 m) or covers a clearly wrong extent. |
| `unsure` | The evidence does not show it (blurred, occluded, outside the observed area in both views). A person looks at it. |

- Use both views. The map is sharp in the middle of the road and blurred at its edges; the camera is sharp up
  close. The camera frame may show a car or pole in front of the item: then judge from the map.
- Labels near the edge of the observed area (black in the map) are often fragments; decide on what is visible.
- A short piece of a real line (e.g. a dashed line segment) is `ok`.
- Road markings are paint on the carriageway (including bicycle lanes on it). Paint on a sidewalk or a plaza for
  pedestrians is `drop`.
- A squiggle on plain or worn asphalt that follows no clear paint edge is `drop`, even where faded paint is nearby:
  it is not a usable outline.

## How to decide a tile
`ok`, or `missed: <what and where>` when the left panel shows clear paint on the road surface (at least ~1 m long) or a
clear kerb that the right panel has no outline for. Black (unobserved) areas and paint on kerbs do not count.

## Scene verdict
`reject` only for a systematic failure of the scene, which the deletions cannot fix:
- outlines shifted against the paint over a large part of the scene (calibration / time offset),
- or 5 or more of the 10 spot-checked items wrong,
- or missed paint / curbs totalling more than ~20 m in different places (count a miss seen in several tiles once).

Otherwise `accept`. Scattered wrong items are expected: the deletions fix the checked ones, and the spot checks of all
scenes together measure the error of the rest (the acceptance page compares it with `audit.max_spot_error`).

## decisions.json
```json
{"reviewer": "agent:<model>",
 "items": {"0": "ok", "1": "drop: shadow of a pole", "2": "off: outline 0.6 m left of the line", "3": "unsure: occluded"},
 "tiles": {"0": "ok", "1": "missed: dashed line along the right lane not outlined"},
 "verdict": "accept",
 "notes": "one short sentence on the scene"}
```
Every item and tile of `items.json` needs a decision (the command refuses otherwise; `--partial` overrides). Notes
after `:` are shown to the person.

What happens to the decisions (`hdlabel/agent_review.py`):
- `drop` becomes a deletion fix: `vectorize` removes the element on the next run.
- `off`, `unsure` and `missed` become `issues`: the person sees them on the acceptance page.
- Every decision is recorded with its evidence image (`debug/scenes/<scene>/reviewed/`).
