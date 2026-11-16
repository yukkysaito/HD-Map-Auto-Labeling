"""review.html: a self-contained page (no server) that walks a reviewer through one scene in three steps.

The reviewer judges the extraction only: is the outline on real paint / a real curb, and is it where the paint is.
What kind of marking it is (arrow directions, crosswalk, zebra ...) is not asked; those attributes stay as the
pipeline set them (a training setup that does not want them can drop them).

1. Check: the elements a person has to look at: every road marking / curb with low confidence plus a fixed random spot
   check of the others (the low-confidence list alone cannot show confident errors). The map zooms to each one in turn;
   the answer is one key (Space = correct, X = no paint / curb here, 1 = there, but the outline is clearly off).
2. Missing: pan over the scene and flag missed paint / curbs or other problems with a note.
3. Finish: accept or reject the scene, save review/<scene>.json (format of docs/ANNOTATION.md).

The decisions are kept in the browser while the page is open (and as a draft in localStorage); the saved json holds
the fixes (deletions, applied by vectorize.py), the checked elements (so a reopened page continues where it stopped)
and the flagged issues (for a person; not applied automatically).
"""

from __future__ import annotations

import json
import random

import numpy as np

LOW_CONFIDENCE = 0.5
SPOT_CHECK = 10  # confident elements drawn at random per scene (seed: scene name)
GROUP = dict(lane_line="marking", stop_line="marking", arrow="marking", marking_area="marking", crosswalk="marking",
             curb="curb", road_boundary="boundary")  # fmt: skip
COLORS = dict(marking="#ffd600", curb="#3d7bff", boundary="#ff4040")
NAMES = dict(marking="road marking", curb="curb", boundary="road boundary")

# What the reviewer is asked per group: ok (Space) keeps the element, drop (X) deletes it, off (1) flags an outline that
# is clearly off (kept, recorded as an issue for a person).
TASKS = {
    "marking": dict(
        question="Is the yellow outline on paint, and does it follow the painted edges? "
        "What kind of marking it is does not matter.",
        ok="Yes, it is on paint",
        drop="No paint here (shadow, patch, tyre marks)",
        off="Paint is there, but the outline is clearly off",
    ),
    "curb": dict(
        question="Is there a curb (raised edge of the road) along the blue line?",
        ok="Yes, a curb",
        drop="No curb here",
        off="A curb is there, but the line is clearly off",
    ),
}


def element_at(f: dict) -> list[float]:
    """a point on the element (map xy), as vectorize.py matches fixes (nearest element of the kind within 1.5 m)."""
    g, pr = f["geometry"], f["properties"]
    if pr["kind"] == "arrow":
        (x0, y0, *_), (x1, y1, *_) = g["coordinates"][0], g["coordinates"][-1]
        return [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)]
    P = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
    if g["type"] == "Polygon":
        return [round(sum(p[0] for p in P) / len(P), 2), round(sum(p[1] for p in P) / len(P), 2)]
    return [round(P[len(P) // 2][0], 2), round(P[len(P) // 2][1], 2)]


def needs_check(pr: dict) -> bool:
    """low-confidence road markings / curbs not reviewed yet (the others are only shown, or spot-checked)."""
    if GROUP.get(pr["kind"]) not in TASKS or pr.get("reviewed"):
        return False
    return (pr.get("confidence") if pr.get("confidence") is not None else 1.0) < LOW_CONFIDENCE


def _walk(elements: list[dict], idx: list[int]) -> list[int]:
    """nearest-neighbour order, so the map moves little from one element to the next."""
    if not idx:
        return []
    xy = np.array([elements[i]["at"] for i in idx], float)
    cur, left, out = int(np.argmin(xy[:, 0] + xy[:, 1])), set(range(len(idx))), []
    while left:
        left.discard(cur)
        out.append(idx[cur])
        if left:
            rest = np.array(sorted(left))
            cur = int(rest[np.argmin(np.linalg.norm(xy[rest] - xy[cur], axis=1))])
    return out


def check_queue(elements: list[dict], seed: str = "", spot: int = SPOT_CHECK) -> tuple[list[int], dict]:
    """(indices of the elements to check, {index: reason}): low-confidence ones first, then a reproducible random spot
    check of the confident ones."""
    low = [i for i, e in enumerate(elements) if needs_check(e["props"])]
    skip = set(low)
    rest = [
        i
        for i, e in enumerate(elements)
        if GROUP.get(e["kind"]) in TASKS and not e["props"].get("reviewed") and i not in skip
    ]
    picked = sorted(random.Random(seed).sample(rest, min(spot, len(rest))))
    reasons = {i: "low confidence" for i in low} | {i: "random spot check" for i in picked}
    return _walk(elements, low) + _walk(elements, picked), reasons


def elements(features: list) -> list[dict]:
    """the elements shown / checked (all but unobserved road boundaries); src = index in features."""
    els = []
    for i, f in enumerate(features):
        k = f["properties"]["kind"]
        if k not in GROUP or (k == "road_boundary" and not f["properties"].get("observed")):
            continue
        g = f["geometry"]
        rings = g["coordinates"] if g["type"] == "Polygon" else [g["coordinates"]]
        props = {x: v for x, v in f["properties"].items() if x not in ("segments", "outlines")}
        els.append(
            dict(
                kind=k,
                group=GROUP[k],
                closed=g["type"] == "Polygon",
                rings=[[[p[0], p[1]] for p in r] for r in rings],
                outlines=[[[p[0], p[1]] for p in o] for o in f["properties"].get("outlines", [])],
                at=element_at(f),
                props=props,
                src=i,
            )
        )
    return els


def decided(review: dict | None, kind: str, at) -> bool:
    """a review entry (checked / fix / issue) is about this element (same kind, same point)."""
    if not review:
        return False
    entries = review.get("checked", []) + review.get("fixes", []) + review.get("issues", [])
    return any(e.get("kind") == kind and e.get("at") and abs(e["at"][0] - at[0]) + abs(e["at"][1] - at[1]) < 0.02
               for e in entries)  # fmt: skip


def write(path: str, scene: str, ortho: str, shape, lo, res: float, origin, els: list, review: dict | None,
          samples: list, review_path: str | None = None) -> tuple[list[int], dict]:  # fmt: skip
    """els: elements(features); ortho: path of ortho.jpg relative to the page; shape / lo / res / origin: the label
    grid (ortho = rot90 of it); samples: [{file, xy}] camera sheets (relative paths) and the ego position (map xy) of
    each; review_path: where the saved json belongs (shown to the reviewer). Returns check_queue(els)."""
    samples = [s if isinstance(s, dict) else dict(file=s, xy=None) for s in samples]
    queue, reasons = check_queue(els, scene)
    data = dict(scene=scene, ortho=ortho, nx=int(shape[0]), ny=int(shape[1]), lo=[float(v) for v in lo],
                res=float(res), origin=[float(v) for v in origin[:2]], elements=els, queue=queue,
                review=review or {}, samples=samples, review_path=review_path or f"review/<name>/{scene}.json",
                reasons=reasons, colors=COLORS, names=NAMES, tasks=TASKS)  # fmt: skip
    with open(path, "w") as fp:
        fp.write(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return queue, reasons


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Review</title><style>
:root{--bg:#fff;--fg:#1d1d1f;--mute:#6e6e73;--line:#d2d2d7;--acc:#0a66d8;--ok:#1e8e3e;--bad:#d93025;--fix:#e37400;--card:#f5f5f7}
*{box-sizing:border-box} body{margin:0;font:14px/1.45 system-ui,sans-serif;color:var(--fg);background:var(--bg);display:flex;height:100vh}
#mapwrap{flex:1;position:relative;background:#111} #map{width:100%;height:100%;cursor:grab;display:block}
#map.adding{cursor:crosshair} .el{fill:none;stroke-width:2px;vector-effect:non-scaling-stroke;cursor:pointer}
.dim .el{opacity:.25} .dim .el.cur{opacity:1} .el.cur{stroke-width:3px} #els.off,#marks.off{display:none} .halo{fill:none;stroke:#00e5ff;stroke-width:2px;vector-effect:non-scaling-stroke;stroke-dasharray:6 4}
.pin{stroke:#fff;stroke-width:1.5px;vector-effect:non-scaling-stroke}
#legend{position:absolute;left:10px;bottom:10px;background:rgba(0,0,0,.7);color:#fff;padding:8px 10px;border-radius:8px;font-size:12px}
#legend label{display:block;cursor:pointer} #legend i{display:inline-block;width:18px;height:4px;margin:0 6px 3px 4px;vertical-align:middle}
#hint{position:absolute;left:10px;top:10px;background:rgba(0,0,0,.7);color:#fff;padding:6px 10px;border-radius:8px;font-size:12px}
#side{width:400px;display:flex;flex-direction:column;border-left:1px solid var(--line)}
header{padding:10px 14px;border-bottom:1px solid var(--line)} header h1{font-size:15px;margin:0 0 6px}
nav{display:flex;gap:4px} nav button{flex:1;padding:6px 4px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer;font-size:13px}
nav button.on{background:var(--acc);border-color:var(--acc);color:#fff}
#bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden} #bar div{height:100%;background:var(--ok)}
#prog{font-size:12px;color:var(--mute);margin-top:4px}
main{flex:1;overflow:auto;padding:12px 14px} h2{font-size:16px;margin:0 0 4px} .mute{color:var(--mute);font-size:12px}
.q{font-size:15px;font-weight:600;margin:10px 0} .cur-val{background:var(--card);border-radius:6px;padding:6px 8px;margin:6px 0}
.btn{display:flex;align-items:center;width:100%;text-align:left;padding:8px 10px;margin:4px 0;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer;font-size:14px}
.btn:hover{border-color:var(--acc)} .btn kbd{min-width:44px;margin-right:8px;font:12px monospace;color:var(--mute)}
.btn.ok{border-color:var(--ok)} .btn.bad{border-color:var(--bad)} .btn.chosen{background:#e8f0fe;border-color:var(--acc)}
.big{padding:12px;font-size:15px;font-weight:600;justify-content:center} .big.ok{background:var(--ok);color:#fff} .big.bad{background:var(--bad);color:#fff}
.row{display:flex;gap:6px} .row .btn{flex:1;justify-content:center}
.status{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}
.s-todo{background:#9e9e9e} .s-ok{background:var(--ok)} .s-off{background:var(--fix)} .s-drop{background:var(--bad)}
#dots{display:flex;flex-wrap:wrap;gap:3px;margin:10px 0} #dots span{width:10px;height:10px;border-radius:2px;cursor:pointer}
#dots span.here{outline:2px solid var(--acc);outline-offset:1px}
.cam img{width:100%;border-radius:6px;cursor:zoom-in;margin-top:4px}
#lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:9;cursor:zoom-out;align-items:center;justify-content:center}
#lightbox img{max-width:98vw;max-height:98vh}
ol.small{padding-left:20px;font-size:12px} input[type=text],textarea{width:100%;font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px}
.note{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:8px;margin:8px 0;font-size:13px}
</style></head><body>
<div id="mapwrap"><svg id="map"><g id="vp"><image id="ortho"/><g id="els"></g><g id="marks"></g></g></svg>
  <div id="hint">drag: move &middot; wheel: zoom &middot; F: whole scene &middot; Z: back to the item &middot; H (hold): hide the lines to see the paint</div>
  <div id="legend"></div></div>
<div id="side"><header><h1 id="title"></h1>
  <nav><button data-step="check">1. Check items</button><button data-step="missing">2. Find missing</button><button data-step="finish">3. Finish</button></nav>
  <div id="bar"><div></div></div><div id="prog"></div></header>
  <main id="panel"></main></div>
<div id="lightbox"><img></div>
<script>
const D = __DATA__, NS = "http://www.w3.org/2000/svg", $ = id => document.getElementById(id), NEAR_M = 40;
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const store = {get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
               set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }};

// ---- state: fixes (as saved), checked elements, flagged issues; restored from the review file or a newer draft
const R = D.review || {}, DRAFT = "hdlabel-review:" + D.scene;
let S = {fixes: (R.fixes || []).slice(), checked: (R.checked || []).slice(), issues: (R.issues || []).slice(),
         notes: R.notes || "", decided: null};
const draft = store.get(DRAFT); let restored = false;
if (draft && draft.base === JSON.stringify(R)) { S = Object.assign(S, draft.state); restored = true; }
const save = () => store.set(DRAFT, {base: JSON.stringify(R), state: S});
let step = "check", pos = 0, mode = null, pending = [];
const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.01;
const on = i => x => x.at && x.kind === D.elements[i].kind && near(x.at, D.elements[i].at);  // entry about element i
function status(i) { if (S.fixes.some(f => on(i)(f) && f.drop)) return "drop"; if (S.issues.some(on(i))) return "off";
  return S.checked.some(on(i)) || D.elements[i].props.reviewed ? "ok" : "todo"; }
function decide(i, answer) {  // "ok" | "drop" | "off"; attribute fixes (set) of earlier reviews are kept
  const el = D.elements[i], is = on(i), T = D.tasks[el.group];
  S.fixes = S.fixes.filter(f => !(is(f) && f.drop)); S.checked = S.checked.filter(c => !is(c)); S.issues = S.issues.filter(x => !is(x));
  if (answer === "drop") S.fixes.push({kind: el.kind, at: el.at, drop: true});
  else S.checked.push({kind: el.kind, at: el.at});
  if (answer === "off") S.issues.push({kind: el.kind, at: el.at, note: T.off});
  save(); }

// ---- map
const svg = $("map"), vp = $("vp"), G = $("els");
const px = p => [(p[0] - D.origin[0] - D.lo[0]) / D.res, D.ny - (p[1] - D.origin[1] - D.lo[1]) / D.res];
const mp = q => [q[0] * D.res + D.origin[0] + D.lo[0], (D.ny - q[1]) * D.res + D.origin[1] + D.lo[1]].map(v => +v.toFixed(2));
const o = $("ortho"); o.setAttribute("href", D.ortho); o.setAttribute("width", D.nx); o.setAttribute("height", D.ny);
let tf = {x: 0, y: 0, k: 1};
const apply = () => { vp.setAttribute("transform", `translate(${tf.x},${tf.y}) scale(${tf.k})`); drawMarks(); };
function fit(x0, y0, x1, y1, pad) { const b = svg.getBoundingClientRect(); const w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad;
  tf.k = Math.min(b.width / w, b.height / h); tf.x = b.width / 2 - (x0 + x1) / 2 * tf.k; tf.y = b.height / 2 - (y0 + y1) / 2 * tf.k; apply(); }
const fitAll = () => fit(0, 0, D.nx, D.ny, 0);
function focus(i) { const P = [...D.elements[i].rings, ...D.elements[i].outlines].flat().map(px);
  const xs = P.map(p => p[0]), ys = P.map(p => p[1]), span = 12 / D.res;  // show at least ~25 m around the element
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const hw = Math.max(span, (Math.max(...xs) - Math.min(...xs)) / 2), hh = Math.max(span, (Math.max(...ys) - Math.min(...ys)) / 2);
  fit(cx - hw, cy - hh, cx + hw, cy + hh, 4 / D.res); }
const hidden = new Set(store.get("hdlabel-review:hidden") || []);
D.elements.forEach((el, i) => [...el.rings, ...el.outlines].forEach((r, k) => {
  const n = document.createElementNS(NS, el.closed || k >= el.rings.length ? "polygon" : "polyline");
  n.setAttribute("points", r.map(p => px(p).join(",")).join(" ")); n.setAttribute("class", "el k-" + el.group + " e" + i);
  n.style.stroke = D.colors[el.group]; n.onclick = ev => { ev.stopPropagation(); if (!mode) openItem(i); }; G.appendChild(n); }));
function legend() { const count = {}; D.elements.forEach(e => count[e.group] = (count[e.group] || 0) + 1);
  $("legend").innerHTML = "<b>show</b>" + Object.keys(D.colors).filter(k => count[k]).map(k =>
    `<label><input type="checkbox" data-k="${k}" ${hidden.has(k) ? "" : "checked"}><i style="background:${D.colors[k]}"></i>${D.names[k]} (${count[k]})</label>`).join("");
  $("legend").querySelectorAll("input").forEach(c => c.onchange = () => { c.checked ? hidden.delete(c.dataset.k) : hidden.add(c.dataset.k);
    store.set("hdlabel-review:hidden", [...hidden]); showLayers(); }); showLayers(); }
const showLayers = () => Object.keys(D.colors).forEach(k => G.querySelectorAll(".k-" + k).forEach(n => n.style.display = hidden.has(k) ? "none" : ""));
function highlight(i) { G.classList.toggle("dim", i !== null); G.querySelectorAll(".cur").forEach(n => n.classList.remove("cur"));
  if (i !== null) G.querySelectorAll(".e" + i).forEach(n => { n.classList.add("cur"); n.style.display = ""; }); drawMarks(); }
let focused = null;
function drawMarks() { const mk = $("marks"); mk.innerHTML = ""; const r = 7 / tf.k;
  const dot = (p, col) => { const c = document.createElementNS(NS, "circle"); const q = px(p);
    c.setAttribute("cx", q[0]); c.setAttribute("cy", q[1]); c.setAttribute("r", r); c.setAttribute("fill", col); c.setAttribute("class", "pin"); mk.appendChild(c); };
  if (focused !== null) { const P = [...D.elements[focused].rings, ...D.elements[focused].outlines].flat().map(px);
    const xs = P.map(p => p[0]), ys = P.map(p => p[1]), m = 12 / tf.k, h = document.createElementNS(NS, "rect");
    h.setAttribute("x", Math.min(...xs) - m); h.setAttribute("y", Math.min(...ys) - m); h.setAttribute("class", "halo");
    h.setAttribute("width", Math.max(...xs) - Math.min(...xs) + 2 * m); h.setAttribute("height", Math.max(...ys) - Math.min(...ys) + 2 * m); mk.appendChild(h); }
  S.issues.filter(x => !x.kind).forEach(x => dot(x.at, "#d93025")); pending.forEach(p => dot(p, "#ffeb3b")); }

// pan / zoom / clicks on the map
svg.addEventListener("wheel", e => { e.preventDefault(); const s = Math.exp(-e.deltaY * 0.0015);
  tf.x = e.offsetX - (e.offsetX - tf.x) * s; tf.y = e.offsetY - (e.offsetY - tf.y) * s; tf.k *= s; apply(); }, {passive: false});
let drag = null;
svg.addEventListener("mousedown", e => drag = {x: e.clientX, y: e.clientY, moved: false});
window.addEventListener("mousemove", e => { if (!drag) return; tf.x += e.clientX - drag.x; tf.y += e.clientY - drag.y;
  drag.moved ||= Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 2; drag.x = e.clientX; drag.y = e.clientY; apply(); });
window.addEventListener("mouseup", e => { const moved = drag && drag.moved; drag = null;
  if (moved || !mode || e.target.closest("#map") === null) return;
  pending.push(mp([(e.offsetX - tf.x) / tf.k, (e.offsetY - tf.y) / tf.k])); drawMarks(); render(); });

// ---- panel
const queue = D.queue, todo = () => queue.filter(i => status(i) === "todo");
const counts = () => ({drop: S.fixes.filter(f => f.drop).length, off: S.issues.filter(x => x.kind).length, flag: S.issues.filter(x => !x.kind).length});
function progress() { const done = queue.length - todo().length, c = counts();
  $("bar").firstElementChild.style.width = (queue.length ? 100 * done / queue.length : 100) + "%";
  $("prog").textContent = `checked ${done} / ${queue.length} · deleted ${c.drop} · outline off ${c.off} · marked ${c.flag}`;
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("on", b.dataset.step === step)); }
function nearestSample(at) { let best = null;
  D.samples.forEach(s => { const d = s.xy ? Math.hypot(s.xy[0] - at[0], s.xy[1] - at[1]) : Infinity; if (!best || d < best.d) best = {s, d}; });
  return best; }
function camera(at) { const b = nearestSample(at);
  if (!b || b.d > NEAR_M) return `<p class="mute">No camera sheet near here (they exist for a few samples of the scene); judge from the map.</p>`;
  return `<div class="cam"><div class="mute">Camera views with the labels drawn in (taken ${b.d.toFixed(0)} m from here; click to enlarge).
    Labels also paint over cars and buildings: that is expected.</div><img src="${esc(b.s.file)}"></div>`; }
const btn = (key, label, cls, onclick, chosen) => `<button class="btn ${cls || ""} ${chosen ? "chosen" : ""}" onclick="${onclick}"><kbd>${key}</kbd>${esc(label)}</button>`;
function why(el) { const p = el.props, c = p.confidence;
  return (c != null ? `confidence ${c.toFixed(2)}` : "") + (p.observations != null ? `, seen in ${p.observations} camera views` : "") + (p.length ? `, ${p.length.toFixed(1)} m long` : ""); }

function openItem(i) { step = "check"; const k = queue.indexOf(i); if (k >= 0) pos = k; focused = i; render(i); focus(i); }
function renderCheck(only) {
  const i = only != null ? only : queue[pos];
  if (i == null) { focused = null; highlight(null); return `<h2>Nothing to check</h2><p>No element of this scene needs a decision. Go on with <b>2. Find missing</b>.</p>`; }
  const el = D.elements[i], T = D.tasks[el.group], st = status(i), inQ = queue.indexOf(i);
  highlight(i); focused = i;
  let h = `<div class="mute">${inQ >= 0 ? `item ${inQ + 1} of ${queue.length} · ${esc(D.reasons[i] || "")}` : "not in the check list"}</div>
    <h2>${esc(D.names[el.group])} <span class="status s-${st}">${{todo: "to check", ok: "correct", off: "outline off", drop: "deleted"}[st]}</span></h2>
    <div class="mute">${esc(why(el))}</div>`;
  if (T) h += `<div class="q">${esc(T.question)}</div>` + btn("Space", T.ok, "ok", `answer(${i}, 'ok')`, st === "ok") +
    btn("X", T.drop, "bad", `answer(${i}, 'drop')`, st === "drop") + btn("1", T.off, "", `answer(${i}, 'off')`, st === "off") +
    `<div class="mute">Hold H to hide the lines and look at the paint itself.</div>`;
  else h += `<p class="mute">Shown for orientation only; nothing to decide.</p>`;
  h += `<div class="row">${btn("←", "back", "", "move(-1)")}${btn("→", "skip", "", "move(1)")}</div>`;
  h += `<div id="dots">${queue.map((j, k) => `<span title="${esc(D.reasons[j])}" class="s-${status(j)} ${j === i ? "here" : ""}" onclick="pos=${k}; openItem(${j})"></span>`).join("")}</div>`;
  h += `<div class="mute">The dashed cyan box marks the element; the others are dimmed. Click any element on the map to look at it.</div>`;
  return h + camera(el.at); }
function renderMissing() { focused = null; highlight(null);
  let h = `<h2>Look for what was missed</h2><p>Move along the road (drag, wheel). Is there paint without a yellow outline, or a curb
    without a blue line? Hold H to compare with the bare image; the legend (bottom left) hides layers.</p>`;
  if (mode === "issue") {
    h += `<div class="note">${pending.length ? "What is wrong here?" : "Click the place on the map."}</div>`;
    if (pending.length) h += `<input type="text" id="issueText" placeholder="e.g. paint not outlined, curb missing"><div class="row">${btn("Enter", "add", "ok", "addIssue()")}${btn("Esc", "cancel", "", "cancelMode()")}</div>`; }
  else h += btn("P", "Mark a missed marking / curb or another problem", "", "startMode('issue')");
  const iss = S.issues.map((x, k) => [x, k]).filter(([x]) => !x.kind);
  if (iss.length) h += `<h2 style="margin-top:14px">Marked</h2><ol class="small">` +
    iss.map(([x, k]) => `<li>${esc(x.note)} <a href="#" onclick="S.issues.splice(${k},1); save(); render(); return false">remove</a></li>`).join("") + "</ol>";
  h += `<p class="mute">Marked problems are stored in the review file for a person to follow up; they are not applied automatically.
    If much is missing, reject the scene in step 3.</p>`;
  return h + `<div class="row">${btn("", "Next: finish →", "ok", "go('finish')")}</div>`; }
function renderFinish() { focused = null; highlight(null); const left = todo().length;
  const c = counts();
  let h = `<h2>Finish the scene</h2>`;
  if (left) h += `<div class="note">${left} item(s) are not checked yet. <a href="#" onclick="go('check'); return false">Check them</a> or finish anyway.</div>`;
  h += `<p>Your result: <b>${queue.length - left}</b> checked, <b>${c.drop}</b> deleted, <b>${c.off}</b> outlines off, <b>${c.flag}</b> problems marked.</p>
    <label>Your name<br><input type="text" id="reviewer" value="${esc(R.reviewer || store.get("hdlabel-review:reviewer") || "")}"></label>
    <label>Notes (optional)<br><textarea id="notes" rows="3">${esc(S.notes)}</textarea></label>
    <p>Can the labels of this scene be used for training (after your changes)?</p>
    <div class="row"><button class="btn big ok" onclick="finish(false)">Accept</button><button class="btn big bad" onclick="finish(true)">Reject</button></div>
    <p class="mute">Reject = the scene is too wrong to fix here (e.g. outlines shifted against the paint, large parts missing).</p>`;
  if (S.decided) h += `<div class="note">Saved as <b>${esc(D.scene)}.json</b> (${S.decided}). It must end up at<br><code>${esc(D.review_path)}</code><br>
    Then run <code>./autolabel run &lt;config&gt;</code> to apply it. Saving again overwrites it.</div>`;
  return h; }
function render(only) { const p = $("panel");
  p.innerHTML = (restored ? `<div class="note">Restored your unsaved work on this scene. <a href="#" onclick="discard(); return false">discard it</a></div>` : "") +
    (step === "check" ? renderCheck(only) : step === "missing" ? renderMissing() : renderFinish());
  p.querySelectorAll(".cam img").forEach(im => im.onclick = () => { $("lightbox").style.display = "flex"; $("lightbox").firstElementChild.src = im.src; });
  const it = $("issueText"); if (it) it.focus(); progress(); }

// ---- actions
function answer(i, a) { decide(i, a); const k = queue.indexOf(i);
  if (k >= 0 && step === "check") { const nxt = queue.findIndex((j, n) => n > k && status(j) === "todo"); const first = todo()[0];
    if (nxt >= 0) return openItem(queue[nxt]); if (first != null) return openItem(first); pos = k; render(); return; }
  render(i); }
function move(d) { pos = Math.max(0, Math.min(queue.length - 1, pos + d)); if (queue.length) openItem(queue[pos]); }
function go(s) { step = s; mode = null; pending = []; svg.classList.remove("adding");
  if (s === "check" && queue.length) { const t = todo(); openItem(t.length ? t[0] : queue[pos]); return; }
  if (s !== "check") fitAll(); render(); }
function startMode(m) { mode = m; pending = []; svg.classList.add("adding"); render(); }
function cancelMode() { mode = null; pending = []; svg.classList.remove("adding"); drawMarks(); render(); }
function addIssue() { const t = $("issueText").value.trim(); if (!t) return; S.issues.push({at: pending[0], note: t}); save(); cancelMode(); }
function discard() { try { localStorage.removeItem(DRAFT); } catch (e) {} location.reload(); }
function reviewJson(reject) {
  const out = Object.assign({}, R, {reviewer: ($("reviewer") || {}).value ?? R.reviewer ?? "", date: new Date().toISOString().slice(0, 10),
    notes: S.notes, fixes: S.fixes, checked: S.checked, issues: S.issues});
  if (reject !== undefined) { out.reject = reject; out.verdict = reject ? "bad" : S.fixes.length || S.issues.length ? "fixed" : "ok"; }
  return out; }
async function finish(reject) { S.notes = $("notes").value; store.set("hdlabel-review:reviewer", $("reviewer").value);
  const text = JSON.stringify(reviewJson(reject), null, 1), name = D.scene + ".json";
  try { if (window.showSaveFilePicker) { const h = await window.showSaveFilePicker({suggestedName: name, types: [{description: "review json", accept: {"application/json": [".json"]}}]});
      const w = await h.createWritable(); await w.write(text); await w.close(); }
    else { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type: "application/json"})); a.download = name; a.click(); } }
  catch (e) { if (e.name === "AbortError") return; throw e; }
  S.decided = reject ? "rejected" : "accepted"; save(); render(); }

document.querySelectorAll("nav button").forEach(b => b.onclick = () => go(b.dataset.step));
document.addEventListener("keyup", e => { if (e.key === "h" || e.key === "H") ["els", "marks"].forEach(id => $(id).classList.remove("off")); });
$("lightbox").onclick = () => $("lightbox").style.display = "none";
document.addEventListener("keydown", e => {
  if (e.target.matches && e.target.matches("input, textarea")) { if (e.key === "Enter" && e.target.id === "issueText") addIssue(); if (e.key === "Escape") cancelMode(); return; }
  if (e.key === "Escape") { $("lightbox").style.display = "none"; return cancelMode(); }
  if (e.key === "h" || e.key === "H") return ["els", "marks"].forEach(id => $(id).classList.add("off"));
  if (e.key === "f" || e.key === "F") return fitAll();
  if ((e.key === "z" || e.key === "Z") && focused !== null) return focus(focused);
  if (step === "missing") { if (e.key === "p" || e.key === "P") startMode("issue"); return; }
  if (step !== "check" || focused === null) return;
  if (e.key === "ArrowRight") return move(1); if (e.key === "ArrowLeft" || e.key === "Backspace") return move(-1);
  if (!D.tasks[D.elements[focused].group]) return;
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); answer(focused, "ok"); }
  else if (e.key === "x" || e.key === "X" || e.key === "Delete") answer(focused, "drop");
  else if (e.key === "1") answer(focused, "off"); });

$("title").textContent = "Review · " + D.scene; legend();
requestAnimationFrame(() => { fitAll(); go("check"); });
</script></body></html>"""
