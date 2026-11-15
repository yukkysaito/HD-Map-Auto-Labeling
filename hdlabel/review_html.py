"""review.html: a self-contained page to review one scene in a browser (no server). The ortho image with the vectors on
top (pan: drag, zoom: wheel); click an element to drop it or change its kind / type / style / arrow directions; add a
missed arrow by clicking its tail and tip; mark the scene rejected; "Save" downloads review/<scene>.json in the format
of docs/ANNOTATION.md (fixes in map coordinates), merged with the fixes already in that file.
"""

from __future__ import annotations

import json

MARKING_TYPES = ["zebra", "stop_line", "arrow", "text", "parking", "double_line", "diamond", "other"]
COLORS = dict(lane_line="#ffffff", stop_line="#00ffff", curb="#0050ff", road_boundary="#ff3030", crosswalk="#ff00ff",
              marking_area="#00ff00", arrow="#ffa000")  # fmt: skip


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


def write(path: str, scene: str, ortho: str, shape, lo, res: float, origin, features: list, review: dict | None,
          samples: list[str]) -> None:  # fmt: skip
    """ortho: path of ortho.jpg relative to the page; shape / lo / res / origin: the label grid (ortho = rot90 of it)."""
    els = []
    for f in features:
        k = f["properties"]["kind"]
        if k not in COLORS or (k == "road_boundary" and not f["properties"].get("observed")):
            continue
        g = f["geometry"]
        rings = g["coordinates"] if g["type"] == "Polygon" else [g["coordinates"]]
        pr = {x: v for x, v in f["properties"].items() if x not in ("segments", "outlines")}
        els.append(dict(kind=k, closed=g["type"] == "Polygon", rings=[[[p[0], p[1]] for p in r] for r in rings],
                        outlines=[[[p[0], p[1]] for p in o] for o in f["properties"].get("outlines", [])],
                        at=element_at(f), props=pr))  # fmt: skip
    data = dict(scene=scene, ortho=ortho, nx=int(shape[0]), ny=int(shape[1]), lo=[float(v) for v in lo], res=float(res),
                origin=[float(v) for v in origin[:2]], elements=els, review=review or {}, samples=samples,
                colors=COLORS, marking_types=MARKING_TYPES)  # fmt: skip
    open(path, "w").write(PAGE.replace("__DATA__", json.dumps(data)))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Review</title><style>
body{margin:0;font:13px sans-serif;display:flex;height:100vh} #map{flex:1;height:100vh;background:#222;cursor:grab}
#side{width:340px;padding:10px;overflow:auto;border-left:1px solid #ccc} .sel{stroke-width:5px!important}
.el{fill:none;stroke-width:2px;vector-effect:non-scaling-stroke;cursor:pointer} .fix{fill:#f00;stroke:#fff}
button{margin:2px} li{margin:2px 0} .mode{background:#fc0}</style></head><body>
<svg id="map"><g id="vp"><image id="ortho"/><g id="els"></g><g id="marks"></g></g></svg>
<div id="side"><h3 id="title"></h3>
<p>drag: pan, wheel: zoom, click: select. <button id="addArrow">add arrow (click tail, then tip)</button></p>
<div id="info">no element selected</div><hr>
<label><input type="checkbox" id="reject"> reject the whole scene</label><br>
reviewer <input id="reviewer" size="10"> verdict <select id="verdict"><option>ok</option><option>fixed</option><option>bad</option></select><br>
notes<br><textarea id="notes" rows="3" cols="38"></textarea>
<h4>fixes</h4><ol id="fixes"></ol><button id="save">Save review json</button>
<h4>camera sheets</h4><div id="samples"></div></div>
<script>
const D = __DATA__, NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("map"), vp = document.getElementById("vp");
const R = D.review || {}; let fixes = (R.fixes || []).slice(), sel = null, adding = null;
document.getElementById("title").textContent = D.scene;
["reviewer", "verdict", "notes"].forEach(k => { if (R[k]) document.getElementById(k).value = R[k]; });
document.getElementById("reject").checked = !!R.reject;
const px = p => [(p[0] - D.origin[0] - D.lo[0]) / D.res, D.ny - (p[1] - D.origin[1] - D.lo[1]) / D.res];
const mp = q => [q[0] * D.res + D.origin[0] + D.lo[0], (D.ny - q[1]) * D.res + D.origin[1] + D.lo[1]];
const o = document.getElementById("ortho"); o.setAttribute("href", D.ortho); o.setAttribute("width", D.nx); o.setAttribute("height", D.ny);
const box = svg.getBoundingClientRect(); let tf = {x: 0, y: 0, k: Math.min(box.width / D.nx, box.height / D.ny)};
const apply = () => vp.setAttribute("transform", `translate(${tf.x},${tf.y}) scale(${tf.k})`); apply();
svg.addEventListener("wheel", e => { e.preventDefault(); const s = Math.exp(-e.deltaY * 0.0015);
  tf.x = e.offsetX - (e.offsetX - tf.x) * s; tf.y = e.offsetY - (e.offsetY - tf.y) * s; tf.k *= s; apply(); });
let drag = null; svg.addEventListener("mousedown", e => drag = {x: e.clientX, y: e.clientY, moved: false});
svg.addEventListener("mousemove", e => { if (!drag) return; tf.x += e.clientX - drag.x; tf.y += e.clientY - drag.y;
  drag.moved ||= Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 2; drag.x = e.clientX; drag.y = e.clientY; apply(); });
svg.addEventListener("mouseup", e => { const moved = drag && drag.moved; drag = null; if (moved || !adding) return;
  const q = [(e.offsetX - tf.x) / tf.k, (e.offsetY - tf.y) / tf.k], m = mp(q).map(v => +v.toFixed(2));
  adding.push(m); if (adding.length === 2) { const dirs = prompt("directions (comma separated: straight,left,right; empty = from the template match)", "");
    const add = {tail: adding[0], tip: adding[1]}; if (dirs) add.directions = dirs.split(",").map(s => s.trim()).filter(s => s);
    fixes.push({kind: "arrow", add}); adding = null; document.getElementById("addArrow").classList.remove("mode"); render(); } });
document.getElementById("addArrow").onclick = e => { adding = []; e.target.classList.add("mode"); };
const g = document.getElementById("els");
D.elements.forEach((el, i) => { [...el.rings, ...el.outlines].forEach(r => {
  const n = document.createElementNS(NS, el.closed || el.outlines.includes(r) ? "polygon" : "polyline");
  n.setAttribute("points", r.map(p => px(p).join(",")).join(" ")); n.setAttribute("class", "el e" + i);
  n.style.stroke = D.colors[el.kind]; n.onclick = ev => { ev.stopPropagation(); select(i); }; g.appendChild(n); }); });
function fixed(i) { const el = D.elements[i]; return fixes.filter(f => f.kind === el.kind && f.at && Math.hypot(f.at[0] - el.at[0], f.at[1] - el.at[1]) < 0.01); }
function setFix(i, f) { const el = D.elements[i]; fixes = fixes.filter(x => !fixed(i).includes(x)); if (f) fixes.push(Object.assign({kind: el.kind, at: el.at}, f)); select(i); render(); }
function select(i) { document.querySelectorAll(".sel").forEach(n => n.classList.remove("sel"));
  document.querySelectorAll(".e" + i).forEach(n => n.classList.add("sel")); sel = i; const el = D.elements[i];
  const p = Object.entries(el.props).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join("<br>");
  let h = `<b>${el.kind}</b> at ${el.at}<br>${p}<br><button onclick="setFix(${i},{drop:true})">drop</button>`;
  if (el.kind === "marking_area" || el.kind === "crosswalk") h += " type " + D.marking_types.concat(["red_white"]).map(t => `<button onclick="setFix(${i},{set:{type:'${t}'}})">${t}</button>`).join("");
  if (el.kind === "marking_area") h += `<br><button onclick="setFix(${i},{set:{kind:'crosswalk'}})">is a crosswalk</button>`;
  if (el.kind === "lane_line") h += `<br>style ${["solid", "dashed"].map(s => `<button onclick="setFix(${i},{set:{style:'${s}'}})">${s}</button>`).join("")} <button onclick="setFix(${i},{set:{kind:'stop_line'}})">is a stop line</button>`;
  if (el.kind === "arrow") h += "<br>directions " + [["straight"], ["left"], ["right"], ["straight", "left"], ["straight", "right"]].map(d =>
    `<button onclick='setFix(${i},{set:{directions:${JSON.stringify(d)},shape:"${d.map(x => x[0].toUpperCase()).join("")}"}})'>${d.join("+")}</button>`).join("");
  h += `<br><button onclick="setFix(${i},null)">undo fix</button>`; document.getElementById("info").innerHTML = h; }
function render() { const ol = document.getElementById("fixes"); ol.innerHTML = ""; const mk = document.getElementById("marks"); mk.innerHTML = "";
  fixes.forEach((f, k) => { const li = document.createElement("li"); li.textContent = JSON.stringify(f) + " ";
    const b = document.createElement("button"); b.textContent = "x"; b.onclick = () => { fixes.splice(k, 1); render(); }; li.appendChild(b); ol.appendChild(li);
    const p = px(f.at || f.add.tail), c = document.createElementNS(NS, "circle"); c.setAttribute("cx", p[0]); c.setAttribute("cy", p[1]);
    c.setAttribute("r", 4 / tf.k); c.setAttribute("class", "fix"); mk.appendChild(c); }); }
function reviewJson() { return Object.assign({}, R, {fixes,
  reviewer: document.getElementById("reviewer").value, verdict: document.getElementById("verdict").value,
  notes: document.getElementById("notes").value, reject: document.getElementById("reject").checked, date: new Date().toISOString().slice(0, 10)}); }
document.getElementById("save").onclick = () => { const out = reviewJson();
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = D.scene + ".json"; a.click(); };
document.getElementById("samples").innerHTML = D.samples.map(s => `<a href="${s}" target="_blank"><img src="${s}" width="320"></a>`).join("<br>");
render();
</script></body></html>"""
