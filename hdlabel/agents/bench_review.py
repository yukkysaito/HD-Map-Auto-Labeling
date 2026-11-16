"""Person review of the benchmark reference: a page with every element, its evidence and the judges' votes
(`./autolabel bench review`), and the person's decisions applied + locked (`./autolabel bench apply`).
Kept out of bench.py, whose code is part of the locked evaluator."""

from __future__ import annotations

import datetime
import json

from hdlabel.agents.bench import bench_dir, lock


# ---------------------------------------------------------------- person review of the reference
def _order(e: dict) -> tuple:
    """disputed elements first: no majority, then split votes, then unanimous."""
    v = ["wrong" if d in ("drop", "off") else d for d in e["judges"].values()]
    return (e["ref"] is not None, len(set(v)) == 1, e["group"], e["key"])


def review_page(name: str) -> str:
    """bench/<name>/review.html: every reference element with its evidence image and the judges' votes; the person
    marks it correct / wrong / unsure and saves the decisions as a JSON file (`bench apply`)."""
    import html

    d = bench_dir(name)
    ref = json.load(open(f"{d}/reference.json"))
    items = sorted(ref["items"], key=_order)
    cards = []
    for e in items:
        votes = ", ".join(
            f"{html.escape(j.split('/')[-1] if '/' in j else j)}: {v}" for j, v in sorted(e["judges"].items())
        )
        unan = len({"wrong" if v in ("drop", "off") else v for v in e["judges"].values()}) == 1
        cards.append(
            f'<div class="card{" unan" if unan else ""}" data-key="{e["key"]}" data-ref="{e["ref"] or ""}">'
            f'<div class="head">#{e["key"]} <b>{e["kind"]}</b> {e["size_m"]:.1f} m, {html.escape(e["why"])}, {e["split"]}'
            f" | AI majority: <b>{e['ref'] or 'none'}</b> ({votes})</div>"
            f'<img loading="lazy" src="{e["evidence"]}">'
            '<div class="btns"><button data-v="ok">correct</button><button data-v="wrong">wrong</button>'
            '<button data-v="unsure">cannot tell</button></div></div>'
        )
    page = (
        """<!doctype html><meta charset="utf-8"><title>Benchmark review</title>
<style>body{font:14px sans-serif;margin:16px;background:#fafafa}img{width:100%;max-width:1536px;display:block}
.card{background:#fff;border:2px solid #ddd;margin:0 0 14px;padding:6px}.card.ok{border-color:#2a2}.card.wrong{border-color:#d22}
.card.unsure{border-color:#999}.btns button{margin:4px 6px 0 0;padding:4px 12px}.btns button.on{font-weight:bold;outline:2px solid #333}
.bar{position:sticky;top:0;background:#fafafa;padding:6px 0;z-index:1}
.guide{background:#fff;border:1px solid #ccc;padding:8px 12px;margin:0 0 14px}.guide table{border-collapse:collapse}
.guide td,.guide th{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}.hide .unan{display:none}</style>
<div class="bar"><b>Benchmark reference review</b>
<label><input type="checkbox" id="h"> hide elements all judges agree on</label>
<button id="s">save decisions (JSON)</button> <span id="n"></span>
<small>Then: ./autolabel bench apply CONFIG bench_review.json --by person:NAME</small></div>
<details open class="guide"><summary><b>How to judge</b> (the same rules the AI judges got: hdlabel/agents/prompts/verifier.md)</summary>
<p><b>Each row</b>, left to right: <i>map</i> (top view made from many camera frames, north up) | <i>map + outline</i> |
two camera frames that see the element, each bare and <i>+ outline</i>. The outline is the element being checked:
<span style="color:#2a62ff"><b>blue = curb</b></span>, <span style="color:#d9a400"><b>yellow = road marking</b></span>
(lane line, stop line, arrow, painted area). The map is sharp in the middle of the road and blurred at its edges; the
camera is sharp up close.</p>
<p>Only judge whether the extraction is right (is it there, is it in the right place), not its meaning (arrow direction,
line type).</p>
<table>
<tr><th></th><th>correct</th><th>wrong</th></tr>
<tr><td><b>curb</b> (blue)</td><td>a kerb: a <b>physically raised edge</b> between the road and a sidewalk, island, median
or planter next to the road, and the line follows it within about 0.3 m. A lowered kerb at a driveway or crossing, where the
kerb line continues, counts.</td><td>no raised edge: flat transition from road to sidewalk or ground (that boundary is a
separate element, <i>road_boundary</i>, not checked here), a painted line, the base of a wall / fence / building,
the edge of a lawn or planter away from the road, a shadow. Or the kerb is there but the line is clearly beside it
(more than about 0.5 m).</td></tr>
<tr><td><b>marking</b> (yellow)</td><td>the outline lies on <b>paint on the road surface</b> (any paint) and follows it
within about 0.3 m. A short piece of a real line (one dash) is correct. Each dash of a dashed line is its own element: an outline that runs across the
unpainted gap between dashes is wrong.</td><td>no paint: shadow, asphalt patch or seam,
tyre marks, wet patch, manhole, paint on a kerb face / wall / pole, paint on a sidewalk or plaza. Or the paint is there but
the outline is clearly beside it or covers a clearly wrong extent.</td></tr>
</table>
<p><b>cannot tell</b>: the images do not show it (blurred, hidden in both views, outside the observed area). If one view
is blocked by a car or pole, judge from the others.</p></details>
<div id="list">"""
        + "\n".join(cards)
        + """</div>
<script>
const K='bench_review_"""
        + name
        + """';let dec={};try{dec=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
function show(){document.querySelectorAll('.card').forEach(c=>{const v=dec[c.dataset.key];c.className='card'+
(c.classList.contains('unan')?' unan':'')+(v?' '+v:'');c.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.v===v))});
const n=Object.keys(dec).length,ch=Object.entries(dec).filter(([k,v])=>v!=='unsure'&&document.querySelector(`[data-key="${k}"]`).dataset.ref!==v).length;
document.getElementById('n').textContent=`${n} of ${document.querySelectorAll('.card').length} decided, ${ch} differ from the AI majority`}
document.getElementById('list').onclick=ev=>{const b=ev.target.closest('button');if(!b)return;const k=b.closest('.card').dataset.key;
if(dec[k]===b.dataset.v)delete dec[k];else dec[k]=b.dataset.v;try{localStorage.setItem(K,JSON.stringify(dec))}catch(e){};show()};
document.getElementById('h').onchange=ev=>document.body.classList.toggle('hide',ev.target.checked);
document.getElementById('s').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(
[JSON.stringify({benchmark:'"""
        + name
        + """',decisions:dec},null,1)],{type:'application/json'}));a.download='bench_review.json';a.click()};
show();
</script>"""
    )
    out = f"{d}/review.html"
    open(out, "w").write(page)
    return out


def apply_review(name: str, decisions: dict, by: str) -> dict:
    """the person's decisions {key: ok | wrong | unsure} replace the judges' majority (ok / wrong) and are recorded as
    a judge; then the reference is locked by that person."""
    if not by.startswith("person:"):
        raise ValueError("only a person reviews the reference (--by person:NAME)")
    d = bench_dir(name)
    ref = json.load(open(f"{d}/reference.json"))
    stat = dict(decided=0, changed=0, added=0, unsure=0)
    for e in ref["items"]:
        v = decisions.get(e["key"])
        if v not in ("ok", "wrong", "unsure"):
            continue
        e["judges"][by] = v
        stat["decided"] += 1
        if v == "unsure":
            stat["unsure"] += 1
            continue
        stat["changed"] += e["ref"] is not None and e["ref"] != v
        stat["added"] += e["ref"] is None
        e["ref"], e["ref_by"] = v, by
    ref.setdefault("reviews", []).append(dict(by=by, date=datetime.date.today().isoformat(), **stat))
    ref["judges"] = sorted({j for e in ref["items"] for j in e["judges"]})
    json.dump(ref, open(f"{d}/reference.json", "w"), indent=1)
    lock(name, by)
    return stat
