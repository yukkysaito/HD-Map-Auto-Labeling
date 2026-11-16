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
.bar{position:sticky;top:0;background:#fafafa;padding:6px 0;z-index:1}.hide .unan{display:none}</style>
<div class="bar"><b>Benchmark reference review</b> - is the outlined element (blue) really there and in the right place?
<label><input type="checkbox" id="h"> hide elements all judges agree on</label>
<button id="s">save decisions (JSON)</button> <span id="n"></span>
<br><small>correct = the element exists where it is drawn; wrong = not a curb / marking there, or clearly offset. Disputed
elements come first. Then: ./autolabel bench apply CONFIG bench_review.json --by person:NAME</small></div>
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
