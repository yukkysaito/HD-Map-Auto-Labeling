"""Acceptance of a labelled output: per scene risk from the automatic checks and the review (an AI agent's decisions
on the evidence packs, docs/AGENT_REVIEW.md), ranked, plus a reproducible random sample, so the person accepting the
batch looks at the riskiest scenes and at what the reviewer decided.

Risk terms (0 = fine, 1 = bad), weighted mean -> risk:
  curb        1 - share of curb profiles with a LiDAR height step (geom_check), below 0.8 counts
  height      vector z vs LiDAR ground p95 above 10 cm (geom_check)
  qa          share of samples failing QA (export)
  calib       time offsets: coarse / fallback 0.5, failed 1 (scene.json)
  boundary    share of the road boundary length that is only the edge of the observed area
  confidence  share of marking / crosswalk / curb elements with confidence < 0.5 (vectorize)
  repeat      1 - agreement of the labels of two drives (./autolabel qa), when measured
  spot_wrong  share of the randomly spot-checked confident elements the reviewer found wrong (no paint / outline off)
  missed      share of the map tiles where the reviewer found paint / curbs without an outline
The batch is accepted when every scene is reviewed, none is rejected, none exceeds max_risk and the pooled spot check
(all scenes, weighted by element length) finds at most max_spot_error of the data wrong. Ten spot checks per scene are
too few to judge one scene by (3 of 10 wrong happens by chance at a 20 % error rate); pooled they are not.
Writes debug/audit.json and debug/audit.html (the acceptance page, with a sign-off form).
"""

from __future__ import annotations

import html
import json
import os
import random

import numpy as np

WEIGHTS = dict(curb=1.0, height=0.5, qa=1.0, calib=1.0, boundary=0.5, confidence=1.0, repeat=1.0, spot_wrong=2.0,
               missed=1.0)  # fmt: skip


def _len(c):
    P = np.asarray(c, float)[:, :2]
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def scene_terms(dataset_dir: str, debug_dir: str, vectors: str, qa_dir: str | None, review: dict | None = None) -> dict:
    """risk terms of one scene (missing inputs -> term absent); review: agent_review.summary of its review."""
    t = {}
    gc = f"{debug_dir}/geom_check.json"
    if os.path.exists(gc):
        g = json.load(open(gc))
        if g.get("curb", {}).get("profiles"):
            t["curb"] = max(0.0, 0.8 - g["curb"]["step_within_0_3m"]) / 0.8
        if g.get("height", {}).get("points"):
            t["height"] = min(1.0, max(0.0, g["height"]["p95_cm"] - 10.0) / 20.0)
    sc = json.load(open(f"{dataset_dir}/scene.json"))
    t["qa"] = 1.0 - sc["qa_ok"] / max(sc["samples"], 1)
    st = [c.get("time_offset_status") for c in sc["cameras"].values()]
    t["calib"] = 1.0 if "failed" in st else (0.5 if any(s in ("coarse", "fallback") for s in st) else 0.0)
    feats = json.load(open(vectors))["features"]
    bl = [
        (_len(f["geometry"]["coordinates"]), f["properties"].get("observed"))
        for f in feats
        if f["properties"]["kind"] == "road_boundary"
    ]
    if bl and sum(x for x, _ in bl) > 0:
        t["boundary"] = sum(x for x, o in bl if not o) / sum(x for x, _ in bl)
    conf = [f["properties"]["confidence"] for f in feats if f["properties"].get("confidence") is not None]
    if conf:
        t["confidence"] = float(np.mean(np.array(conf) < 0.5))
    if qa_dir and os.path.exists(f"{qa_dir}/labels.json"):
        t["repeat"] = 1.0 - float(json.load(open(f"{qa_dir}/labels.json"))["agreement"])
    if review and review["spot"]:
        t["spot_wrong"] = review["spot_wrong"] / review["spot"]
    if review and review["tiles"]:
        t["missed"] = review["tiles_missed"] / review["tiles"]
    return {k: round(float(v), 3) for k, v in t.items()}


def risk(terms: dict, weights: dict = WEIGHTS) -> float:
    w = {k: weights[k] for k in terms if k in weights}
    return round(sum(terms[k] * w[k] for k in w) / max(sum(w.values()), 1e-9), 3)


def audit(
    scenes: list[dict], sample: int = 5, seed: int = 0, max_risk: float = 0.3, max_spot_error: float = 0.1
) -> dict:
    """scenes: dicts with scene, terms (+ review: agent_review.summary or None) -> ranking, random sample, verdict."""
    for s in scenes:
        s["risk"] = risk(s["terms"])
    ranked = sorted(scenes, key=lambda s: -s["risk"])
    rnd = random.Random(seed)
    pick = sorted(rnd.sample([s["scene"] for s in scenes], min(sample, len(scenes))))
    over = [s["scene"] for s in ranked if s["risk"] > max_risk]
    unreviewed = [s["scene"] for s in ranked if not s.get("review")]
    rejected = [s["scene"] for s in ranked if s.get("review") and s["review"]["reject"]]
    spot = sum(s["review"]["spot"] for s in scenes if s.get("review"))
    spot_wrong = sum(s["review"]["spot_wrong"] for s in scenes if s.get("review"))
    spot_m = sum(s["review"].get("spot_m", 0.0) for s in scenes if s.get("review"))
    spot_wrong_m = sum(s["review"].get("spot_wrong_m", 0.0) for s in scenes if s.get("review"))
    spot_error = round(spot_wrong_m / spot_m, 3) if spot_m else None  # by length: what a model learns from
    too_wrong = spot_error is not None and spot_error > max_spot_error
    return dict(rule=dict(max_risk=max_risk, max_spot_error=max_spot_error, sample=sample, seed=seed, weights=WEIGHTS),
                accept=not over and not unreviewed and not rejected and not too_wrong, over_max_risk=over,
                not_reviewed=unreviewed, rejected=rejected, spot=spot, spot_wrong=spot_wrong, spot_m=round(spot_m, 1),
                spot_wrong_m=round(spot_wrong_m, 1), spot_error=spot_error, too_wrong=too_wrong, random_sample=pick,
                scenes=ranked)  # fmt: skip


PAGE_STYLE = """body{font:14px/1.45 system-ui,sans-serif;margin:0;color:#1d1d1f;background:#fff}
main{max-width:1180px;margin:0 auto;padding:20px 16px} h1{font-size:22px;margin:0 0 8px} h2{font-size:17px;margin:28px 0 8px}
.verdict{padding:14px 16px;border-radius:10px;font-size:16px;margin:12px 0} .ready{background:#e6f4ea;border:1px solid #1e8e3e}
.notready{background:#fce8e6;border:1px solid #d93025} .kpi{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.kpi div{border:1px solid #d2d2d7;border-radius:10px;padding:10px 14px;min-width:150px} .kpi b{display:block;font-size:20px}
.kpi span{color:#6e6e73;font-size:12px} table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #e5e5ea;padding:5px 6px;text-align:right;white-space:nowrap} th{color:#6e6e73;font-weight:600}
td:first-child,th:first-child{text-align:left} tr.bad td{background:#fdecea} .mute{color:#6e6e73;font-size:12px}
details{border:1px solid #d2d2d7;border-radius:10px;margin:8px 0;padding:6px 12px} summary{cursor:pointer;font-weight:600}
figure{margin:8px 0} figure img{max-width:100%;border-radius:6px} figcaption{font-size:13px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px;margin-right:6px}
.t-drop{background:#d93025} .t-off{background:#e37400} .t-unsure{background:#7e57c2} .t-missed{background:#d93025} .t-ok{background:#1e8e3e}
.form{border:1px solid #d2d2d7;border-radius:10px;padding:14px;margin-top:10px} input[type=text]{font:inherit;padding:5px;border:1px solid #d2d2d7;border-radius:6px}
button{font:inherit;padding:8px 16px;border-radius:8px;border:0;background:#1e8e3e;color:#fff;cursor:pointer} code{background:#f5f5f7;padding:1px 4px;border-radius:4px}"""


def _pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def write_html(res: dict, path: str, link, name: str = "") -> None:
    """the acceptance page; link(scene) -> relative path of the scene's debug directory."""
    esc = html.escape
    keys = list(WEIGHTS)
    S = res["scenes"]
    rv = [s["review"] for s in S if s.get("review")]
    tot = lambda k: sum(r[k] for r in rv)  # noqa: E731
    if res["accept"]:
        verdict = "<div class='verdict ready'><b>Ready to accept.</b> Every scene is reviewed, none is rejected and none is over the risk limit. Look at the flagged items below, then sign off.</div>"
    else:
        why = []
        if res["not_reviewed"]:
            why.append(f"{len(res['not_reviewed'])} scenes not reviewed yet")
        if res["rejected"]:
            why.append(f"{len(res['rejected'])} scenes rejected ({', '.join(map(esc, res['rejected']))})")
        if res.get("too_wrong"):
            why.append(f"spot-check error {res['spot_error']:.0%} by length, limit {res['rule']['max_spot_error']:.0%}")
        if res["over_max_risk"]:
            why.append(f"{len(res['over_max_risk'])} scenes over the risk limit {res['rule']['max_risk']}")
        verdict = f"<div class='verdict notready'><b>Not ready:</b> {'; '.join(why)}.</div>"
    samples = sum(s.get("samples", 0) for s in S)
    qa_ok = sum(s.get("qa_ok", 0) for s in S if not (s.get("review") or {}).get("reject"))
    kpi = [
        (f"{len(rv)} / {len(S)}", "scenes reviewed"),
        (f"{qa_ok} / {samples}", "samples usable for training"),
        (
            _pct(res["spot_wrong_m"], res["spot_m"]),
            f"error of the unchecked data by length (random spot check: {res['spot_wrong_m']:.0f} of "
            f"{res['spot_m']:.0f} m wrong; limit {res['rule']['max_spot_error']:.0%})",
        ),
        (_pct(res["spot_wrong"], res["spot"]), f"spot-checked elements wrong ({res['spot_wrong']} of {res['spot']})"),
        (f"{tot('tiles_missed')} / {tot('tiles')}", "map tiles with missed paint / curbs"),
        (f"{tot('low_wrong')} / {tot('low')}", "uncertain elements found wrong (fixed)"),
        (str(tot("unsure")), "items the reviewer could not judge"),
    ]
    rows = []
    for s in S:
        r, d = s.get("review"), link(s["scene"])
        mark = " *" if s["scene"] in res["random_sample"] else ""
        bad = " class='bad'" if s["scene"] in res["over_max_risk"] or (r and r["reject"]) else ""
        rcell = (
            "<td colspan=5 class='mute'>not reviewed</td>"
            if not r
            else f"<td>{esc(str(r['reviewer']))}</td><td>{'REJECT' if r['reject'] else esc(str(r['verdict']))}</td>"
            f"<td>{r['spot_wrong']} / {r['spot']}</td><td>{r['tiles_missed']} / {r['tiles']}</td><td>{r['low_wrong']} / {r['low']}</td>"
        )
        cells = "".join(f"<td>{s['terms'].get(k, '')}</td>" for k in keys)
        rows.append(
            f"<tr{bad}><td><a href='#{esc(s['scene'])}'>{esc(s['scene'])}</a>{mark}</td><td><b>{s['risk']}</b></td>{rcell}{cells}"
            f"<td><input type='checkbox' class='rej' data-s='{esc(s['scene'])}' {'checked' if r and r['reject'] else ''}></td>"
            f"<td><a href='{d}/review.html'>open</a></td></tr>"
        )
    det = []
    for s in S:
        r, d = s.get("review"), link(s["scene"])
        if not r:
            continue
        figs = []
        for e in r["flagged"]:
            tag = e.get("decision") or e.get("result")
            what = "map tile" if "center" in e else e["kind"].replace("_", " ")
            note = e.get("note") or ""
            figs.append(f"<figure><figcaption><span class='tag t-{tag}'>{tag}</span>{esc(what)} {esc(note)}</figcaption>"
                        f"<img loading='lazy' src='{d}/{esc(e['evidence'])}'></figure>")  # fmt: skip
        ok = random.Random(s["scene"]).sample(r["ok"], min(3, len(r["ok"])))
        figs += [f"<figure><figcaption><span class='tag t-ok'>ok</span>{esc(e['kind'].replace('_', ' '))} "
                 f"(random check of the reviewer)</figcaption><img loading='lazy' src='{d}/{esc(e['evidence'])}'></figure>"
                 for e in ok]  # fmt: skip
        n_flag = len(r["flagged"])
        det.append(
            f"<details id='{esc(s['scene'])}' {'open' if n_flag and s['scene'] in res['random_sample'] else ''}>"
            f"<summary>{esc(s['scene'])}: {n_flag} flagged, risk {s['risk']}"
            f"{' - REJECTED' if r['reject'] else ''}</summary>"
            f"<p class='mute'>{esc(r['notes'])}</p>{''.join(figs) or '<p>nothing flagged</p>'}</details>"
        )
    th = "".join(f"<th>{k}</th>" for k in keys)
    open(path, "w").write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Acceptance</title>
<style>{PAGE_STYLE}</style></head><body><main>
<h1>Acceptance{": " + esc(name) if name else ""}</h1>{verdict}
<div class="kpi">{"".join(f"<div><b>{v}</b><span>{esc(t)}</span></div>" for v, t in kpi)}</div>
<h2>What to check</h2>
<ol><li>Below each scene: the items the reviewer deleted (<span class="tag t-drop">drop</span>), found off
(<span class="tag t-off">off</span>), could not judge (<span class="tag t-unsure">unsure</span>), map tiles with missed paint
(<span class="tag t-missed">missed</span>), and 3 random items it accepted (<span class="tag t-ok">ok</span>) to check the reviewer.
Each image: map, map + outline, then two camera views without / with the outline.</li>
<li>Scenes marked * are a fixed random sample: look at them even when nothing is flagged.</li>
<li>Tick <b>reject</b> for scenes that must not be used, then sign off at the bottom.</li></ol>
<h2>Scenes</h2><table><tr><th>scene</th><th>risk</th><th>reviewer</th><th>verdict</th><th>spot wrong</th><th>tiles missed</th>
<th>uncertain wrong</th>{th}<th>reject</th><th></th></tr>{"".join(rows)}</table>
<p class="mute">risk = weighted mean of the terms (0 fine .. 1 bad, hdlabel/audit.py); red = over the limit {res["rule"]["max_risk"]} or rejected.</p>
<h2>Evidence per scene</h2>{"".join(det) or "<p>no scene reviewed yet</p>"}
<h2>Sign-off</h2><div class="form">Your name <input type="text" id="by">
<label style="margin-left:12px"><input type="checkbox" id="accept" {"checked" if res["accept"] else ""}> I accept the batch (except the scenes ticked reject)</label>
<p><button id="save">Save sign-off</button></p><p class="mute">Saves <code>signoff.json</code>. Hand it to the agent, or run
<code>./autolabel signoff &lt;config&gt; --signoff signoff.json</code>: rejected scenes are taken out of the training data by the next run.</p></div>
</main><script>
document.getElementById("save").onclick = () => {{ const reject = {{}};
  document.querySelectorAll(".rej:checked").forEach(c => reject[c.dataset.s] = prompt("Why is " + c.dataset.s + " rejected?", "") || "");
  const out = {{by: document.getElementById("by").value, date: new Date().toISOString().slice(0, 10),
    accept: document.getElementById("accept").checked, reject}};
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}}));
  a.download = "signoff.json"; a.click(); }};
</script></body></html>""")
