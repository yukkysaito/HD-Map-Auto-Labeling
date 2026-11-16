"""Verification agent: independent check of a scene's labels -> accepted | needs_revision | rejected | uncertain.

Independent of the annotation: it sees only the sensor evidence (evidence pack: map + camera views per item, map tiles;
hdlabel/evidence.py) and the fixed criteria (prompts/verifier.md), never how the labels were made; by default it runs on
another vendor's model than the annotator (config agents.roles.verifier). Its visual judgement is combined with
measurements that need no AI: LiDAR height steps along curbs (geom_check.py), vector height vs LiDAR ground, the
spot-check error rate.

Per scene it writes
  review/<name>/<scene>.json                    the decisions (hdlabel/agent_review.py format: drops become fixes)
  debug/scenes/<scene>/verification.json        state, measurements and structured issues:
      {id, kind, at, type (false_positive | offset | missed | uncertain | systematic | lidar_conflict), evidence,
       confidence, source (vision | lidar | geometry), action (drop | rerun | tool | person)}
State rules (deterministic, from the decisions and measurements, see scene_state):
  rejected        systematic failure: >= 5 of 10 spot checks wrong, or curbs systematically off the LiDAR steps
  needs_revision  decided deletions not yet applied to the vectors (the next label run applies them)
  uncertain       the verifier could not run, or more than verification.max_unsure items are unsure (a person looks)
  accepted        otherwise
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

from hdlabel.agents.llm import Agents, ProviderError

PROMPTS = os.path.join(os.path.dirname(__file__), "prompts")
STATES = ("accepted", "needs_revision", "rejected", "uncertain")
DECISIONS = ("ok", "drop", "off", "unsure")


def prompt_version(name="verifier") -> str:
    return hashlib.sha1(open(f"{PROMPTS}/{name}.md", "rb").read()).hexdigest()[:10]


def _item_prompt(rows: list[dict]) -> str:
    lines = [
        f"Image {k + 1} = item {r['key']}: {r['kind']} ({r.get('reason', '')}), confidence {r.get('confidence')}, "
        f"size {r.get('size_m')} m"
        for k, r in enumerate(rows)
    ]
    keys = ", ".join(f'"{r["key"]}"' for r in rows)
    return (
        "Items to decide (one image each, in this order):\n" + "\n".join(lines) + "\n\nAnswer with JSON only: "
        '{"items": {"<item>": {"decision": "ok|drop|off|unsure", "confidence": 0.0-1.0, "reason": "..."}}} '
        f"with the items {keys}."
    )


TILE_PROMPT = (
    "Each image is a 30 m map square: left without, right with every extracted outline (road marking yellow, curb "
    "blue). Is there clear paint on the road surface (at least ~1 m long) or a clear kerb in the left panel that has no "
    "outline in the right panel? Black (unobserved) areas and paint on kerbs or sidewalks do not count.\n"
    'Answer with JSON only: {"tiles": {"<tile>": {"missed": false} | {"missed": true, "what": "...", "where": "...", '
    '"length_m": 3}}} for the tiles '
)


def judge_items(agents: Agents, rows: list[dict], role="verifier", batch=5, jobs=4) -> dict:
    """rows: [{key, image, kind, reason, confidence, size_m}] -> {key: {decision, confidence, reason}}; items whose
    call failed get decision unsure with the error."""
    system = open(f"{PROMPTS}/verifier.md").read()
    chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]

    def one(ch):
        try:
            rep = agents.ask(role, _item_prompt(ch), [r["image"] for r in ch], system=system)
            got = rep.json().get("items", {})
            out = {}
            for r in ch:
                g = got.get(str(r["key"])) or got.get("*") or {}  # "*": one answer for all (replay)
                d = str(g.get("decision", "unsure")).lower()
                out[r["key"]] = dict(decision=d if d in DECISIONS else "unsure",
                                     confidence=float(g.get("confidence") or 0.0), reason=str(g.get("reason", ""))[:300],
                                     model=f"{rep.provider}/{rep.model}")  # fmt: skip
            return out
        except (ProviderError, ValueError, AttributeError) as e:
            return {r["key"]: dict(decision="unsure", confidence=0.0, reason=f"verifier failed: {e}"[:300], error=True)
                    for r in ch}  # fmt: skip

    res = {}
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for o in pool.map(one, chunks):
            res.update(o)
    return res


def judge_confirmed(agents: Agents, rows: list[dict], cfg: dict) -> dict:
    """the verifier's decisions; a deletion (drop / off) also needs the confirmer (another model, config
    verification.confirm_role) to call the item wrong - if it says ok, the item becomes unsure (a person decides).
    Measured on the benchmark: one model alone deletes too many correct elements (bench status / verifier)."""
    role, conf = cfg.get("role", "verifier"), cfg.get("confirm_role")
    res = judge_items(agents, rows, role, jobs=cfg.get("jobs", 4))
    wrong = [r for r in rows if res[r["key"]]["decision"] in ("drop", "off")]
    if conf and wrong and agents.available(conf):
        res2 = judge_items(agents, wrong, conf, jobs=cfg.get("jobs", 4))
        for r in wrong:
            a, b = res[r["key"]], res2[r["key"]]
            a["confirm"] = dict(decision=b["decision"], reason=b["reason"], model=b.get("model"))
            if b.get("error"):  # no confirmation, no deletion
                a.update(decision="unsure", reason=f"not confirmed ({b['reason'][:80]}): {a['reason']}"[:300])
            elif b["decision"] == "ok":
                a.update(
                    decision="unsure",
                    reason=f"verifiers disagree: {a['reason']} | other model: ok, {b['reason']}"[:300],
                )
    return res


def judge_tiles(agents: Agents, tiles: list[dict], role="verifier", batch=4) -> dict:
    """tiles: [{key, image}] -> {key: "ok" | "missed: what, where"}."""
    system = open(f"{PROMPTS}/verifier.md").read()
    out = {}
    for i in range(0, len(tiles), batch):
        ch = tiles[i : i + batch]
        try:
            rep = agents.ask(role, TILE_PROMPT + ", ".join(f'"{t["key"]}"' for t in ch) + " (one image each, in order).",
                             [t["image"] for t in ch], system=system)  # fmt: skip
            got = rep.json().get("tiles", {})
            for t in ch:
                g = got.get(str(t["key"])) or got.get("*") or {}
                out[t["key"]] = (f"missed: {g.get('what', '')} {g.get('where', '')} (~{g.get('length_m', '?')} m)"
                                 if g.get("missed") else "ok")  # fmt: skip
        except (ProviderError, ValueError, AttributeError) as e:
            for t in ch:
                out[t["key"]] = f"error: {e}"[:200]
    return out


def lidar_index(geom: dict | None) -> list[dict]:
    return ((geom or {}).get("curb") or {}).get("per_curb") or []


def lidar_of(per_curb: list[dict], at) -> dict | None:
    return next((c for c in per_curb if abs(c["at"][0] - at[0]) + abs(c["at"][1] - at[1]) < 0.05), None)


def present(features: list[dict], kind: str, at) -> bool:
    """is the element (kind at `at`, element_at) still in the vectors?"""
    from hdlabel.review_html import element_at

    return any(f["properties"]["kind"] == kind and sum(abs(a - b) for a, b in zip(element_at(f), at)) < 0.05
               for f in features)  # fmt: skip


def scene_state(review: dict | None, geom: dict | None, features: list[dict], cfg: dict, verifier_ok=True):
    """(state, reasons) from the scene's review file, its LiDAR checks and current vectors (rules: module doc)."""
    from hdlabel.agent_review import summary

    s = summary(review)
    reasons = []
    if s is None:
        return ("uncertain", ["not verified"]) if not verifier_ok else ("uncertain", ["no review"])
    curb = (geom or {}).get("curb") or {}
    if s["spot"] and s["spot_wrong"] >= cfg.get("reject_spot_wrong", 5):
        reasons.append(f"{s['spot_wrong']} of {s['spot']} spot checks wrong")
    if (curb.get("profiles") or 0) >= 50 and (curb.get("step_within_0_3m") or 1) < cfg.get("min_curb_step", 0.35) \
            and (curb.get("median_offset_m") or 0) > 0.3:  # fmt: skip
        reasons.append(
            f"curbs off the LiDAR steps (step share {curb['step_within_0_3m']}, offset {curb['median_offset_m']} m)"
        )
    if review.get("reject"):
        reasons.append("rejected by " + (review.get("rejected_by") or review.get("reviewer") or "reviewer"))
    if reasons:
        return "rejected", reasons
    pending = [f for f in review.get("fixes", []) if f.get("drop") and present(features, f["kind"], f["at"])]
    if pending:
        return "needs_revision", [f"{len(pending)} deletions not applied yet"]
    if not verifier_ok:
        return "uncertain", ["verifier unavailable"]
    if s["unsure"] > cfg.get("max_unsure", 5):
        return "uncertain", [f"{s['unsure']} items unsure"]
    return "accepted", [f"{s['spot']} spot checks, {s['spot_wrong']} wrong; {s['unsure']} unsure"]


def issues_of(review: dict, per_curb: list[dict], scene: str) -> list[dict]:
    out = []
    for e in review.get("checked", []) + [f for f in review.get("fixes", []) if f.get("drop")]:
        d = e.get("decision")
        lid = lidar_of(per_curb, e["at"]) if e["kind"] == "curb" else None
        if d in ("drop", "off", "unsure"):
            out.append(dict(kind=e["kind"], at=e["at"], type={"drop": "false_positive", "off": "offset"}.get(d, "uncertain"),
                            evidence=e.get("evidence"), note=e.get("note", ""), confidence=e.get("confidence"),
                            source="vision", action={"drop": "drop", "off": "tool"}.get(d, "person")))  # fmt: skip
        if (
            lid
            and lid["profiles"] >= 5
            and ((d == "ok" and lid["step_share"] == 0) or (d == "drop" and lid["step_share"] >= 0.8))
        ):
            out.append(dict(kind="curb", at=e["at"], type="lidar_conflict", source="lidar", action="person",
                            evidence=e.get("evidence"), note=f"vision {d}, LiDAR step share {lid['step_share']} of {lid['profiles']} profiles"))  # fmt: skip
    for t in review.get("tiles", []):
        if t["result"] == "missed":
            out.append(dict(kind=None, at=t["center"], type="missed", evidence=t.get("evidence"), note=t.get("note"),
                            source="vision", action="tool"))  # fmt: skip
    for i, x in enumerate(out):
        x["id"] = f"{scene}#{i}"
    return out


def verify_scene(agents: Agents, pack_dir: str, review_path: str, keep_dir: str, geom_path: str, vectors_path: str,
                 out_path: str, cfg: dict) -> dict:  # fmt: skip
    """judge the open items / tiles of the pack, apply the decisions, write verification.json; returns it."""
    from hdlabel.agent_review import apply

    pack = json.load(open(f"{pack_dir}/items.json"))
    if os.path.exists(review_path):  # the pack may predate decisions made since (it is rebuilt by the debug step)
        from hdlabel.review_html import decided

        rv0 = json.load(open(review_path))
        pack["items"] = [r for r in pack["items"] if not decided(rv0, r["kind"], r["at"])]
        done_tiles = [t["center"] for t in rv0.get("tiles", [])]
        pack["tiles"] = [t for t in pack["tiles"] if t["center"] not in done_tiles]
    geom = json.load(open(geom_path)) if os.path.exists(geom_path) else None
    role = cfg.get("role", "verifier")
    verifier_ok = agents.available(role)
    decided = {}
    if (pack["items"] or pack["tiles"]) and verifier_ok:
        rows = [dict(r, key=str(r["n"]), image=f"{pack_dir}/{r['image']}") for r in pack["items"]]
        items = judge_confirmed(agents, rows, cfg)
        tiles = judge_tiles(
            agents, [dict(key=str(t["n"]), image=f"{pack_dir}/{t['image']}") for t in pack["tiles"]], role
        )
        failed = sum(bool(v.get("error")) for v in items.values()) + sum(v.startswith("error") for v in tiles.values())
        if failed:
            verifier_ok = False
        dec = dict(reviewer=f"agent:{agents.model_of(role)['provider']}/{agents.model_of(role)['model']}"
                            f"@verifier-{prompt_version()}",
                   items={k: v["decision"] + (f": {v['reason']}" if v["reason"] else "") for k, v in items.items()},
                   tiles={k: v for k, v in tiles.items() if not v.startswith("error")},
                   verdict="accept", notes="")  # fmt: skip
        rv = apply(pack_dir, dec, review_path, keep_dir, partial=bool(failed))
        decided = dict(items=items, tiles=tiles)
        # the confidence of each decision is kept with the checked entry
        for e in rv.get("checked", []) + rv.get("fixes", []):
            for r in pack["items"]:
                if e.get("at") == r["at"] and e.get("kind") == r["kind"] and str(r["n"]) in items:
                    e["confidence"] = items[str(r["n"])]["confidence"]
        json.dump(rv, open(review_path, "w"), indent=1)
    review = json.load(open(review_path)) if os.path.exists(review_path) else None
    feats = json.load(open(vectors_path))["features"] if os.path.exists(vectors_path) else []
    state, reasons = scene_state(review, geom, feats, cfg, verifier_ok)
    if review is not None and review.get("verdict") != "bad":  # the deterministic state decides a reject
        review["reject"] = state == "rejected" or bool(review.get("rejected_by"))
        json.dump(review, open(review_path, "w"), indent=1)
    res = dict(scene=pack["scene"], state=state, reasons=reasons, verifier=agents.model_of(role),
               prompt=prompt_version(), geometry={k: v for k, v in ((geom or {}).get("curb") or {}).items() if k != "per_curb"}
               | {"height": (geom or {}).get("height")},
               issues=issues_of(review or {}, lidar_index(geom), pack["scene"]), new_decisions=decided)  # fmt: skip
    json.dump(res, open(out_path, "w"), indent=1)
    return res
