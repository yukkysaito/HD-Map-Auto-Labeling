"""Turn a reviewer's decisions on an evidence pack (debug/scenes/<scene>/check/, hdlabel/evidence.py) into the scene's
review file, and a person's sign-off of the acceptance page into the review files. docs/AGENT_REVIEW.md has the
procedure and the formats.

decisions.json (written by the reviewer, an AI agent or a person):
  {"reviewer": "agent:<model>",
   "items": {"0": "ok", "1": "drop: shadow", "2": "off: outline 0.5 m beside the paint", "3": "unsure: occluded"},
   "tiles": {"0": "ok", "1": "missed: dashed line on the left not outlined"},
   "verdict": "accept" | "reject", "notes": "..."}
  item: ok = correct; drop = no paint / curb there (deleted by the next run); off = there, but the outline is clearly
  off; unsure = cannot tell (a person looks at it in the acceptance page). Text after ":" is kept as the note.
signoff.json (saved by the acceptance page): {"by": "...", "date": "...", "accept": true, "reject": {"<scene>": "why"}}
"""

from __future__ import annotations

import datetime
import json
import os
import shutil

DECISIONS = ("ok", "drop", "off", "unsure")


def _parse(v: str) -> tuple[str, str]:
    d, _, note = str(v).partition(":")
    d = d.strip().lower()
    if d not in DECISIONS and d not in ("missed",):
        raise ValueError(f"unknown decision {v!r} (expected one of {DECISIONS} or 'missed: ...' for tiles)")
    return d, note.strip()


def _same(e: dict, kind: str | None, at) -> bool:
    return (
        e.get("kind") == kind and e.get("at") is not None and abs(e["at"][0] - at[0]) + abs(e["at"][1] - at[1]) < 0.02
    )


def apply(pack_dir: str, decisions: dict, review_path: str, keep_dir: str, partial: bool = False) -> dict:
    """merge the decisions on pack_dir/items.json into review_path; evidence images of the decided items / tiles are
    copied to keep_dir (they outlive the pack, which is rewritten by every run). Returns the review."""
    pack = json.load(open(f"{pack_dir}/items.json"))
    items, tiles = decisions.get("items", {}), decisions.get("tiles", {})
    missing = [str(it["n"]) for it in pack["items"] if str(it["n"]) not in items]
    missing += [f"tile {t['n']}" for t in pack["tiles"] if str(t["n"]) not in tiles]
    if missing and not partial:
        raise ValueError(f"no decision for {', '.join(missing)} (use partial to apply anyway)")
    if decisions.get("verdict") not in ("accept", "reject"):
        raise ValueError("verdict must be 'accept' or 'reject'")
    rv = json.load(open(review_path)) if os.path.exists(review_path) else {}
    for k in ("checked", "fixes", "issues", "tiles"):
        rv.setdefault(k, [])
    os.makedirs(keep_dir, exist_ok=True)

    def keep(src, name):
        shutil.copyfile(f"{pack_dir}/{src}", f"{keep_dir}/{name}")
        return f"{os.path.basename(os.path.normpath(keep_dir))}/{name}"

    for it in pack["items"]:
        if str(it["n"]) not in items:
            continue
        d, note = _parse(items[str(it["n"])])
        kind, at = it["kind"], it["at"]
        for k in ("checked", "fixes", "issues"):
            rv[k] = [e for e in rv[k] if not (_same(e, kind, at) and (k != "fixes" or e.get("drop")))]
        ev = keep(it["image"], f"{kind}_{at[0]:.2f}_{at[1]:.2f}.jpg")
        base = dict(kind=kind, at=at, why=it["reason"], decision=d, size_m=it.get("size_m"), evidence=ev)
        if d == "drop":
            rv["fixes"].append(dict(base, drop=True, **({"note": note} if note else {})))
        else:
            rv["checked"].append(base)
        if d in ("off", "unsure"):
            rv["issues"].append(dict(kind=kind, at=at, note=f"{d}: {note}" if note else d, evidence=ev))
    for t in pack["tiles"]:
        if str(t["n"]) not in tiles:
            continue
        d, note = _parse(tiles[str(t["n"])])
        ev = keep(t["image"], f"tile_{t['center'][0]:.2f}_{t['center'][1]:.2f}.jpg")
        rv["tiles"] = [x for x in rv["tiles"] if x["center"] != t["center"]]
        rv["issues"] = [e for e in rv["issues"] if not _same(e, None, t["center"])]
        rv["tiles"].append(dict(center=t["center"], result="missed" if d == "missed" else "ok", note=note, evidence=ev))
        if d == "missed":
            rv["issues"].append(dict(at=t["center"], note=f"missed: {note}", evidence=ev))
    reject = decisions["verdict"] == "reject"
    rv.update(
        reviewer=decisions.get("reviewer", "agent"),
        date=datetime.date.today().isoformat(),
        reject=reject or bool(rv.get("rejected_by")),  # a person's reject (signoff) stays
        verdict="bad" if reject else ("fixed" if any(f.get("drop") for f in rv["fixes"]) or rv["issues"] else "ok"),
    )
    if decisions.get("notes"):
        rv["notes"] = decisions["notes"]
    os.makedirs(os.path.dirname(os.path.abspath(review_path)), exist_ok=True)
    json.dump(rv, open(review_path, "w"), indent=1)
    return rv


def summary(rv: dict | None) -> dict | None:
    """what the acceptance page shows about a scene's review."""
    if not rv or "checked" not in rv:
        return None
    dec = [e for e in rv["checked"] + [f for f in rv["fixes"] if f.get("drop")] if e.get("decision")]
    spot = [e for e in dec if e.get("why") == "random spot check"]
    low = [e for e in dec if e.get("why") == "low confidence"]
    wrong = lambda es: sum(e["decision"] in ("drop", "off") for e in es)  # noqa: E731
    judged = [e for e in spot if e["decision"] != "unsure"]
    size = lambda es: round(sum(e.get("size_m") or 0.0 for e in es), 2)  # noqa: E731
    return dict(
        reviewer=rv.get("reviewer"),
        verdict=rv.get("verdict"),
        reject=bool(rv.get("reject")),
        rejected_by=rv.get("rejected_by"),
        low=len(low),
        low_wrong=wrong(low),
        spot=len(spot),
        spot_wrong=wrong(spot),
        spot_m=size(judged),
        spot_wrong_m=size([e for e in judged if e["decision"] in ("drop", "off")]),
        unsure=sum(e["decision"] == "unsure" for e in dec),
        tiles=len(rv.get("tiles", [])),
        tiles_missed=sum(t["result"] == "missed" for t in rv.get("tiles", [])),
        notes=rv.get("notes", ""),
        flagged=[e for e in dec if e["decision"] != "ok"] + [t for t in rv.get("tiles", []) if t["result"] != "ok"],
        ok=[e for e in dec if e["decision"] == "ok"],
    )


def signoff(signoff: dict, review_file, out_path: str) -> list[str]:
    """record a person's acceptance: scenes they reject get reject = true in their review file (export then drops their
    samples); the sign-off itself is kept at out_path. review_file(scene) -> path. Returns the rejected scenes."""
    rejected = sorted(signoff.get("reject", {}))
    for s in rejected:
        p = review_file(s)
        rv = json.load(open(p)) if os.path.exists(p) else {}
        rv.update(reject=True, verdict="bad", rejected_by=signoff.get("by", "person"), reject_note=signoff["reject"][s])
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        json.dump(rv, open(p, "w"), indent=1)
    json.dump(signoff, open(out_path, "w"), indent=1)
    return rejected
