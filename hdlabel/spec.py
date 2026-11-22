"""The label specification (spec/labels.yaml): the raster classes (raster_classes()), what each vector element is, its
geometry and attributes, and when it is correct. The verifier prompt and the benchmark review page render their judging rules from it (rules()), and
`validate` checks exported elements against it (conform()). Changing the spec changes the verifier prompt version, so
the next run verifies again under the new rules; `./autolabel spec` brings the tools in line (docs/AGENTS.md)."""

from __future__ import annotations

import hashlib
import os

import yaml

from hdlabel.common import ROOT

PATH = os.path.join(ROOT, "spec", "labels.yaml")


def load(path: str = PATH) -> dict:
    return yaml.safe_load(open(path))


def version(path: str = PATH) -> str:
    """content hash of the spec (what the tools were aligned to is recorded in tools/registry.yaml `spec`)."""
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:10]


def raster_classes(spec: dict | None = None) -> list[str]:
    """names of the label raster classes in id order (spec `raster`)."""
    rs = (spec or load())["raster"]
    assert [r["id"] for r in rs] == list(range(len(rs))), "spec raster ids must be 0, 1, 2, ... in order"
    return [r["name"] for r in rs]


def rules(groups=("marking", "curb"), spec: dict | None = None, html: bool = False) -> str:
    """the judging rules of the verified kinds, as text for a prompt (or an HTML table for a page)."""
    sp = spec or load()
    kinds = {k: v for k, v in sp["kinds"].items() if v.get("group") in groups}
    if html:
        import html as h

        e = h.escape
        rows = "".join(
            f"<tr><td><b>{e(k)}</b> ({e(v['group'])})<br><small>{e(v['definition'])}</small></td>"
            f"<td>{'<br>'.join(e(x) for x in v.get('correct', []))}</td>"
            f"<td>{'<br>'.join(e(x) for x in v.get('wrong', []))}</td></tr>"
            for k, v in kinds.items()
        )
        return (f"<p>{'<br>'.join(e(p) for p in sp.get('principles', []))}</p><table><tr><th>element</th><th>correct</th>"
                f"<th>wrong</th></tr>{rows}</table><p><b>cannot tell</b>: {e(sp.get('unsure', ''))}</p>")  # fmt: skip
    out = [f"- {p}" for p in sp.get("principles", [])]
    out.append("- wrong = `drop` when nothing real is there, `off` when it is there but the outline is beside it")
    for k, v in kinds.items():
        out.append(f"\n{k} ({v['group']}): {v['definition']}")
        out += [f"  - `ok`: {x}" for x in v.get("correct", [])]
        out += [f"  - wrong: {x}" for x in v.get("wrong", [])]
    out.append(f"\n`unsure`: {sp.get('unsure', '')}")
    return "\n".join(out)


def conform(features: list[dict], spec: dict | None = None) -> tuple[list[str], list[str]]:
    """(errors, warnings) of exported vector features against the spec: unknown kinds, wrong geometry, missing
    attributes are errors; attributes the spec does not name are warnings (one per kind and name)."""
    sp = spec or load()
    errors, warnings, seen = [], [], set()
    for i, f in enumerate(features):
        pr = f.get("properties", {})
        k = pr.get("kind")
        ks = sp["kinds"].get(k)
        if ks is None:
            if ("kind", k) not in seen:
                errors.append(f"kind {k!r} is not in the spec")
                seen.add(("kind", k))
            continue
        base = sp["kinds"].get(pr.get("candidate_of"), {}) if ks["geometry"] == "as candidate_of" else {}
        want = base.get("geometry", ks["geometry"])
        g = f.get("geometry", {}).get("type")
        if g != want and (k, "geometry") not in seen:
            errors.append(f"{k}: geometry {g}, spec says {want}")
            seen.add((k, "geometry"))
        req = set(ks.get("attributes") or {})
        for a in sorted(req - set(pr)):
            if (k, a) not in seen:
                errors.append(f"{k}: attribute {a!r} missing (element {i})")
                seen.add((k, a))
        known = req | set(ks.get("optional") or {}) | {"kind", "reviewed", "checked"}
        if "*" in (ks.get("optional") or {}):  # a candidate keeps the attributes of the kind it was
            known |= set(base.get("attributes") or {}) | set(base.get("optional") or {})
        for a in sorted(set(pr) - known):
            if (k, "+" + a) not in seen:
                warnings.append(f"{k}: attribute {a!r} is not in the spec")
                seen.add((k, "+" + a))
    return errors, warnings
