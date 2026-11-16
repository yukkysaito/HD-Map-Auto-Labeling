"""review.html: which elements a reviewer is asked about, and the page is written with them."""

import json

from hdlabel import review_html


def line(kind, x, **props):
    return dict(
        geometry=dict(type="LineString", coordinates=[[x, 0, 0], [x, 4, 0]]), properties=dict(kind=kind, **props)
    )


def test_needs_check_only_uncertain_extraction():
    nc = review_html.needs_check
    assert nc(dict(kind="arrow", confidence=0.2)) and nc(dict(kind="curb", confidence=0.4))
    assert not nc(dict(kind="arrow", confidence=0.9)) and not nc(dict(kind="lane_line", confidence=0.1, reviewed=True))
    assert not nc(dict(kind="road_boundary", observed=True)) and not nc(dict(kind="marking_area"))


def test_queue_low_confidence_first_then_spot_check(tmp_path):
    feats = [line("lane_line", 30, confidence=0.2), line("arrow", 20, confidence=0.3), line("curb", 0, confidence=0.1)]
    feats += [line("lane_line", 100 + 10 * k, confidence=0.9) for k in range(20)] + [
        line("road_boundary", 5, observed=True)
    ]
    p = tmp_path / "review.html"
    review_html.write(str(p), "s", "ortho.jpg", (100, 100), [0, 0], 0.5, [0, 0, 0], review_html.elements(feats), None,
                      [dict(file="sample_0000.jpg", xy=[0, 0])], "review/x/s.json")  # fmt: skip
    text = p.read_text()
    data = json.loads(text.split("const D = ", 1)[1].split(", NS =", 1)[0])
    q, why = data["queue"], data["reasons"]
    assert [data["elements"][i]["at"][0] for i in q[:3]] == [0, 20, 30]  # nearest-neighbour walk
    assert {why[str(i)] for i in q[:3]} == {"low confidence"}
    assert len(q) == 3 + review_html.SPOT_CHECK and {why[str(i)] for i in q[3:]} == {"random spot check"}
    again = review_html.check_queue(data["elements"], "s")[0]
    assert again == q  # reproducible per scene
    assert data["review_path"] == "review/x/s.json"
