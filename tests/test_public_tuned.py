"""#86: blends whose weights were picked by public-LB probes (`public-tuned`) are not honest hedges. In UAT 13 such
blends had public anti-correlated with private (-0.59), and one became the hedge final."""

import datetime as dt

from kgkit import finals as FN
from kgkit import state as S
from kgkit.experiment import Ledger


def _ws(tmp_path, **kw):
    st = S.CompetitionState(slug="pg", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=2)).isoformat(), **kw)
    st.save(tmp_path)
    led = Ledger(tmp_path)
    led.attach_lb(led.log("stack", 0.96225)["id"], 0.96185)
    for i, (w, lb) in enumerate([("0.30", 0.96209), ("0.35", 0.96212), ("0.40", 0.96214)]):
        led.attach_lb(led.log(f"blend w{w}", None, tags=["lb-blend"])["id"], lb)
    return st, led


def test_untagged_probe_groups_are_listed_until_tagged(tmp_path):
    st, led = _ws(tmp_path)
    groups = FN.probe_groups(led.records())
    assert [sorted(r["id"] for r in g) for g in groups] == [["0002_blend-w0-30", "0003_blend-w0-35",
                                                             "0004_blend-w0-40"]]
    note = FN.tuned_lines(st, led.records(), probes=True)
    assert len(note) == 1 and "3 LB-scored blend variants differ only in numbers (0002_blend-w0-30 .. " in note[0]
    assert "ledger tag <id> public-tuned" in note[0]
    assert FN.tuned_lines(st, led.records()) == []  # the brief and status do not repeat it
    led.tag("0004", "public-tuned")
    assert FN.probe_groups(led.records()) == []


def test_a_public_tuned_file_is_not_the_best_public_candidate_nor_a_hedge(tmp_path):
    st, led = _ws(tmp_path)
    for i in ("0002", "0003", "0004"):
        led.tag(i, "public-tuned")
    st.final_picks = ["0001", "0004"]
    st.save(tmp_path)
    recs = led.records()
    sel = FN.selection(st, recs)
    honest = dict(sel["honest"])
    assert honest["0001_stack"] == ["best CV", "best public score neither fitted nor public-tuned"]
    assert "0004_blend-w0-40" not in honest
    lines = FN.lines(tmp_path, st, recs)
    note = next(line for line in lines if line.startswith("note: final pick(s) 0004_blend-w0-40 are public-tuned"))
    assert "not a hedge" in note and "The CV-chosen alternative: 0001_stack (CV 0.96225, LB 0.96185)." in note
    # an accepted baseline that is public-tuned is still a candidate, marked
    led.decide("0001", "baseline")
    led.tag("0001", "public-tuned")
    assert "public-tuned: not a hedge" in dict(FN.selection(st, led.records())["honest"])["0001_stack"]
