"""Fewer bookkeeping calls per experiment (#127): the templates compare a run with its parent when they log it, and
`kgkit exp finish` records the decision and closes the backlog item in one call."""

import json

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import permissions as PM
from kgkit.backlog import Backlog
from kgkit.experiment import Ledger, report_comparison


@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    return tmp_path


def _preds(binary_df, noise_base=1.0, noise_new=0.5):
    rng = np.random.default_rng(3)
    signal = 1.5 * binary_df["x1"] - binary_df["x2"]
    return signal + rng.normal(0, noise_base, len(signal)), signal + rng.normal(0, noise_new, len(signal))


def _import_pair(ws, binary_df, capsys, **kw):
    """0001_base (the accepted baseline) and 0002_new (less noise: a real gain), logged as a template logs them."""
    base, new = _preds(binary_df, **kw)
    np.save(ws / "base.npy", base)
    np.save(ws / "new.npy", new)
    spec = ["--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold"]
    cli.main(["ledger", "import", "base", "base.npy", *spec])
    cli.main(["ledger", "decide", "0001", "baseline"])
    cli.main(["ledger", "import", "new", "new.npy", *spec])
    Ledger(ws).update("0002", parent="0001_base")
    capsys.readouterr()


def _comparisons(ws) -> list[dict]:
    p = ws / ".kaggle-gm" / "comparisons.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def test_exp_finish_records_the_decision_and_closes_the_backlog_item(ws, binary_df, capsys):
    _import_pair(ws, binary_df, capsys)
    bl = Backlog(ws)
    bl.add("denoise the signal", gain=3, prob=0.5, cost=1, status="running")
    cli.main(["exp", "finish", "0002", "keep", "less", "noise", "--backlog", "1"])
    out = capsys.readouterr().out
    assert "0002_new vs 0001_base: KEEP" in out
    assert "0002_new: keep (vs 0001_base)" in out and "accepted baseline is now 0002_new" in out
    rec = Ledger(ws).get("0002")
    assert rec["decision"] == "keep" and rec["parent"] == "0001_base" and rec["decision_note"] == "less noise"
    item = Backlog(ws).items()[0]
    assert item["status"] == "done" and item["exp_ids"] == ["0002_new"]
    assert item["result"].startswith("keep: less noise") and "#1" in out


def test_exp_finish_without_a_decision_suggests_one_and_records_nothing(ws, binary_df, capsys):
    _import_pair(ws, binary_df, capsys)
    Backlog(ws).add("denoise", status="running")
    cli.main(["exp", "finish", "0002", "--backlog", "1"])
    out = capsys.readouterr().out
    assert "KEEP" in out and 'kgkit exp finish 0002_new keep "<why>" --backlog 1' in out
    assert Ledger(ws).get("0002").get("decision") is None
    assert Backlog(ws).items()[0]["status"] == "running"


def test_exp_finish_reuses_the_comparison_made_when_the_run_was_logged(ws, binary_df, capsys):
    """The template (or `ledger add --pred`) already compared the run: finishing it re-runs no bootstrap, and the
    measured noise is logged once, not once per call."""
    _import_pair(ws, binary_df, capsys)
    rec = Ledger(ws).get("0002")
    report_comparison(rec, truth="data/train.csv:target")
    out = capsys.readouterr().out
    assert "0002_new vs 0001_base: KEEP" in out and "kgkit exp finish 0002_new keep" in out
    n = len(_comparisons(ws))
    assert n == 1 and _comparisons(ws)[-1]["k"] == 5
    cli.main(["exp", "finish", "0002", "keep"])
    out = capsys.readouterr().out
    assert "compared when it was logged" in out and "KEEP" in out and "folds 5/5" in out
    assert len(_comparisons(ws)) == n
    # without a note, the decision records the numbers it rests on
    note = Ledger(ws).get("0002")["decision_note"]
    assert "gain +" in note and "folds 5/5" in note
    # options of the paired test ask for a fresh comparison
    cli.main(["exp", "finish", "0002", "--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold",
              "--boot", "50"])
    assert "compared when it was logged" not in capsys.readouterr().out and len(_comparisons(ws)) == n + 1


def test_exp_finish_finds_the_backlog_item_that_names_the_experiment(ws, binary_df, capsys):
    _import_pair(ws, binary_df, capsys)
    bl = Backlog(ws)
    bl.add("other idea", status="running")
    i = bl.add("denoise", status="running")
    bl.set(i["id"], exp="0002_new")
    cli.main(["exp", "finish", "0002", "discard", "not worth it"])
    out = capsys.readouterr().out
    items = {it["id"]: it for it in Backlog(ws).items()}
    assert items[i["id"]]["status"] == "done" and items[i["id"]]["result"].startswith("discard")
    assert items[1]["status"] == "running"
    assert "against the comparison's KEEP" in out  # a decision other than the verdict says so


def test_exp_finish_says_which_backlog_items_are_open_when_none_names_it(ws, binary_df, capsys):
    _import_pair(ws, binary_df, capsys)
    Backlog(ws).add("denoise", status="running")
    cli.main(["exp", "finish", "0002", "keep"])
    out = capsys.readouterr().out
    assert "backlog not updated" in out and "--backlog <n>" in out and "#1" in out
    assert Backlog(ws).items()[0]["status"] == "running"


def test_exp_finish_inconclusive_leaves_the_idea_running(ws, binary_df, capsys):
    _import_pair(ws, binary_df, capsys, noise_base=1.0, noise_new=0.99)
    Backlog(ws).add("tiny change", status="running")
    cli.main(["exp", "finish", "0002", "inconclusive", "second seed next", "--backlog", "1"])
    item = Backlog(ws).items()[0]
    assert item["status"] == "running" and item["result"].startswith("inconclusive") and item["exp_ids"] == ["0002_new"]
    assert Ledger(ws).get("0002")["decision"] == "inconclusive"


def test_exp_finish_baseline_and_nothing_to_compare_with(ws, binary_df, capsys):
    base, _ = _preds(binary_df)
    np.save(ws / "base.npy", base)
    cli.main(["ledger", "import", "base", "base.npy", "--truth", "data/train.csv:target"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:  # nothing accepted yet: a keep would rest on no comparison
        cli.main(["exp", "finish", "0001", "keep"])
    assert "no accepted baseline" in str(e.value) and "kgkit exp finish 0001_base baseline" in str(e.value)
    cli.main(["exp", "finish", "0001", "baseline", "first model"])
    assert "0001_base: baseline" in capsys.readouterr().out
    assert Ledger(ws).baseline(fallback=False)["id"] == "0001_base"
    with pytest.raises(SystemExit) as e:
        cli.main(["exp", "finish", "0001", "maybe"])
    assert "keep|discard|inconclusive|baseline" in str(e.value)


def test_report_comparison_compares_with_the_accepted_baseline_and_never_raises(ws, binary_df, capsys, monkeypatch):
    _import_pair(ws, binary_df, capsys)
    led = Ledger(ws)
    led.update("0002", parent=None)
    v = report_comparison(led.get("0002"))  # no parent: the accepted baseline; no truth given: the target's column
    out = capsys.readouterr().out
    assert v["decision"] == "KEEP" and v["base"] == "0001_base" and "rows)" in out  # row-level test
    # a parent this ledger does not know (a remote run's own ledger): one line, no exception
    assert report_comparison(led.get("0002"), parent="0099") is None
    assert "not compared" in capsys.readouterr().out
    # the baseline itself, and a re-check on re-drawn folds, are not compared
    assert report_comparison(led.get("0001")) is None
    monkeypatch.setenv("KG_RECHECK", "seed-7")
    assert report_comparison(led.get("0002")) is None and capsys.readouterr().out == ""


def test_report_comparison_before_any_baseline_says_how_to_set_one(ws, binary_df, capsys):
    base, _ = _preds(binary_df)
    np.save(ws / "base.npy", base)
    cli.main(["ledger", "import", "base", "base.npy", "--truth", "data/train.csv:target"])
    capsys.readouterr()
    assert report_comparison(Ledger(ws).get("0001")) is None
    assert "kgkit exp finish 0001_base baseline" in capsys.readouterr().out


def test_report_comparison_falls_back_to_fold_scores_when_the_truth_does_not_fit(ws, binary_df, capsys):
    """A default truth that does not line up with the OOF (another row count) must not cost the verdict."""
    _import_pair(ws, binary_df, capsys)
    pd.DataFrame({"target": [0, 1] * 10}).to_csv(ws / "data" / "short.csv", index=False)
    v = report_comparison(Ledger(ws).get("0002"), truth="data/short.csv:target", fallback=True)
    out = capsys.readouterr().out
    assert v is not None and v["decision"] == "KEEP" and "folds:" in out and "rows)" not in out


def test_exp_is_a_bookkeeping_command_the_permissions_allow():
    assert "Bash(kgkit exp *)" in PM.rules()
