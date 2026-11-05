"""`kgkit audit-repo`: competition data in git (UAT #69: a subset of train.csv, report text and expert labels, sat in
a public repo for 2.5 months; the repo's .gitignore named the downloaded files only)."""

import csv
import json
import random
import shutil
import subprocess

import pytest

from kgkit import audit as AU
from kgkit import cli
from kgkit import state as S

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

LABELS = ["ACL", "MCL", "Effusion"]
BOILER = "No acute osseous abnormality is identified on this examination today."
WORDS = ("meniscus ligament cartilage effusion marrow edema tear fluid signal joint lateral medial posterior anterior "
         "horn root tibial femoral patellar intact thickened partial").split()


def git(root, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false",
                           "-c", "commit.gpgsign=false", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout


def commit(root, msg="change"):
    git(root, "add", "-A")
    git(root, "commit", "-qm", msg)


def _csv(path, header, rows, newline="\n"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator=newline)
        w.writerow(header)
        w.writerows(rows)


def _report(rng, i):
    s = [f"Study {i} shows " + " ".join(rng.choice(WORDS) for _ in range(9)) + "." for _ in range(4)]
    return f"{s[0]} {s[1]}\n{BOILER} {s[2]} {s[3]}"


@pytest.fixture
def comp(tmp_path, monkeypatch):
    """A competition repo like the UAT one: its .gitignore names the downloaded files and nothing else."""
    root = tmp_path / "comp"
    root.mkdir()
    git(root, "init", "-q")
    S.CompetitionState(slug="knee", metric="auc", id_col="StudyInstanceUID", target=",".join(LABELS)).save(root)
    rng = random.Random(0)
    rows = []
    for i in range(60):  # the first 20 studies have expert labels, the rest only a report
        labels = [rng.choice("01") for _ in LABELS] if i < 20 else ["", "", ""]
        rows.append([f"1.2.826.0.1.{3680043 + i}.8.498.{10**12 + i * 7919}", _report(rng, i), *labels])
    data = root / "data"
    data.mkdir()
    _csv(data / "train.csv", ["StudyInstanceUID", "Report", *LABELS], rows)
    _csv(data / "test.csv", ["StudyInstanceUID", "Report"], [[f"9.9.{i}", _report(rng, 100 + i)] for i in range(10)])
    _csv(data / "sample_submission.csv", ["StudyInstanceUID", *LABELS], [[f"9.9.{i}", "0.5", "0.5", "0.5"]
                                                                          for i in range(10)])
    (root / ".gitignore").write_text("data/train.csv\ndata/test.csv\ndata/sample_submission.csv\n"
                                     ".kaggle-gm/audit.json\n")
    (root / "src").mkdir()
    (root / "src" / "train.py").write_text("print('train')\n")
    commit(root, "init")
    monkeypatch.chdir(root)
    return root, rows


def _gold(root, rows, path="data/gold.csv"):
    _csv(root / path, ["StudyInstanceUID", "Report", *LABELS], [r for r in rows if r[2]])


def _finding(res, path):
    return next(x for x in res["findings"] if x["path"] == path)


def test_a_derived_subset_deleted_later_is_still_a_leak(comp, capsys):
    root, rows = comp
    _gold(root, rows)
    commit(root, "gold")
    git(root, "rm", "-q", "data/gold.csv")
    commit(root, "remove gold")
    res = AU.run(root, online=False)
    leak = _finding(res, "data/gold.csv")
    assert leak["severity"] == "leak" and leak["where"] == ["history"] and not leak["in_head"]
    why = " ".join(leak["why"])
    assert "20 row(s) of data/train.csv" in why and "passage(s) of competition text" in why
    assert "label column(s) equal to the competition's" in why and "ACL, MCL, Effusion" in why
    assert res["leaks"] == ["data/gold.csv"] and res["unignored"] == []
    with pytest.raises(SystemExit) as e:
        cli.main(["audit-repo", "--offline"])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "LEAK: competition data in git (1)" in out and "not in HEAD (deleted" in out
    assert "git filter-repo --invert-paths --path data/gold.csv" in out
    assert "compared with 3 competition file(s)" in out


def test_labels_made_from_the_data_are_a_judgment_call(comp):
    root, rows = comp
    rng = random.Random(1)
    _csv(root / "data" / "llm_labels.csv", ["StudyInstanceUID", *LABELS],
         [[r[0], *(f"{rng.random():.2f}" for _ in LABELS)] for r in rows])
    commit(root)
    res = AU.run(root, online=False)
    x = _finding(res, "data/llm_labels.csv")
    assert x["severity"] == "check" and x["in_head"] and res["leaks"] == []
    assert "keyed by the competition's IDs (StudyInstanceUID, 60 rows)" in x["why"][0]


def test_copies_in_other_shapes(comp):
    root, rows = comp
    (root / "backup").mkdir()
    shutil.copy(root / "data" / "sample_submission.csv", root / "backup" / "sub.csv")
    # five training rows saved again: labels as floats, CRLF line endings (git keeps them: autocrlf is off here)
    _csv(root / "backup" / "rows.csv", ["StudyInstanceUID", "Report", *LABELS],
         [[r[0], r[1], *(f"{float(v):.1f}" for v in r[2:])] for r in rows[:5]], newline="\r\n")
    (root / "notes.txt").write_bytes((rows[7][1] + "\n").replace("\n", "\r\n").encode())
    (root / "README.md").write_text(f"Most reports say: {BOILER}\nIDs look like {rows[0][0]}\n")
    commit(root)
    found = {x["path"]: x for x in AU.run(root, online=False)["findings"]}
    assert found["backup/sub.csv"]["why"][0] == "an exact copy of data/sample_submission.csv"
    assert any("5 row(s) of data/train.csv" in w for w in found["backup/rows.csv"]["why"])
    assert found["notes.txt"]["severity"] == "leak"
    assert "README.md" not in found  # a template sentence and an ID are not competition text


def test_report_text_in_an_uncommitted_notebook(comp):
    root, rows = comp
    (root / "notebooks").mkdir()
    nb = {"cells": [{"cell_type": "code", "source": ["df.Report.head(2)"],
                     "outputs": [{"text": rows[30][1].splitlines(True) + rows[31][1].splitlines(True)}]}]}
    (root / "notebooks" / "eda.ipynb").write_text(json.dumps(nb, indent=1))
    res = AU.run(root, online=False)
    x = _finding(res, "notebooks/eda.ipynb")
    assert x["severity"] == "leak" and x["where"] == ["pending"]
    assert "Not committed yet" in "\n".join(AU.report(res))


def test_untracked_data_files_and_fix_gitignore(comp, capsys):
    root, _ = comp
    (root / "data" / "subset.csv").write_text("a,b\n1,2\n")
    assert AU.run(root, online=False)["unignored"] == ["data/subset.csv"]
    with pytest.raises(SystemExit) as e:
        cli.main(["audit-repo", "--offline", "--fix-gitignore"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "added to .gitignore: /data/" in out and "NOT IGNORED" not in out
    assert (root / ".gitignore").read_text().count("/data/") == 1
    assert AU.ensure_ignored(root, ["data"]) == []


def test_clean_repo(comp, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["audit-repo", "--offline"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "No competition data found in git." in out and "remote: none" in out


def test_without_the_competition_files_names_are_checked(comp, capsys):
    root, rows = comp
    _gold(root, rows)
    commit(root)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (root / "data" / name).unlink()
    res = AU.run(root, online=False)
    x = _finding(res, "data/gold.csv")
    assert x["severity"] == "check" and not res["tables"]
    assert "no competition table on this machine to compare its values with" in " ".join(x["why"])
    assert "only names and folders were checked" in "\n".join(AU.report(res))
    # the competition's file, kept outside the repo, can be named instead
    rows_file = root.parent / "train.csv"
    _csv(rows_file, ["StudyInstanceUID", "Report", *LABELS], rows)
    assert _finding(AU.run(root, [str(rows_file)], online=False), "data/gold.csv")["severity"] == "leak"


def test_a_public_remote_gets_every_cleanup_step(comp, monkeypatch, capsys):
    root, rows = comp
    _gold(root, rows)
    commit(root)
    git(root, "remote", "add", "origin", "https://someone:s3cret@github.com/me/knee.git")
    asked = []
    monkeypatch.setattr(AU, "visibility", lambda repo, timeout=20: asked.append(repo) or
                        {"visibility": "PUBLIC", "forks": 2, "via": "gh", "error": None})
    with pytest.raises(SystemExit):
        cli.main(["audit-repo"])
    out = capsys.readouterr().out
    assert asked == ["me/knee"] and "s3cret" not in out
    assert "github.com/me/knee is PUBLIC" in out and "2 fork(s)" in out and "in git since" in out
    for step in ("Make the repository private", "git rm --cached", "git filter-repo", "GitHub Support",
                 "notify Kaggle"):
        assert step in out
    res = AU.run(root, online=False)  # an offline run keeps what the last audit learnt
    assert res["public"] is True and "as of the audit of" in "\n".join(AU.report(res))
    cache = (root / ".kaggle-gm" / "audit.json").read_text()
    assert json.loads(cache)["leaks"] == ["data/gold.csv"] and "s3cret" not in cache and "Report" not in cache


def test_sampled_fingerprints_still_find_a_large_copy(comp, monkeypatch):
    root, rows = comp
    monkeypatch.setattr(AU, "CAPS", {"passages": 40, "rows": 10, "ids": 10, "pairs": 10})
    _csv(root / "all.csv", ["StudyInstanceUID", "Report", *LABELS], rows)
    commit(root)
    assert _finding(AU.run(root, online=False), "all.csv")["severity"] == "leak"


def test_integer_ids_0_to_n(tmp_path):
    """Playground style, IDs 0..N-1: an OOF file that carries the true labels is a leak; a small table with its own
    `id` column (run numbers) is not "keyed by the competition's IDs"."""
    root = tmp_path / "pg"
    root.mkdir()
    git(root, "init", "-q")
    S.CompetitionState(slug="pg", id_col="id", target="target").save(root)
    rng = random.Random(2)
    rows = [[str(i), f"{rng.random():.6f}", f"{rng.random():.6f}", str(rng.randint(0, 1))] for i in range(300)]
    (root / "data").mkdir()
    _csv(root / "data" / "train.csv", ["id", "f1", "f2", "target"], rows)
    (root / ".gitignore").write_text("data/train.csv\n")
    _csv(root / "oof.csv", ["id", "target", "pred"], [[r[0], r[3], f"{rng.random():.3f}"] for r in rows])
    _csv(root / "runs.csv", ["id", "model", "cv"], [[str(i), f"m{i}", f"0.{i:02d}"] for i in range(1, 21)])
    commit(root)
    found = {x["path"]: x for x in AU.run(root, online=False)["findings"]}
    assert found["oof.csv"]["severity"] == "leak" and "): target" in " ".join(found["oof.csv"]["why"])
    assert "runs.csv" not in found


def test_a_csv_rule_does_not_ignore_the_data_folder(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("*.csv\n")  # data/x.parquet or data/x.json would still be committed
    assert AU.ensure_ignored(tmp_path, ["data"]) == ["/data/"]
    assert AU.ensure_ignored(tmp_path, ["data"]) == []


def test_brief_lines(comp):
    root, rows = comp
    assert AU.brief_lines(root) == []  # never audited, no GitHub remote
    git(root, "remote", "add", "origin", "git@github.com:me/knee.git")
    assert "not checked yet" in AU.brief_lines(root)[0] and "me/knee" in AU.brief_lines(root)[0]
    AU.run(root, online=False)
    assert AU.brief_lines(root) == []  # audited, clean
    _gold(root, rows)
    commit(root)
    (root / "data" / "new.csv").write_text("a,b\n1,2\n")
    AU.run(root, online=False)
    warning, unignored = AU.brief_lines(root)
    assert warning.startswith("WARNING: competition data in git") and "data/gold.csv" in warning
    assert "data/new.csv" in unignored and "--fix-gitignore" in unignored
    cache = AU.cache_path(root)
    cache.write_text(json.dumps({"at": "2020-01-01T00:00:00Z", "public": True, "leaks": [], "unignored": []}))
    assert "last competition-data audit was 2020-01-01" in AU.brief_lines(root)[0]
    cache.write_text("{not json")
    assert "not checked yet" in AU.brief_lines(root)[0]


def test_init_adds_data_to_an_existing_gitignore(tmp_path, capsys):
    gi = tmp_path / ".gitignore"
    gi.write_text("data/train.csv")  # names the download only, no final newline
    cli.main(["init", "knee", "--dir", str(tmp_path)])
    assert "added /data/ to .gitignore" in capsys.readouterr().out
    assert gi.read_text().splitlines()[0] == "data/train.csv" and gi.read_text().endswith("/data/\n")
    cli.main(["init", "knee", "--dir", str(tmp_path), "--force"])
    assert "added" not in capsys.readouterr().out and gi.read_text().count("/data/") == 1


def test_adopt_runs_the_audit(comp, capsys):
    root, _ = comp
    cli.main(["adopt", "knee", "--dir", str(root), "--no-submissions"])
    assert "Competition data in git: audit of" in capsys.readouterr().out
    cli.main(["adopt", "knee", "--dir", str(root), "--no-submissions", "--no-audit"])
    assert "audit of" not in capsys.readouterr().out


def test_not_a_git_repository(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["audit-repo"])
    assert e.value.code == 0 and "not a git repository" in capsys.readouterr().out


def test_visibility_asks_nobody_offline(monkeypatch):
    monkeypatch.setenv("KGKIT_OFFLINE", "1")
    monkeypatch.setattr(AU.shutil, "which", lambda *a: pytest.fail("gh was looked up"))
    assert AU.visibility("me/knee")["visibility"] == "UNKNOWN"


@pytest.mark.parametrize("url,repo", [
    ("https://github.com/me/knee.git", "me/knee"), ("git@github.com:me/knee.v2.git", "me/knee.v2"),
    ("ssh://git@github.com/me/knee", "me/knee"), ("https://u:tok@github.com/me/knee/", "me/knee"),
    ("https://gitlab.com/me/knee.git", None), ("C:/repos/knee", None)])
def test_github_repo(url, repo):
    assert AU.github_repo(url) == repo


def test_passages_compare_words_not_ids():
    assert AU.passages("1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149") == []
    s = "The anterior cruciate ligament is intact without any sign of a tear today. Short one."
    assert AU.passages(s) == ["the anterior cruciate ligament is intact without any sign of a tear today"]
    assert AU.passages(s.upper().replace(" ", "  ")) == AU.passages(json.dumps(s)[1:-1]) == AU.passages(s)
    assert AU.safe_url("https://u:tok@github.com/me/knee") == "https://github.com/me/knee"
