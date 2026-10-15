"""scripts/bump_version.py: the versions stay in sync and a release bump rewrites every copy."""

import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bump_version", REPO / "scripts" / "bump_version.py")
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)


def _copy_repo(tmp_path: Path) -> Path:
    for f in bv.files(REPO).values():
        dst = tmp_path / f.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    return tmp_path


def test_versions_in_sync():
    assert bv.main(["--check"]) == 0
    v = bv.versions(REPO)
    assert len(set(v.values())) == 1, v


def test_next_version():
    assert bv.next_version("1.2.9", "patch") == "1.2.10"
    assert bv.next_version("1.2.9", "minor") == "1.3.0"
    assert bv.next_version("1.2.9", "major") == "2.0.0"


def test_bump_rewrites_all_and_renames_unreleased(tmp_path):
    root = _copy_repo(tmp_path)
    old = bv.current(root)
    log = bv.files(root)["changelog"]
    log.write_text("# Changelog\n\n## Unreleased\n\n- thing\n\n## 0.1.0 — 2020-01-01\n\n- old\n", encoding="utf-8")
    new = bv.bump(root, "minor", note="ignored (#1)", today="2030-05-06")
    assert new == bv.next_version(old, "minor")
    assert set(bv.versions(root).values()) == {new}
    text = log.read_text(encoding="utf-8")
    assert f"## {new} — 2030-05-06\n\n- thing" in text and "Unreleased" not in text and "ignored" not in text
    assert bv.release_notes(root, new) == "- thing"


def test_bump_without_unreleased_inserts_note(tmp_path):
    root = _copy_repo(tmp_path)
    log = bv.files(root)["changelog"]
    log.write_text("# Changelog\n\n## 0.1.0 — 2020-01-01\n\n- old\n", encoding="utf-8")
    new = bv.bump(root, "patch", note="Fix X (#12)", today="2030-05-06")
    text = log.read_text(encoding="utf-8")
    assert text.index(f"## {new}") < text.index("## 0.1.0") and "- Fix X (#12)" in text
    # the marketplace's other fields are untouched (formatting kept, only the version line changed)
    a = (REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8").splitlines()
    b = bv.files(root)["marketplace.json"].read_text(encoding="utf-8").splitlines()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
