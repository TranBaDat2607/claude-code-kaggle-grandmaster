"""scripts/router_ab.py: the router names an expected skill for every eval prompt, stays silent on the negative
cases, and the A/B copy has every hook but the router."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("router_ab", ROOT / "scripts" / "router_ab.py")
router_ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router_ab)


def test_router_routes_every_eval_prompt(capsys):
    """A route change that drops the skill a case expects, or routes a negative case, fails here before an eval run."""
    code = router_ab.cmd_routes(router_ab.PLUGIN)
    out = capsys.readouterr().out
    assert code == 0, "\n".join(line for line in out.splitlines() if "MISS" in line)


def test_expected_skills_parse_the_grader():
    case = router_ab.PLUGIN / "evals" / "cv-lb-gap"
    assert router_ab.expected_skills(case) == ["cv-lb-debugging", "kg-debug", "time-series", "validation-strategy"]


def test_copy_drops_only_the_router(tmp_path):
    router_ab.cmd_copy(router_ab.PLUGIN, tmp_path)
    copy = json.loads((tmp_path / "kaggle-grandmaster" / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    real = json.loads((router_ab.PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert "prompt_router" not in json.dumps(copy)
    assert set(copy["hooks"]) == set(real["hooks"]) - {"UserPromptSubmit"}
    assert copy["modules"] == real["modules"]
