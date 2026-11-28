"""Measure the prompt router (hooks/prompt_router.py): what it routes, and what it adds over skill descriptions alone.

    python scripts/router_ab.py routes                       # offline, free: the router's pick for every eval prompt
    python scripts/router_ab.py copy <dest>                  # the plugin without the router, for an eval "off" arm
    python scripts/router_ab.py compare <on.json> <off.json> # two `claude plugin eval --json` runs, case by case

``routes`` runs the hook on each case's prompt outside a workspace and checks that it names at least one skill of
the case's ``skill-fired`` grader, and that it stays silent on ``negative`` cases (``ws-*`` cases are routed inside
their workspace in a real run, so they are listed but not scored here). It exits 1 on a miss.

The A/B, from ``plugins/kaggle-grandmaster`` (``--ablation none``: the no-plugin arm is not needed here)::

    python ../../scripts/router_ab.py copy /tmp/no-router
    claude plugin eval . --tag portable --ablation none --model haiku --judge-model sonnet --no-publish --json on.json
    claude plugin eval /tmp/no-router/kaggle-grandmaster --tag portable --ablation none --model haiku \\
        --judge-model sonnet --no-publish --trust-plugin --json off.json
    python ../../scripts/router_ab.py compare on.json off.json

``compare`` prints, per case, how often the expected skill loaded (``skill-fired``) and the mean result score (the
other graders) in each arm. Standard library only; works on Windows.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent / "plugins" / "kaggle-grandmaster"
PROCESS = {"skill-fired", "hook-fired"}


def _case_prompt(case: Path) -> tuple[list[str], str]:
    text = (case / "prompt.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    front, body = (m.group(1), m.group(2)) if m else ("", text)
    tags = re.search(r"tags:\s*\[(.*?)\]", front)
    return ([t.strip() for t in tags.group(1).split(",")] if tags else []), body.strip()


def expected_skills(case: Path) -> list[str]:
    """Skill names the case's skill-fired grader accepts: the alternatives of its last ``(?:a|b)`` group."""
    f = case / "graders" / "skill-fired.md"
    if not f.is_file():
        return []
    m = re.search(r"input_match:\s*'(.*)'", f.read_text(encoding="utf-8"))
    groups = re.findall(r"\(\?:([a-z0-9|-]+)\)", m.group(1)) if m else []
    return groups[-1].split("|") if groups else []


def route(plugin: Path, prompt: str, cwd: str) -> list[str]:
    payload = {"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": cwd,
               "session_id": "ab" + uuid.uuid4().hex[:12]}
    r = subprocess.run([sys.executable, str(plugin / "hooks" / "prompt_router.py")], input=json.dumps(payload),
                       capture_output=True, text=True, cwd=cwd, timeout=30)
    if not r.stdout.strip():
        return []
    ctx = json.loads(r.stdout)["hookSpecificOutput"].get("additionalContext", "")
    if "follow their checklists:" not in ctx:
        return ["(text only)"]
    return re.findall(r"kaggle-grandmaster:([a-z0-9-]+)", ctx.split("follow their checklists:")[-1].split(".")[0])


def cmd_routes(plugin: Path) -> int:
    misses = 0
    with tempfile.TemporaryDirectory() as cwd:
        for case in sorted(p for p in (plugin / "evals").iterdir() if (p / "prompt.md").is_file()):
            tags, prompt = _case_prompt(case)
            got, want = route(plugin, prompt, cwd), expected_skills(case)
            if "negative" in tags:
                verdict = "ok" if not got else "MISS (routes a negative case)"
            elif case.name.startswith("ws-"):
                verdict = "-  (routed inside its workspace)"
            elif not want:
                verdict = "-"
            else:
                verdict = "ok" if set(got) & set(want) else "MISS"
            misses += verdict.startswith("MISS")
            print(f"{case.name:32} {verdict:34} router={','.join(got) or '-'}  expected={','.join(want) or '-'}")
    print(f"\n{misses} miss(es)")
    return 1 if misses else 0


def cmd_copy(plugin: Path, dest: Path) -> int:
    out = dest.resolve() / plugin.name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(plugin, out, ignore=shutil.ignore_patterns("results", "__pycache__", "node_modules"))
    hooks_json = out / "hooks" / "hooks.json"
    cfg = json.loads(hooks_json.read_text(encoding="utf-8"))
    kept = []
    for group in cfg["hooks"].get("UserPromptSubmit", []):
        group["hooks"] = [h for h in group["hooks"] if "prompt_router.py" not in h.get("command", "")]
        if group["hooks"]:
            kept.append(group)
    if kept:
        cfg["hooks"]["UserPromptSubmit"] = kept
    else:
        cfg["hooks"].pop("UserPromptSubmit", None)
    hooks_json.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


def _arm(path: Path) -> dict:
    out = {}
    for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        fired, res = [], []
        for run in case["arms"]["with"]:
            graders = run.get("graders") or []
            fired += [g["passed"] for g in graders if g["name"] in PROCESS][:1]
            rest = [g for g in graders if g["name"] not in PROCESS]
            weight = sum(g.get("weight", 1) for g in rest)
            if weight:
                res.append(sum(g.get("weight", 1) for g in rest if g["passed"]) / weight)
        out[case["name"]] = (fired, res)
    return out


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


def cmd_compare(on_path: Path, off_path: Path) -> int:
    on, off = _arm(on_path), _arm(off_path)
    f = lambda x: "   -" if x is None else f"{x:4.2f}"  # noqa: E731
    print(f"{'case':32} {'fired on':>8} {'off':>5}   {'result on':>9} {'off':>5}")
    pairs = []
    for name in sorted(set(on) & set(off)):
        (fo, ro), (ff, rf) = on[name], off[name]
        pairs.append((_mean(fo), _mean(ff), _mean(ro), _mean(rf)))
        print(f"{name:32} {f(pairs[-1][0]):>8} {f(pairs[-1][1]):>5}   {f(pairs[-1][2]):>9} {f(pairs[-1][3]):>5}")
    for label, i, j in (("skill fired", 0, 1), ("result", 2, 3)):
        both = [(p[i], p[j]) for p in pairs if p[i] is not None and p[j] is not None]
        if both:
            print(f"mean {label}: on {_mean([a for a, _ in both]):.3f}, off {_mean([b for _, b in both]):.3f} "
                  f"({len(both)} cases)")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["routes"]:
        return cmd_routes(PLUGIN)
    if argv[:1] == ["copy"] and len(argv) == 2:
        return cmd_copy(PLUGIN, Path(argv[1]))
    if argv[:1] == ["compare"] and len(argv) == 3:
        return cmd_compare(Path(argv[1]), Path(argv[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
