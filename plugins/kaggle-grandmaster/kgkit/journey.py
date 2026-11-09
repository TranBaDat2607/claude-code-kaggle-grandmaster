"""The journey of a competition, written from the ledger: ``kgkit ledger journey``.

A finished competition's repository has two ways in: ``solution/`` (the final pipeline, for readers who want the
result) and ``journey/`` (how the ideas developed, for readers who want to learn from it). ``journey/README.md`` is
the story, written with /kg-showcase from its template. This module writes the other half, from the records alone:
``journey/experiments.md``, every record grouped by phase (``kgkit ledger phase``) with its parent, the change it
tested, its scores, the decision and a link to the code it ran from, and the lineage of the decided records as a
Mermaid graph (GitHub renders it).

A code link opens the repository at ``git_snapshot`` (the code as it ran, :mod:`kgkit.snapshot`) or at
``git_commit``; a record logged from uncommitted changes before snapshots existed links its commit, marked
approximate. Standard library only.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from . import audit as AU
from . import evalset as E
from . import snapshot as SN
from . import state as S

MAX_GRAPH_NODES = 60
NOTE_CHARS = 110
DECISION_NOTE_CHARS = 70
NO_PHASE = "(no phase)"
SHOWN_TAGS = ("lb-only", "lb-blend", "public-lb-fitted", "public-tuned", "teammate", "notebook", "external")
CLASSES = {"baseline": "fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3a", "keep": "fill:#dcfce7,stroke:#15803d,color:#052e16",
           "discard": "fill:#fee2e2,stroke:#b91c1c,color:#450a0a",
           "inconclusive": "fill:#fef9c3,stroke:#a16207,color:#422006", "none": "fill:#f3f4f6,stroke:#6b7280,color:#111827"}


def repo_web_url(root: str | Path) -> tuple[str | None, str | None, str]:
    """(``https://github.com/<owner>/<repo>``, the remote's name, the workspace's folder inside the repository) from
    the GitHub remote (``origin`` first); (None, None, sub) without one."""
    top = AU.toplevel(root)
    if top is None:
        return None, None, ""
    try:
        sub = Path(root).resolve().relative_to(top.resolve()).as_posix()
    except ValueError:
        sub = ""
    sub = "" if sub == "." else sub
    rs = sorted((r for r in AU.remotes(top) if r.get("repo")), key=lambda r: r["name"] != "origin")
    if not rs:
        return None, None, sub
    return f"https://github.com/{rs[0]['repo']}", rs[0]["name"], sub


def _cell(text) -> str:
    return " ".join(str(text or "").replace("|", "\\|").split())


def _cut(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _num(rec: dict) -> str:
    return rec["id"].split("_")[0]


def _score(v) -> str:
    return f"{v:.5f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else ""


def code_cell(rec: dict, parent: dict | None, url: str | None, sub: str = "") -> str:
    """The record's code link (≈ when approximate) and a diff link against its parent's code."""
    ref, exact = SN.code_ref(rec)
    if not ref:
        return ""
    mark = "" if exact else "≈"
    if not url:
        return f"`{mark}{ref[:7]}`"
    out = f"[{mark}{ref[:7]}]({url}/tree/{ref}{'/' + sub if sub else ''})"
    pref = SN.code_ref(parent)[0] if parent else None
    if pref and pref != ref:
        out += f" · [diff]({url}/compare/{pref}...{ref})"
    return out


def _finals(recs: list[dict], st) -> set[str]:
    picks = [str(p) for p in (getattr(st, "final_picks", None) or [])]
    out = set()
    for p in picks:
        for r in recs:
            if r["id"] == p or (p.split("_")[0].isdigit() and _num(r).isdigit() and int(_num(r)) == int(p.split("_")[0])):
                out.add(r["id"])
    return out


def _lineage(by_id: dict, rid: str) -> list[str]:
    out, seen = [], set()
    while rid in by_id and rid not in seen:
        out.append(rid)
        seen.add(rid)
        rid = by_id[rid].get("parent") or ""
    return out


def graph(recs: list[dict], finals: set[str], holdout: bool = False) -> tuple[list[str], int, int]:
    """(Mermaid lines, nodes drawn, decided records left out) for the decided records, their parents and the finals'
    lineage; at most MAX_GRAPH_NODES, the finals' lineage and the accepted records first."""
    by_id = {r["id"]: r for r in recs}
    final_line = [i for f in sorted(finals) for i in _lineage(by_id, f)]
    decided = [r["id"] for r in recs if r.get("decision")]
    parents = [by_id[i]["parent"] for i in decided if by_id[i].get("parent") in by_id]
    accepted = [i for i in decided if by_id[i]["decision"] in ("baseline", "keep")]
    others = [i for i in reversed(decided) if i not in accepted]
    order = list(dict.fromkeys([*final_line, *accepted, *parents, *others]))
    if not order:
        return [], 0, 0
    nodes = set(order[:MAX_GRAPH_NODES])
    dropped = sum(1 for i in decided if i not in nodes)
    lines = ["```mermaid", "flowchart LR"]
    key = "holdout" if holdout else "CV"
    for r in recs:
        if r["id"] not in nodes:
            continue
        rest = r["id"].split("_", 1)[1] if "_" in r["id"] else r.get("name", "")
        score = (f"{key} {_score(r['cv'])}" if _score(r.get("cv")) else
                 f"public {_score(r['lb_public'])}" if _score(r.get("lb_public")) else "no score")
        label = f"{'FINAL ' if r['id'] in finals else ''}{_num(r)} {rest}<br/>{score}".replace('"', "#quot;")
        lines.append(f'  n{_num(r)}["{label}"]:::{r.get("decision") or "none"}')
    for r in recs:
        if r["id"] in nodes and r.get("parent") in nodes:
            lines.append(f"  n{_num(by_id[r['parent']])} --> n{_num(r)}")
    lines += [f"  classDef {k} {v}" for k, v in CLASSES.items()]
    lines.append("```")
    return lines, len(nodes), dropped


def _phases(recs: list[dict]) -> list[tuple[str, list[dict]]]:
    out: dict[str, list[dict]] = {}
    for r in recs:
        out.setdefault(r.get("phase") or NO_PHASE, []).append(r)
    return list(out.items())


def _note(r: dict) -> str:
    text = r.get("notes") or r.get("name") or ""
    tags = [t for t in (r.get("tags") or ()) if t in SHOWN_TAGS]
    if r.get("recheck"):
        tags.append("recheck")
    return _cell(_cut(text, NOTE_CHARS)) + (f" _[{', '.join(tags)}]_" if tags else "")


def _decision(r: dict) -> str:
    d = r.get("decision") or ""
    note = r.get("decision_note") or ""
    return _cell(f"{d}: {_cut(note, DECISION_NOTE_CHARS)}" if d and note else d)


def render(recs: list[dict], st, url: str | None = None, sub: str = "", today: str | None = None) -> tuple[str, dict]:
    """(the markdown of journey/experiments.md, counts for the summary)."""
    today = today or dt.date.today().isoformat()
    holdout = bool(getattr(st, "holdout", False))
    key = "holdout" if holdout else "CV"
    by_id = {r["id"]: r for r in recs}
    finals = _finals(recs, st)
    phases = _phases(recs)
    refs = [SN.code_ref(r) for r in recs]
    stats = {"records": len(recs), "phases": len([p for p, _ in phases if p != NO_PHASE]),
             "no_phase": sum(1 for r in recs if not r.get("phase")),
             "decided": sum(1 for r in recs if r.get("decision")),
             "snapshots": sum(1 for r in recs if r.get("git_snapshot")),
             "exact": sum(1 for ref, ok in refs if ref and ok), "approx": sum(1 for ref, ok in refs if ref and not ok),
             "no_code": sum(1 for ref, _ in refs if not ref), "finals": sorted(finals)}
    metric = getattr(st, "metric", "") or (recs[0].get("metric") if recs else "") or "?"
    better = "higher" if getattr(st, "greater_is_better", True) else "lower"
    out = ["# Every experiment", "",
           f"Written by `kgkit ledger journey` on {today} from `.kaggle-gm/ledger.jsonl`: {len(recs)} records in "
           f"{len(phases)} phase(s), {stats['decided']} with a decision. The story is in [README.md](README.md).", "",
           f"Metric: {metric} ({better} is better). {key} is the record's own validation score; Δ is its difference "
           "from the parent (the baseline it was compared against)."
           + (f" Finals: {', '.join(sorted(finals))}." if finals else ""), "",
           "**code** opens the repository at the code the record ran from: the commit, or a snapshot commit of the "
           "uncommitted changes that ran (`refs/kg/<id>`). ≈ marks a record logged from uncommitted changes before "
           "snapshots existed: its link shows the last commit before them. **diff** compares the code with the "
           "parent's.", ""]
    lines, n_nodes, dropped = graph(recs, finals, holdout)
    stats["graph_nodes"] = n_nodes
    out += ["## Lineage", ""]
    if lines:
        out += ["Decided records and the baselines they were compared against: blue baseline, green keep, red "
                "discard, yellow inconclusive, grey undecided parent."
                + (f" {dropped} more decided records are in the tables below." if dropped else ""), "", *lines, ""]
    else:
        out += ["No record has a decision yet (`kgkit ledger decide`), so there is no lineage to draw.", ""]
    out += ["## Phases", "", f"| # | phase | records | dates | best {key} | keep | discard |", "|---:|---|---:|---|---:|---:|---:|"]
    for i, (name, rs) in enumerate(phases, 1):
        own = [r for r in rs if E.own_score(r, holdout) and not r.get("recheck")]
        best = sorted(own, key=lambda r: r["cv"], reverse=getattr(st, "greater_is_better", True))[:1]
        when = f"{str(rs[0].get('time') or '')[:10]} – {str(rs[-1].get('time') or '')[:10]}"
        out.append(f"| {i} | [{_cell(name)}](#{i}-{_anchor(name)}) | {len(rs)} | {when} | "
                   f"{_score(best[0]['cv']) + ' (' + _num(best[0]) + ')' if best else ''} | "
                   f"{sum(1 for r in rs if r.get('decision') in ('keep', 'baseline'))} | "
                   f"{sum(1 for r in rs if r.get('decision') == 'discard')} |")
    out.append("")
    for i, (name, rs) in enumerate(phases, 1):
        out += [f"## {i}. {name}", "",
                f"{len(rs)} record(s), {str(rs[0].get('time') or '')[:10]} – {str(rs[-1].get('time') or '')[:10]}.", "",
                f"| id | change | parent | {key} | Δ | public | private | decision | code |",
                "|---|---|---|---:|---:|---:|---:|---|---|"]
        for r in rs:
            parent = by_id.get(r.get("parent") or "")
            delta = (E.fmt_gain(r["cv"] - parent["cv"]) if parent and _score(r.get("cv")) and _score(parent.get("cv"))
                     else "")
            rid = f"**{r['id']}** (final)" if r["id"] in finals else r["id"]
            out.append(f"| {rid} | {_note(r)} | {_num(parent) if parent else ''} | {_score(r.get('cv'))} | {delta} | "
                       f"{_score(r.get('lb_public'))} | {_score(r.get('lb_private'))} | {_decision(r)} | "
                       f"{code_cell(r, parent, url, sub)} |")
        cmds = [(r["id"], r["cmd"]) for r in rs if r.get("cmd")]
        if cmds:
            out += ["", f"<details><summary>Commands ({len(cmds)})</summary>", "", "```",
                    *[f"{rid}: {cmd}" for rid, cmd in cmds], "```", "", "</details>"]
        out.append("")
    return "\n".join(out).rstrip() + "\n", stats


def _anchor(name: str) -> str:
    """GitHub's heading anchor for the text after "<n>. "."""
    keep = "".join(ch for ch in name.lower() if ch.isalnum() or ch in " -_")
    return keep.replace(" ", "-")


def main(led, a) -> None:
    """`kgkit ledger journey`: write the file and say what is missing for the links and the story."""
    st = S.load(led.root)
    recs = led.records()
    if not recs:
        raise SystemExit("the ledger is empty: there is no journey to write")
    url, remote, sub = repo_web_url(led.root)
    if a.repo_url:
        url = a.repo_url.rstrip("/")
    text, stats = render(recs, st, url, sub)
    out = Path(a.out)
    if not out.is_absolute():
        out = led.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    try:
        shown = out.relative_to(led.root).as_posix()
    except ValueError:
        shown = out.as_posix()
    print(f"wrote {shown}: {stats['records']} records in {stats['phases']} phase(s), {stats['decided']} decided, "
          f"lineage graph of {stats['graph_nodes']} record(s)")
    print(f"code links: {stats['exact']} exact ({stats['snapshots']} of them snapshots of uncommitted changes), "
          f"{stats['approx']} approximate (logged from uncommitted changes before snapshots existed), "
          f"{stats['no_code']} without a commit")
    if not url:
        print("no GitHub remote: the code column shows commit ids. `--repo-url https://github.com/<owner>/<repo>` "
              "links them")
    elif stats["snapshots"]:
        print(f"the snapshot links resolve once the snapshots are on GitHub: run `kgkit audit-repo` (it reads every "
              f"ref), then, with the user's go-ahead, `git push {remote or 'origin'} 'refs/kg/*:refs/kg/*'`")
    if stats["no_phase"]:
        print(f"{stats['no_phase']} record(s) have no phase: `kgkit ledger phase \"<name>\" --records <a>..<b>` puts "
              "them in one")
    undecided = sum(1 for r in recs if not r.get("decision"))
    if undecided:
        print(f"{undecided} record(s) have no decision: they are not in the lineage graph "
              "(`kgkit ledger decide <ids> keep|discard|inconclusive \"<why>\"`)")
