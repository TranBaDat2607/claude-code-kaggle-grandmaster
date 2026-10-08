# kaggle-grandmaster

Claude Code plugin for competing at Kaggle Grandmaster level. See the repository README for
full documentation.

Start a competition: `/kaggle-grandmaster:kg-start <competition-slug>`

- `skills/` — 19 knowledge playbooks + 15 `kg-*` slash commands
- `agents/` — 9 specialist subagents
- `hooks/` — session brief, prompt→skill router, submission guard, submission logger
- `kgkit/` — Python toolkit (`python -m kgkit --help`)
- `templates/` — GBDT / image / transformer training, offline inference kernel, workspace CLAUDE.md
- `evals/` — `claude plugin eval` suite
