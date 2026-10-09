# kaggle-grandmaster

Claude Code plugin for competing at Kaggle Grandmaster level. See the repository README for
full documentation.

Start a competition: `/kaggle-grandmaster:kg-start <competition-slug>`

- `skills/` — 20 knowledge playbooks + 16 `kg-*` slash commands
- `agents/` — 9 specialist subagents
- `hooks/` — session brief, prompt→skill router, submission + GPU-quota guard, submission / GPU-run logger
- `kgkit/` — Python toolkit (`python -m kgkit --help`), incl. `kgkit gpu` for quota-aware training on Kaggle GPUs
- `templates/` — GBDT / image / transformer training, remote GPU training runner, offline inference kernel, workspace CLAUDE.md
- `evals/` — `claude plugin eval` suite
