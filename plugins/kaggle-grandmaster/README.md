# kaggle-grandmaster

Claude Code plugin for competing at Kaggle Grandmaster level. See the repository README for
full documentation.

Install: `/plugin install kaggle-grandmaster --marketplace TranBaDat2607/claude-code-kaggle-grandmaster`

Start a competition: `/kaggle-grandmaster:kg-start <competition-slug>`

- `skills/`: 21 knowledge playbooks (including `competition-strategy`: choosing competitions, the solo
  gold, teaming and merging) + 16 `kg-*` slash commands
- `agents/`: 9 specialist subagents
- `hooks/`: session brief (accepted baseline, backlog top, GPU quota), prompt→skill router, submission +
  GPU-quota guard, submission / GPU-run logger
- `kgkit/`: Python toolkit (`python -m kgkit --help`):
  - experiment loop: `ledger compare` (paired per-fold test + OOF bootstrap), `ledger decide / baseline /
    lineage / import`, `backlog`, `recheck` (does the gain survive a re-drawn fold split?)
  - modelling: `features search` (thousands of generated features, batch-screened), `blend` with hill
    climbing, multi-level and residual stacking, `--prune`
  - prior art: `kernels top / pull` (notebook review + local script), `discussions sync / top / search /
    solutions / read` (forum via Meta Kaggle)
  - `gpu` for quota-aware training on Kaggle GPUs
- `templates/`: tabular (GBDTs + linear / kNN / SVM / MLP), image and transformer training, remote GPU
  training runner, offline inference kernel, workspace CLAUDE.md
- `evals/`: `claude plugin eval` suite (10 cases)
