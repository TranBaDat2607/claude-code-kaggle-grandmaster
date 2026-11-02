---
type: llm
---

PASS when all three hold:

1. It advises against `/kg-start` for this repo, because `/kg-start` is a from-scratch bootstrap (data download,
   new folds, new baseline) that would redo or clash with two months of existing work.
2. It recommends adopting the repo in place instead: `/kg-adopt` or `python -m kgkit adopt` (or `kgkit init`)
   with the competition slug, which only writes `.kaggle-gm/`.
3. It brings the existing history into the plugin: importing the past submissions (so the daily budget counts
   today's 3) and/or recording the existing experiments in the kgkit ledger (`ledger add` / `ledger import`).

A plus, not required: passing the code-competition facts (`--code-competition --runtime-hours 9 --no-internet
--daily-subs 5 --deadline 2026-12-01`), marking an accepted baseline, and saying explicitly to keep the existing
validation split.

FAIL if it tells the user to run `/kg-start` on this repo, or to rebuild the folds or the baseline from scratch.
