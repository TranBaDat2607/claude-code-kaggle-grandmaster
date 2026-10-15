---
type: llm
weight: 1
---

The response advises against `/kg-start` for this repo, because it is a from-scratch bootstrap (data download,
new folds, new baseline) that would redo or clash with two months of existing work. It recommends adopting the
repo in place instead: `/kg-adopt` or `python -m kgkit adopt` (or `kgkit init`) with the slug and the
code-competition facts (`--code-competition --runtime-hours 9 --no-internet --daily-subs 5 --deadline 2026-12-01`),
which only writes `.kaggle-gm/`. It mentions importing the existing submission history so the daily budget
counts today's 3 submissions, recording existing experiments in the kgkit ledger (`ledger add` / `ledger import`)
and marking an accepted baseline, and keeping the existing validation split rather than redrawing folds.
Fail if it tells the user to run `/kg-start` on this repo, or to rebuild the folds or baseline from scratch.
