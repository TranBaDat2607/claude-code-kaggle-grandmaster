---
name: kg-final
description: Choose the final Kaggle submissions to maximise expected private leaderboard score — evidence table, LB noise estimate, CV-vs-LB weighting, hedge selection and robustness checks.
allowed-tools: [Read, Glob, Grep, Agent]
---

# /kg-final

Follow the `final-submission-selection` skill.

1. Evidence table from `python -m kgkit ledger best -n 20` + all submissions with LB scores:
   id, CV ± std, honest blend CV, public LB, risk elements, correlation with the top candidate.
2. Estimate public-LB noise per pair of candidates (`python -m kgkit ledger compare <a> <b> --public-n <rows>`:
   paired SE of their LB gap; not the SE of one score, see `cv-lb-debugging`) and the CV-LB correlation.
   LB-only records (blends with public files) are left out of that correlation; in a `holdout+lb` workspace
   compare candidates on their stored holdout predictions (`kgkit status` prints the measured paired noise).
3. Recommend pick 1 (best honest CV) and pick 2 (credible hedge on a different risk axis), with a
   short argument for each and what scenario each protects against. An LB-only blend can only be the hedge.
4. Robustness checks: reproduce the chosen files from code (or confirm the kernel versions ran on
   the hidden test with runtime margin); validate files.
5. Remind the user to **select them on the website** before the deadline (otherwise Kaggle
   auto-selects by public LB) and note the exact deadline in UTC and local time.
Write the decision and reasoning to `reports/final-selection.md`.
