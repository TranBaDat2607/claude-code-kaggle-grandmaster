---
name: kg-final
description: Choose the final Kaggle submissions to maximise expected private leaderboard score — evidence table, LB noise estimate, CV-vs-LB weighting, hedge selection and robustness checks.
allowed-tools: [Bash, PowerShell, Read, Write, Glob, Grep, Agent]
---

# /kg-final

Follow the `final-submission-selection` skill.

1. Evidence table from `python -m kgkit ledger best -n 20` + all submissions with LB scores:
   id, CV ± std, honest blend CV, public LB, risk elements, correlation with the top candidate.
2. Estimate public-LB noise (bootstrap the metric on OOF at the public-set size; see
   `cv-lb-debugging`) and the CV-LB correlation.
3. Recommend pick 1 (best honest CV) and pick 2 (credible hedge on a different risk axis), with a
   short argument for each and what scenario each protects against.
4. Robustness checks: reproduce the chosen files from code (or confirm the kernel versions ran on
   the hidden test with runtime margin); validate files.
5. Remind the user to **select them on the website** before the deadline (otherwise Kaggle
   auto-selects by public LB) and note the exact deadline in UTC and local time.
Write the decision and reasoning to `reports/final-selection.md`.
