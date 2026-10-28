---
name: kg-final
description: Choose the final Kaggle submissions to maximise expected private leaderboard score — evidence table, LB noise estimate, CV-vs-LB weighting, hedge selection and robustness checks.
allowed-tools: [Read, Glob, Grep, Agent]
---

# /kg-final

Follow the `final-submission-selection` skill.

1. Evidence table from `python -m kgkit ledger best -n 20` + all submissions with LB scores:
   id, CV ± std, honest blend CV, public LB, risk elements, correlation with the top candidate.
   Records tagged `public-lb-fitted` (files tilted toward an estimate of the public labels; `kgkit status`
   marks them `[public-LB fitted]`) go in their own group, "display only": never pick 1, pick 2 or the hedge.
   A file that is fitted but not tagged yet: `python -m kgkit ledger tag <id> public-lb-fitted`.
2. Estimate public-LB noise per pair of candidates (`python -m kgkit ledger compare <a> <b> --public-n <rows>`:
   paired SE of their LB gap; not the SE of one score, see `cv-lb-debugging`) and the CV-LB correlation.
   LB-only records (blends with public files) are left out of that correlation; in a `holdout+lb` workspace
   compare candidates on their stored holdout predictions (`kgkit status` prints the measured paired noise).
3. Recommend pick 1 (best honest CV) and pick 2 (credible hedge on a different risk axis), with a
   short argument for each and what scenario each protects against. An LB-only blend can only be the hedge;
   a public-LB-fitted file cannot be either.
4. Robustness checks: reproduce the chosen files from code (or confirm the kernel versions ran on
   the hidden test with runtime margin); validate files.
5. Run `python -m kgkit final`: it asks Kaggle which submissions are selected. With none selected, Kaggle
   auto-selects the best public scores, which late in a Playground are often the fitted files: say which ones
   they are. Then tell the user to **select the picks on the website** before the deadline (Kaggle has no API
   for it), with the steps and the deadline in UTC and local time that `kgkit final` prints. Once they have
   selected, run `kgkit final` again: it reads Kaggle's selection back, records it (`final_picks`), and the
   warning in `kgkit status` and the session brief gives way to a one-line note, or to a louder warning if a
   selected file is fitted.
Write the decision and reasoning to `reports/final-selection.md`: the picks (with the file name, description
and date as the Submissions page shows them), the submissions Kaggle would auto-select and why that must not
stand when one is fitted, the website steps and the deadline from `kgkit final`.
