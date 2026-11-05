---
name: kg-final
description: Choose the final Kaggle submissions to maximise expected private leaderboard score — evidence table, LB noise estimate, CV-vs-LB weighting, hedge selection and robustness checks.
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit final *)'
  - 'Bash(python -m kgkit final *)'
  - 'PowerShell(python -m kgkit final *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
---

# /kg-final

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

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
   A pair with an LB-only record has no OOF to resample, so the same command compares the two submission files
   instead (a file path works in place of an id; log the blend with `ledger add <name> lb-only --sub FILE`, or
   `ledger lb <id> <score> --sub FILE`). It prints their correlation and an *estimated* SD of the public LB gap
   from simulated labels; say it is an estimate, and `--ref <id>` calibrates it on a record that has OOF.
3. Recommend pick 1 (best honest CV) and pick 2 (credible hedge on a different risk axis), with a
   short argument for each and what scenario each protects against. An LB-only blend can only be the hedge;
   a public-LB-fitted file cannot be either.
4. Robustness checks: reproduce the chosen files from code (or confirm the kernel versions ran on
   the hidden test with runtime margin); validate files.
5. Run `python -m kgkit final`: it asks Kaggle which submissions are selected. With none selected, Kaggle
   auto-selects the best public scores, which late in a Playground are often the fitted files: say which ones
   they are. Then tell the user to **select the picks on the website** before the deadline (Kaggle has no API
   for it), with the steps and the deadline in UTC and local time that `kgkit final` prints. Once they have
   selected, run `kgkit final` again: it reads Kaggle's selection back, records it (`final_picks`) and stamps
   when it did, and only then does the warning in `kgkit status` and the session brief give way to a one-line
   note, or to a louder warning if a selected file is fitted or fewer than the allowed finals are selected.
   That warning shows in the last 3 days whenever no selection is confirmed (earlier, only when a fitted file
   would be auto-selected): an unticked selection leaves the pick to Kaggle, among files that often tie.
   `kgkit config final_picks=...` records *intended* picks only: the warning stays, listing them as not
   confirmed on Kaggle, so never use it as a substitute for `kgkit final`.
Write the decision and reasoning to `reports/final-selection.md`: the picks (with the file name, description
and date as the Submissions page shows them), the submissions Kaggle would auto-select and why that must not
stand when one is fitted, the website steps and the deadline from `kgkit final`.
