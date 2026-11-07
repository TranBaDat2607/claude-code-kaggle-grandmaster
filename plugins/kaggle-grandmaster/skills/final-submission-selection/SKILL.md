---
name: final-submission-selection
description: Choosing the final Kaggle submissions to maximise expected private leaderboard score and survive shake-ups — CV vs public LB weighting, hedging strategies, robustness checks, and endgame timeline. Use in the last days of a competition or when the user asks which submissions to select.
---

# Final Submission Selection

You usually get **2 final picks** (check the rules; some comps allow more or auto-select). The goal
is the best *private* score, which is a different sample (and sometimes a different time period)
than the public LB.

## 1. Gather evidence

`python -m kgkit ledger best -n 20` and `python -m kgkit status`. For each candidate: CV mean/std,
public LB, CV-LB agreement, number of models, use of risky elements (leaks, PL, LB-tuned weights,
aggressive post-processing), and its correlation with other candidates.

## 2. How much to trust public LB

Weight public LB by its statistical reliability:
- Public set small (a few thousand rows or fewer), noisy metric, or adversarial AUC high → trust CV.
- Public set large and drawn like private, CV-LB correlation strong → LB and CV mostly agree; use both.
- Private is a *future* period → favour robustness (simpler, ensemble-heavy, less feature drift).
Formalise per pair of candidates: `python -m kgkit ledger compare <a> <b> --public-n <rows>` gives the
paired SE of their gap at the public-set size (both are scored on the same public rows, so this is far
smaller than the SE of one score); treat LB gaps below ~2 paired SE as ties broken by CV. The SE of one
absolute score (`kgkit ledger noise <id> --public-n <rows>`) is for judging one candidate's CV-LB gap,
not the gap between two candidates. See `cv-lb-debugging`. A pair that includes an LB-only candidate has no
OOF: the same command then reads the two submission files (`ledger lb <id> <score> --sub FILE` attaches one) and
prints their correlation plus an *estimated* gap SD from simulated labels. Report it as an estimate, never as a
measured SE; the rule of thumb "public SD = 0.003 x SD of the rank difference" depends on the metric's scale and
measured about 3x too small on a binary AUC task.

## 3. Default selection policy

- **Pick 1 — best CV**: the strongest honest-CV ensemble (nested-validated blend).
- **Pick 2 — hedge**: different from pick 1 in its risk profile: e.g. best public LB if it is
  sufficiently different and credible; or a version with/without the risky component (PL,
  leak, post-processing); or the full-data retrained version.
- Avoid two near-identical submissions (correlation > 0.995) — the second pick buys nothing.
- LB-only candidates (blends with public submission files, forks with no OOF; `LB-only` in the ledger)
  have no honest CV. They can be the hedge, never both picks: keep a CV-backed candidate as pick 1.
  Their weights were chosen on the public LB, so expect them to lose some of their public edge on private.
- **Public-tuned** candidates (tag `public-tuned`: weights, members or a threshold picked by comparing public
  scores, as in LB probing) are not a hedge. Their public edge is selection, and on a large-test synthetic
  competition it is noise at best. When one is proposed as a final, show the CV-only alternative next to it
  (`kgkit status` names it). Hedge instead with a file that differs in its risk profile and whose choices were
  fixed before it was scored.
- Files fitted to the public labels (tilted toward an estimate of them, built from publicly shared scored files;
  tag them `public-lb-fitted`, `kgkit ledger tag <id> public-lb-fitted`) are display entries: not pick 1, and
  unlike an LB-only blend not even the hedge. Their public edge is mostly fit (reported for a Playground
  S6E9 2nd place: 0.94945 public, 0.94313 private). `kgkit` keeps them out of best CV, the baseline and the
  CV-LB correlation.
- If both CV and LB agree on a candidate, take it, and hedge with the second-best on a
  different axis.

## 4. Robustness checks before locking

- Re-run the final pipeline from a clean checkout (or the exact kernel version) and diff the
  predictions against the selected submission file.
- Code competitions: confirm the selected *notebook versions* succeeded on the hidden test and
  that runtime has margin (private rerun may be on more data).
- Check prediction distributions vs OOF, no NaNs, correct id coverage.
- Make sure the selection is actually set on the website (the user must tick the submissions;
  otherwise Kaggle auto-selects the best public scores, which are often the fitted files). Kaggle has no API
  to select, but `python -m kgkit final` reads the selection back. In the last 3 days `kgkit status` and the
  session brief warn until it has confirmed the selection; earlier, only when a fitted file would be auto-selected.

## 5. Endgame timeline (back-planned from the deadline)

| T-minus | Action |
|---|---|
| 2 weeks | stop adding new model families; start scaling + seeds; finalise inference budget |
| 1 week | ensemble design frozen; full-data retrains; kernels tested on hidden-size data |
| 3 days | final ensemble candidates submitted; robustness checks |
| 1 day | buffer for failures (Kaggle queue gets slow near deadlines); select finals |
| deadline | 23:59 UTC — never submit in the last hour if avoidable |

Team merger deadlines usually fall ~1 week before the end; team up early if you plan to. Merging has rules
(no private sharing before the merge, combined submission counts) and a technical protocol (shared folds,
OOF exchange via `kgkit ledger import`): see `competition-strategy`.

## 6. Post-competition

When the private LB is revealed: `kgkit final` says how the finals did (or, with none selected, which pool
Kaggle picked from) and where the team ranks; `kgkit ledger sync-lb` records every private score in the
ledger; `kgkit ledger predictiveness` says which signal predicted private, over all submissions and among
the strongest (a CV-LB correlation over the whole competition is inflated by early, large gains). Read the
top solution write-ups and write `reports/postmortem.md`: this compounds into the next competition
(`/kg-postmortem`).
