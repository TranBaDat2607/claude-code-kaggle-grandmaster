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
Formalise: bootstrap the metric at the public set size (see `cv-lb-debugging`) to get the LB noise
std; treat LB gaps below ~2 std as ties broken by CV.

## 3. Default selection policy

- **Pick 1 — best CV**: the strongest honest-CV ensemble (nested-validated blend).
- **Pick 2 — hedge**: different from pick 1 in its risk profile: e.g. best public LB if it is
  sufficiently different and credible; or a version with/without the risky component (PL,
  leak, post-processing); or the full-data retrained version.
- Avoid two near-identical submissions (correlation > 0.995) — the second pick buys nothing.
- If both CV and LB agree on a candidate, take it, and hedge with the second-best on a
  different axis.

## 4. Robustness checks before locking

- Re-run the final pipeline from a clean checkout (or the exact kernel version) and diff the
  predictions against the selected submission file.
- Code competitions: confirm the selected *notebook versions* succeeded on the hidden test and
  that runtime has margin (private rerun may be on more data).
- Check prediction distributions vs OOF, no NaNs, correct id coverage.
- Make sure the selection is actually set on the website (the user must tick the submissions;
  otherwise Kaggle auto-selects by public LB).

## 5. Endgame timeline (back-planned from the deadline)

| T-minus | Action |
|---|---|
| 2 weeks | stop adding new model families; start scaling + seeds; finalise inference budget |
| 1 week | ensemble design frozen; full-data retrains; kernels tested on hidden-size data |
| 3 days | final ensemble candidates submitted; robustness checks |
| 1 day | buffer for failures (Kaggle queue gets slow near deadlines); select finals |
| deadline | 23:59 UTC — never submit in the last hour if avoidable |

Team merger deadlines usually fall ~1 week before the end; team up early if you plan to.

## 6. Post-competition

When the private LB is revealed: record private scores in the ledger (`ledger lb <id> <public>
<private>`), compute which signals predicted private best (CV? LB?), read the top solution write-ups,
and write `reports/postmortem.md` — this compounds into the next competition (`/kg-postmortem`).
