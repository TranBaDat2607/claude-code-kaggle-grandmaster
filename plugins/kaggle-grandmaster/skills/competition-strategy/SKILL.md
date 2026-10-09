---
name: competition-strategy
description: The career layer of becoming a Kaggle Competitions Grandmaster — what the title requires (golds, a solo gold), which competitions award medals and which to enter, planning a solo gold, joining late, running several competitions at once, and teaming — when and with whom to merge, the private-sharing and submission-count rules around merging, and the technical merge protocol (shared folds, OOF exchange, joint final selection). Use when the user asks about teams, merging, choosing competitions, medals or the Grandmaster title.
---

# Competition Strategy (the path to Grandmaster)

Winning one competition is the playbook (`grandmaster-playbook`). Becoming a Grandmaster is a
portfolio problem: enough golds, at least one of them solo, from competitions you chose well.

## 1. What the title requires (verify on Kaggle's progression page — tiers change)

- **Competitions Grandmaster**: 5 gold medals, at least 1 of them **solo**. Master: 1 gold + 2 silver.
- **Gold cut-off scales slowly with field size**: top 10 teams + 0.2% of the field for 250+ teams
  (about the top 11 of 500 teams, the top 20 of 5,000); top 10% below 100 teams. A big field does not
  mean many more golds.
- **Only medal-awarding competitions count**: Featured and Research competitions do. Playground Series,
  Getting Started and most Community competitions do **not** award medals. They are excellent practice
  for the experiment loop and tabular ensembling, not progress toward the title.
- Team medals count for every member, so 4 of the 5 golds can come from teams. The solo gold is the
  bottleneck: plan for it deliberately.

## 2. Choosing competitions

Score each candidate before joining (write it in `reports/recon.md`):

| Factor | Good sign | Bad sign |
|---|---|---|
| Medals | Featured / Research | Playground, Community (unless practice is the goal) |
| Your edge | your modality/domain, a past similar competition, a strong `~/.kaggle-gm/lessons.md` match | unfamiliar modality with a short timeline |
| Validation | a CV you can make mirror the test split (big test set, clear grouping) | tiny or odd public/private split: LB lottery, shake-ups dominate |
| Compute fit | fits your GPUs / the weekly Kaggle quota (`kgkit gpu plan`) | needs multi-GPU weeks you don't have |
| Time | 6-10 weeks of your real availability ahead | overlaps the endgame of another competition |
| Code-comp limits | runtime/offline limits you can engineer for | inference limit that forbids your strengths |

A shake-up-heavy competition is fine for a team medal hedge and wrong for a planned solo gold.

## 3. Planning the solo gold

- Pick a competition with a trustworthy CV and a field where your edge is real; join early (first
  2-3 weeks) so the exploration phase is long.
- Run the full loop yourself: frozen folds, ledger, accepted baseline, backlog, paired comparisons.
  The plugin's agents raise throughput (the March 2026 Playground churn win used LLM agents to run
  850 experiments); validation design and final selection remain your judgement.
- Do not merge in that competition, even late: a team gold does not count as the solo gold.

## 4. Joining late (last 3-5 weeks)

1. Recon in a day: `kgkit kernels top`, then `kgkit kernels pull` the 3-5 best notebooks; read
   `REVIEW.md` for their CV schemes and external inputs; skim discussion threads by votes.
2. Build your folds first. Re-run the best public pipelines on *your* folds and import their OOFs
   (`kgkit ledger import --source notebook`); public LB numbers are not your CV.
3. Spend the time on what public notebooks lack: diversity (a different model family), a better CV,
   and error analysis. Do not fork the most-forked notebook; its LB is saturated and its errors are shared by
   hundreds of teams.
4. Consider a merge before the merger deadline (section 5).

## 5. Teaming

**When.** Early teams split exploration (one owns the data pipeline/CV, others own model families);
late merges (before the merger deadline, usually ~1 week before the end) are mostly an ensembling play.
Merging is not free: each merge adds coordination cost, and repeated merging shows diminishing returns.

**With whom.** Prefer teammates whose models are *different* from yours (GBDT vs NN, CNN vs ViT, other
features or inputs). Their OOF correlation with yours matters more than their LB rank. Judge them from
public signals: LB position over time, discussion posts, past write-ups.

**Rules that bite (check this competition's rules).**
- **No private sharing outside a team.** Exchanging code, data or predictions (including OOF files) with
  someone *not yet on your team* violates the rules and can disqualify both. Merge first, then share.
  Publicly posted code/data (a public notebook or dataset, available to everyone) is not private sharing.
- **Submission-count rule at merge time**: the merged team's combined submissions usually must not exceed
  what a single team could have made by the merge date. Two teams that each used every daily slot may be
  unable to merge.
- Team size limits and the merger deadline (`python -m kgkit discussions sync` prints both, plus
  whether merging is banned, from Kaggle's export). External-data and pretrained-model rules apply to
  every member's pipeline.

**Technical merge protocol (right after the merge).**
1. **One fold split.** Agree on a single `data/folds.csv` (same grouping logic, same row order) and share
   it. The ledger's `folds_hash` shows who trained on what.
2. **Exchange OOF + test predictions** in one format (`.npy`, same row order as train / sample
   submission, probabilities not labels). Each member imports the others' predictions:
   `python -m kgkit ledger import <name> oof.npy --test test.npy --truth data/train.csv:<target>
   --folds data/folds.csv:fold --source teammate`. CV is recomputed on the shared folds, not copied
   from a message.
3. **Mismatched folds**: re-train cheap models on the shared folds. Expensive models that can't be
   re-run go only into simple, low-degree-of-freedom blends (plain or rank average), and their blended
   CV is optimistic (`ensembling` explains the leak mechanism).
4. Blend with honest nested scoring (`kgkit blend ... --folds`), then pick finals together: one
   CV-best, one hedge with a different risk profile (`final-submission-selection`).
5. Split the remaining daily submissions in advance; keep one shared ledger or one naming convention.

## 6. Several competitions at once

At most one main competition and one side competition. Their endgames (last two weeks) must not
overlap. The endgame needs full attention: robustness checks, inference within limits, and final
picks. Side competitions are good for practising a new modality or trying a teammate before a merge.

## 7. After every competition

Run `/kg-postmortem`: lessons go to `~/.kaggle-gm/lessons.md`, which feeds the next competition's
recon. Publishing a good write-up builds the reputation that brings strong teammates.
