# journey/README.md: how the solution developed

For the reader who wants to learn from the competition, not only see the result. It tells the work phase by phase,
in the order it happened, with the numbers and the dead ends. `journey/experiments.md` (written by
`kgkit ledger journey`) holds every record; this file is the story that makes sense of them.

What makes a journey worth reading:

- **Organised by question, not by file.** Each phase asks something ("does stacking public OOFs beat our own
  models?") and ends with what the numbers answered.
- **Numbers, not adjectives.** Each claim has its record id and its score; gains have their noise ("+0.0021,
  5/5 folds, paired SE 0.0006").
- **Dead ends with their evidence.** What was expected, what happened, why it failed (a gate set in advance, a gain
  inside the noise, a public score that did not hold on private). Readers learn as much from these as from what
  worked, and winners' repositories almost never show them.
- **Links to the code as it ran.** A record's code link opens the commit (or the snapshot) it ran from; a diff link
  shows what changed from its parent. Link the record (`experiments.md`), the study (`../studies/<name>/`) or the
  diff, rather than describing code.
- **Turning points named.** The two or three moments the direction changed, and what triggered each: an error
  analysis, a forum post, a plateau, a failed gate.
- **Honest about what was not yours.** When a phase used public notebooks or datasets, say so and credit them, as
  the main README does.

Fill it from the local sources only (the ledger, `experiments.md`, the backlog with its evidence and results, the
commit messages, the workspace `CLAUDE.md` notes, `studies/*/README.md`, `reports/`). Leave out a section with
nothing true to say. Quote numbers as the ledger has them.

```markdown
# How the solution developed

<One paragraph: where the work started (the baseline and its score), where it ended (the finals and their private
scores), and the one or two things that made the difference.> The final pipeline alone is in
[`../solution/`](../solution/README.md); every experiment is in [experiments.md](experiments.md).

![<What the figure shows, e.g. "Private score of every submission over time, rising from 0.80 to 0.94 with a plateau in
week 3">](../docs/figures/private_over_time.png)

## The map

<The lineage graph from experiments.md is the map: link it, or paste the Mermaid block here if it is small.>

| Phase | Question | Answer | Best <CV/holdout> | Records |
|---|---|---|---:|---:|
| [1. <name>](#1-<anchor>) | <what this phase asked> | <what the numbers said, in a few words> | <score (id)> | <n> |

## 1. <Phase name> (<dates>)

**The question.** <What this phase set out to learn, and why then: what the previous phase left open.>

**What I tried.**

| Record | Change | <CV> | Δ vs parent | Decision |
|---|---|---:|---:|---|
| [<id>](experiments.md#1-<anchor>) | <one line> | <score> | <gain> | <keep / discard: why> |

<Two to five sentences on what happened, with the numbers. Link the code of the change that mattered:
[the diff](<compare link from experiments.md>).>

**Dead ends.** <Each idea that did not work: what was expected, what happened, why. A study that settled it:
[`studies/<name>`](../studies/<name>/README.md).>

**What I took from it.** <One or two lessons, each with its evidence.>

## 2. <Phase name> (<dates>)

<The same shape.>

## Turning points

- **<Date>: <what changed>.** <What triggered it: an error analysis, a forum post, a plateau, a gate.> <Its effect,
  in numbers.>

## What I would do differently

- **<The change.>** <Why, with the evidence from the phases above.>

## Reproducing a step

Every record in [experiments.md](experiments.md) lists the command it ran and links the code it ran from. To run a
step as it ran: `git checkout <commit>` (the code link's commit), then its command. The snapshot commits live under
`refs/kg/`; `git fetch origin 'refs/kg/*:refs/kg/*'` fetches them.
```
