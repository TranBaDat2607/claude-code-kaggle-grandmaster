# README for a finished competition

The first screen answers three questions: what the competition was, how it went, and what is the author's own work.
Fill it from the commands' output (`kgkit final`, `kgkit ledger predictiveness`, `kgkit ledger list`) and from
`reports/postmortem.md`. Leave out a section that has nothing true to say. Long tables go to `docs/results.md`.

Winners' repositories keep the README short and put the narrative in the write-up: a link to it, the hardware, the
steps to run, the license. A portfolio README for any rank adds the result and the honest split between own and
public work, because that is what a reader checks first.

```markdown
# <Competition title>

<One sentence: the task and the metric.> [Kaggle competition](https://www.kaggle.com/competitions/<slug>)

| | |
|---|---|
| **Private leaderboard** | **<score>, rank <rank> / <teams> (top <p> %)<, medal>** |
| Public leaderboard | <score>, rank <rank> |
| Winning private score | <score> |
| Final notebook | [<name>](<url>): <which submission, its private score> |
| Dates | <start> → <end>, <code competition (offline, N h limit) / file submissions> |

Standings read <date>. <The medal zone is computed from Kaggle's progression table when the site was not
checked; ties at the shown decimals, if they matter.>

**The problem.** <Two or three sentences: what made it hard.>

![<What the figure shows, e.g. "Private score of every submission, rising from 0.80 to 0.94">](docs/figures/private_over_time.png)

## My work and other people's work

<The honest split, with a number: "my own models alone reached <x> private".>

**Mine**
- <component>: <what it does, one line> ([code](<path>))

**From public work**
- <component> by <author> ([link](<url>)): <how it was used>

## What moved the score

| Step | Public | Private |
|---|---:|---:|
| <baseline> | | |
| <step> | | |
| **<final>** | **<x>** | **<y>** |

<One or two sentences: which levers worked and which did not.>

## What I learned

- **<The lesson.>** <Its evidence: numbers, with n and an interval where it matters.>

## Repo guide

<The main files and folders, one line each, starting with where to read first.>

## Reproduce

- Hardware and time: <GPU / CPU, RAM, training time, inference time>
- Environment: <environment.yml or requirements.txt, and the command that installs it>
- Data: accept the competition's rules, then `kaggle competitions download -c <slug>`. The competition data is
  not in this repository.
- Steps, in order, as commands: <prepare data> → <train> → <predict>

## Credit

<Every author whose notebook, dataset, weights or idea the solution uses, with links.>

## License

My code: <license> ([LICENSE](LICENSE)). Competition data: the competition's rules. Third-party notebooks,
datasets and weights keep their authors' licenses: <list them; flag non-commercial and share-alike ones>.
```
