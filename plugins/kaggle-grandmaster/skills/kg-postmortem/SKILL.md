---
name: kg-postmortem
description: After a competition ends — record every public and private score, check what the finals scored, measure which signals predicted the private leaderboard, study the top solutions, and distil reusable lessons into a post-mortem and a cross-competition lessons file.
argument-hint: "[competition-slug]"
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
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit discussions *)'
  - 'Bash(python -m kgkit discussions *)'
  - 'PowerShell(python -m kgkit discussions *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(kaggle competitions leaderboard *)'
  - 'PowerShell(kaggle competitions leaderboard *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
---

# /kg-postmortem

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Competition: `$ARGUMENTS` (default: from `.kaggle-gm/competition.json`).

Every score, rank and final pick below comes from a command's output. Never from notes, memory or the
workspace docs: the endgame often left them wrong.

1. `python -m kgkit ledger sync-lb`: one read of Kaggle's list attaches the public and private score of every
   submission whose description or file name gives a ledger id. It lists the submissions it could not link.
   Link the ones you can identify with `python -m kgkit ledger lb <id> <public> <private>`, and say which you could
   not.
2. `python -m kgkit final`: the outcome. It shows the finals, or the pool Kaggle picked from when nothing was
   selected, with their private scores and which of them the team score matches. It also shows the team's
   private and public rank, the medal zone (computed from Kaggle's table, not read from the site: say so), how
   many teams share our shown score and whether a medal cut-off falls inside that tie, the teams per score within
   10 display units of ours with the cut-offs marked, and the best private score of any submission. It caches the
   private board for step 3. Was our final pick our best private? If not, by how much? When a crowd of teams sits
   within a unit or two of us and a cut-off falls inside it, say that the medal there was a coin flip.
3. `python -m kgkit ledger predictiveness --plot reports/figures`: which signal predicted private, over every
   submission and among the strongest, each with n and a 95% interval. Also the private score and private rank
   each way of picking the finals would have given (best CV, best public, best public with fitted files, ours,
   best possible), the public-to-private drop, and a row for the public-LB-fitted files: how far they fell, and the
   rank the best of them by public would have got as a final. If the team fitted the public LB, that rank is the
   cost of fitting: report it. Read a correlation together with its interval: a
   gap between two signals is a finding only when it is large next to the intervals. Over the whole competition
   any signal looks good, because early gains are large; the top-k rows are the ones that matter. Then: which
   risky components helped or hurt, and how big was the shake-up for us and at the top of the LB
   (`kaggle competitions leaderboard <slug> --show`)?
4. `python -m kgkit discussions sync` (live, so write-ups posted today are included), then
   `python -m kgkit discussions solutions`. Launch `solution-researcher` with those thread ids to
   digest the top-5 write-ups of *this* competition: what did winners do that we didn't? Which of our ideas
   matched theirs?
5. Write `reports/postmortem.md`: result (from step 2), what predicted private (the table from step 3, with n
   and intervals), what worked, what didn't, what winners did, validation lessons, process lessons (time
   allocation, mistakes), and 5 reusable takeaways.
6. Append the reusable takeaways to `~/.kaggle-gm/lessons.md` (create if missing) under a heading that
   names the competition slug, so future competitions start smarter. The SessionStart hook surfaces this file in
   every workspace, and after the deadline its brief looks for the slug there to tell whether this step is done.
7. Fix what the endgame left stale:
   - Append the outcome from step 2 to `reports/final-selection.md`: what was selected or auto-selected, and
     their private scores.
   - Update the notes in the workspace `CLAUDE.md`: the competition is over, the result, and a pointer to the
     post-mortem.
   - Correct any claim that `kgkit final` contradicts.
8. Showing the work is the next step, and it is the user's to start: README rebuilt around the result, license,
   figures, public notebook and weights, a Kaggle write-up. Tell them to run `/kg-showcase`. It audits the repo
   (competition data in git, and whether the remote is public) and asks before anything goes public. Make
   nothing public from here.
