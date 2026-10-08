---
name: kg-postmortem
description: After a competition ends — record private scores, measure which signals predicted the private leaderboard, study the top solutions, and distil reusable lessons into a post-mortem and a cross-competition lessons file.
argument-hint: "[competition-slug]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch]
---

# /kg-postmortem

Competition: `$ARGUMENTS` (default: from `.kaggle-gm/competition.json`).

1. Fetch submissions with private scores (`kaggle competitions submissions <slug> --format json`)
   and attach them: `python -m kgkit ledger lb <id> <public> <private>`.
2. Analyse: correlation of CV vs private and public vs private; was our final pick our best private?
   Which risky components helped/hurt? Shake-up magnitude for us and for the top of the LB.
3. Launch `solution-researcher` to digest the top-5 write-ups of *this* competition: what did
   winners do that we didn't? Which of our ideas matched theirs?
4. Write `reports/postmortem.md`: result, what worked, what didn't, what winners did, validation
   lessons, process lessons (time allocation, mistakes), and 5 reusable takeaways.
5. Append the reusable takeaways to `~/.kaggle-gm/lessons.md` (create if missing) so future
   competitions start smarter — the SessionStart hook surfaces this file in every workspace.
