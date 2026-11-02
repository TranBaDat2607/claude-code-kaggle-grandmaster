---
description: "Should not trigger: a game leaderboard in Redis"
tags: [portable, regression, negative]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm building the backend for a mobile game. How do I implement a weekly leaderboard with Redis sorted sets:
the top 100 players, a player's own rank, ties broken by who reached the score first, and a reset every Monday?
Show the commands.
