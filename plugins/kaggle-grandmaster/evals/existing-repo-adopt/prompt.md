---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

I just installed the kaggle-grandmaster plugin. I'm about two months into a Kaggle code competition
(`rsna-knee`, 9h runtime limit, internet off, 5 submissions a day, ends 2026-12-01). My repo already
has its own experiment log (`experiments.csv` with CV per run), Kaggle kernel folders with
`kernel-metadata.json`, a fixed validation split, and I have made 40 submissions so far, 3 of them today.
The session start message says I'm not in a competition workspace and to run `/kg-start rsna-knee`.
Should I run it? How do I get the plugin's hooks and tools working on this repo?
