---
description: "Deterministic optimisation competition: no CV or shake-up worry, write an exact scorer and local search"
tags: [portable, capability, simulation]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm in a Santa-style optimisation competition on Kaggle: I submit a solution file (an assignment of items to
days), and the score is a deterministic cost function of the whole file that the hosts published. I want to
set up 5-fold cross-validation to tune my simulated-annealing parameters, and I'm worried about a
public/private shake-up. How should I set this up?
