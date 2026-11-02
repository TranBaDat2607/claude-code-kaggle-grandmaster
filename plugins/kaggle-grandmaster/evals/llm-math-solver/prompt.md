---
description: "LLM math-olympiad code competition: local eval set, self-consistency voting, vLLM on 2xT4, time budget"
tags: [portable, regression, nlp]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle math-olympiad competition (AIMO-style): 50 hidden problems, each answer is an integer from 0 to 999.
It's a code competition: 5 hours on 2x T4, no internet. I run a 7B reasoning model with one greedy generation
per problem. I tuned my prompt on the 10 public example problems and get 7/10. How do I improve this, and how
do I make sure it fits the time limit?
