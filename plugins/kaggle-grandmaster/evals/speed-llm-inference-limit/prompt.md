---
description: "Code competition, 9h limit on 2xT4: 14B LLM multiple choice over 25k prompts estimated at 20h with generate(): logits not generation, batching, fp16/quantised, both GPUs, measured"
tags: [portable, capability, compute]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

In a Kaggle code competition (internet off, 9-hour limit, 2x T4) I have to answer 25,000 multiple-choice questions
(A-E) with a fine-tuned Qwen2.5-14B. With Hugging Face generate() and batch size 1, my estimate is about 20 hours. How
can I make it fit?
