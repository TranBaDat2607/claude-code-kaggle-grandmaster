# Modelling the generator of synthetic data: one worked example

Evidence from one competition, Playground Series S6E10 (binary target, AUC, synthetic rows made by a
GReaT-style GPT-2 model fine-tuned on an original dataset). Treat it as an example of the idea in
`tabular-mastery` section 7, not as a recipe for every Playground: other months use other generators.

## What the top places did
- 1st: a GPT-2 replica of the generator, trained on the original data, then fine-tuned on the labelled
  training fold with the target written first. Each row is scored
  `log P(x | T) - log P(x | F) + log prior odds`, with a log-sum-exp over about 30 column orders. One such pair
  gave +4.9e-4 CV (+3.7e-4 public) and was about 80% of the final stack.
- 4th: an any-order autoregressive transformer trained from scratch, its log-likelihood ratio averaged over 32
  column orders; 0.96258 private on its own.
- 7th: a causal transformer over the joint distribution of features and label, 8 orders, 16 seeds.
- 9th and 11th: masked or replaced-token pre-training on train + test, then fine-tuning (+2e-4 private).
- The same generative model trained without labels (a density of the rows) added +0 to the stack.

## How the generator showed itself
Number tokenization artefacts: values split and recombined the way a GPT-2 tokenizer splits digits, and values
the original data never had. A CTGAN-style generator leaves other traces (mode collapse, smoothed tails);
look before assuming.

## Cost on Kaggle
About 10 A100-hours per GPT pair (30-50 T4-hours); the from-scratch transformers took 2-11 hours per 2xT4
kernel. On T4, use fp16 with loss scaling (no bf16). Screen one fold before committing the quota.
