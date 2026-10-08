---
name: leaderboard-boosters
description: Advanced techniques that move a solid solution into the medal zone — pseudo-labelling, knowledge distillation, test-time augmentation, external data and pretraining, data cleaning/relabelling, multi-stage pipelines, auxiliary targets, leak exploitation, and full-data retraining. Use when a strong baseline exists and the user wants extra leaderboard gains.
---

# Leaderboard Boosters

Apply these only once validation is trustworthy — each one can silently leak.

## Pseudo-labelling (PL)

1. Train an ensemble; predict test (and/or external unlabelled data).
2. Select confident predictions (or keep *soft* labels for all), add them to training.
3. Retrain; repeat 1–3 rounds.
**Leak-safe protocol:** pseudo-label only *test / unlabelled* data, add the same PL set to every
fold's training data, and keep validation folds pure (real labels only). Never pseudo-label
training rows with predictions from a model that saw their validation fold — that leaks the
validation labels back into training and inflates CV. Gains are largest when test is big relative to train or domain-shifted.
Soft labels + a weight < 1 for PL samples is usually safer than hard thresholds.

## Knowledge distillation

Train a big ensemble (teacher), train a student on a mix of hard labels and teacher soft
labels (OOF predictions for train rows, test predictions for PL rows). Use it to: fit inference
time limits, denoise labels, compress a 20-model blend into 1–2 models.

## Test-time augmentation (TTA)

Flips/rotations/multi-scale for images, multiple crops for audio, different truncations
(head/tail) for long texts, reverse-complement in genomics. Validate TTA on OOF — some TTAs hurt
(e.g. flips when orientation carries meaning).

## External data & pretraining

- Check rules: external data must usually be public and freely available to all, sometimes
  declared in a forum thread before a deadline.
- Previous competitions on the same task, public datasets, original data behind synthetic
  Playground data, pretraining on in-domain unlabelled data (MLM for text, self-supervised for
  images), larger pretrained checkpoints.
- Validate only on competition data; check external data isn't a copy of test (that would be
  a leak, and possibly against the rules).

## Data cleaning

- Find label noise with OOF: highest-loss training samples; inspect them. Options: drop, relabel
  (if allowed), soft-label, or down-weight.
- Remove near-duplicates that conflict; fix obvious data-entry errors consistently in train and test.

## Multi-stage pipelines

Detection/localisation → crop → classification; candidate generation → re-ranking; coarse →
fine segmentation; per-group models after a global one. Each stage gets its own CV and OOF.

## Auxiliary targets & multi-task

Predict related columns available only in train (e.g. sub-scores, metadata, segmentation masks)
as auxiliary heads — regularises and transfers information unavailable at test time.

## Leaks

Allowed unless rules say otherwise, but: verify on CV, combine with a genuine model (hosts may
patch data), and be transparent with the user about relying on one. Typical: ID/order/time
correlations, duplicates between train and test, file metadata, public external sources.

## Full-data retraining

Retrain the final configuration on 100% of training data (fixed number of iterations/epochs ≈
1.1–1.2× the fold average) and blend with or replace fold models. Usually +, but unverifiable on
CV — keep the fold-average version as one of the final picks.
