---
name: recsys-and-ranking
description: "Playbook for Kaggle recommendation and ranking competitions: candidate generation, GBDT rankers, MAP@K/NDCG. Use when predictions are ranked lists of items per user, query or session."
---

# RecSys & Ranking Playbook

The winning architecture is almost always **two-stage**: generate ~50–200 candidates per
user/session with cheap methods (maximise recall@N), then re-rank them with a GBDT ranker using
rich user–item features.

## 1. Validation

Reproduce the test construction in time: train on weeks 1..k, label week k+1 (or truncate
sessions at a random point like the test). Measure **candidate recall@N** (upper bound for the
ranker) and the final metric (MAP@12, recall@20, NDCG). Use a subsample of users for speed;
confirm on more.

## 2. Candidate generation (union of sources)

- User history: recently interacted / repurchased items (often the strongest).
- Co-visitation matrices (item → items seen together within a time window), weighted by time
  and event type (click/cart/order); separate matrices for clicks vs buys.
- Popularity: global and per segment (age bucket, region, category), recent-window popularity.
- Embedding similarity: item2vec/word2vec on sessions, ALS/BPR (implicit), two-tower models;
  approximate kNN (faiss).
- Item-attribute similarity (same product group/colour), "bought together".
Track each source's recall contribution; tune N per source.

## 3. Ranker features

- User: activity counts, recency, diversity, price sensitivity, preferred categories.
- Item: popularity over multiple windows and trends, price, conversion rates, attributes.
- User × item: past interactions with the item / its category, time since last interaction,
  co-visitation scores from each source, candidate rank within each source, embedding similarity.
- Context: time to target period, seasonality.
Labels: whether the candidate is interacted with in the target window.

## 4. Ranking models

LightGBM `lambdarank` / XGBoost `rank:ndcg` / CatBoost `YetiRank` with query groups, or binary
classifiers (often comparable). Negative subsampling for speed; ensemble rankers trained on
different windows.

## 5. Engineering

Data is huge: use polars/cuDF, int32 ids, parquet, chunked processing per user shard, GPU
co-visitation computation. Cache candidates & features per week. Rankers: GPU XGBoost/CatBoost on millions of
pairs. How-to and the hardware limits (cuDF needs a T4, not a P100; Linux or WSL2): `compute-efficiency`
(`references/tabular.md`).

## 6. Search relevance / LTR with text

Cross-encoders or LLM re-rankers over (query, document) pairs; GBDT stack over BM25,
embedding similarities, and cross-encoder scores; group CV by query.

## 7. Cheap screens (sweeps)
Many values of one lever are one sweep (`/kg-grind --sweep`): screened cheap against a control at the same fidelity, the best confirmed by a full run judged as the best of all of them.

Screen on whole users or queries (`--group user_id`), on the latest week when the split is by time (`--time-col`).
Per-query metrics (MAP@k, NDCG) log a unit table, one row per query (`mean`): the comparison is paired per query.
Cheap settings that usually keep the ranking: fewer users, sampled negatives in training, fewer ranker rounds. Not
a screen question: candidate generation (its recall changes with the candidates a screen keeps).
