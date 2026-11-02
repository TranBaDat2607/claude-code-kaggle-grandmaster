"""Synthetic binary-AUC competition for the shell eval cases. Standard library only, deterministic.

Writes, into the current directory:
  data/train.csv (id, f1, f2, f3, target), data/test.csv, data/sample_submission.csv, data/folds.csv
  preds/oof_base.csv (f1 only), preds/oof_f2.csv (+ f2: a large gain), preds/oof_f3.csv (+ a little f3: a small
  gain that is consistent on every fold, i.e. clearly real in a paired comparison although far below the fold std)
  subs/sub_0001.csv, subs/sub_0002.csv (valid), subs/sub_0003.csv (invalid: one row short, wrong column name),
  subs/sub_0004.csv (valid, never submitted)
"""

import csv
import math
import os
import random

N_TRAIN, N_TEST, N_FOLDS = 2000, 500, 5
rng = random.Random(7)


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def write(path: str, header: list[str], rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


train = []
for i in range(N_TRAIN):
    f1, f2, f3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    y = int(rng.random() < sigmoid(1.2 * f1 + 0.6 * f2 + 0.25 * f3 - 0.5))
    train.append((i, f1, f2, f3, y))
write("data/train.csv", ["id", "f1", "f2", "f3", "target"],
      ((i, f"{a:.5f}", f"{b:.5f}", f"{c:.5f}", y) for i, a, b, c, y in train))
write("data/folds.csv", ["id", "fold"], ((r[0], r[0] % N_FOLDS) for r in train))

test_ids = list(range(N_TRAIN, N_TRAIN + N_TEST))
write("data/test.csv", ["id", "f1", "f2", "f3"],
      ((i, f"{rng.gauss(0, 1):.5f}", f"{rng.gauss(0, 1):.5f}", f"{rng.gauss(0, 1):.5f}") for i in test_ids))
write("data/sample_submission.csv", ["id", "target"], ((i, 0.5) for i in test_ids))

noise = [rng.gauss(0, 1) for _ in train]
# f3 is weak: its model moves every row's logit a little toward the truth, so it ranks better on every fold
# by a small, consistent margin
F3_STEP = 0.012  # +0.0034 AUC over 0002, better on 5/5 folds (paired t ~20) vs a fold std of ~0.018
for name, w2, step in (("base", 0.0, 0.0), ("f2", 0.6, 0.0), ("f3", 0.6, F3_STEP)):
    write(f"preds/oof_{name}.csv", ["id", "pred"],
          ((r[0], f"{sigmoid(1.2 * r[1] + w2 * r[2] + step * (2 * r[4] - 1) + e):.6f}")
           for r, e in zip(train, noise)))

for k in (1, 2, 4):
    write(f"subs/sub_000{k}.csv", ["id", "target"], ((i, f"{rng.random():.5f}") for i in test_ids))
write("subs/sub_0003.csv", ["id", "prediction"], ((i, f"{rng.random():.5f}") for i in test_ids[:-1]))
