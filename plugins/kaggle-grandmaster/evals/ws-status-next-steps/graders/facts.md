---
type: llm
---

The workspace is a Kaggle competition `kgeval-binary-auc` (metric AUC). Its true state: accepted baseline
0002_lgbm-f2 with CV 0.73887 and public LB 0.7012 (the earlier 0001_lgbm-base: CV 0.70438, LB 0.6790); 1 of 5
submissions used today; the deadline is 20 days away; the idea backlog's top item is target-encoding f1 x f2
quantile bins, then CatBoost for blend diversity, then LightGBM tuning.

PASS when the reply reports this state correctly (at least the accepted baseline with its CV or LB, the
submissions left today or used, and the time left) and recommends a concrete next step consistent with it (the
top backlog idea, or a justified alternative such as building blend diversity).

FAIL if it invents experiments, scores or numbers that contradict the state above, or says it cannot tell
where the competition stands.
