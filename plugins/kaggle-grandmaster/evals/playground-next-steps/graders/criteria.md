---
type: llm
---

PASS when the plan follows the Playground winning pattern, with at least three of these as main steps:

- Append the original real dataset the synthetic data was generated from (flagged with an indicator column,
  after checking the rules).
- Train diverse model families on the same frozen folds (CatBoost, XGBoost, neural nets such as MLP/TabM,
  plus simple ones like linear/kNN/SVR), saving OOF and test predictions for every model.
- Feature engineering that suits synthetic data: count/frequency encodings, rounding or digit features, group
  aggregates, out-of-fold target encoding, or an automated feature search on the frozen folds.
- Ensemble the OOFs with hill climbing or a ridge/logistic stack validated honestly (nested), and average seeds.

It should also say to choose the final submissions on CV rather than the public LB, since small public
differences (around 0.001) are noise in Playground. FAIL if the plan is mainly more tuning of the single
LightGBM, or choosing submissions by public LB.
