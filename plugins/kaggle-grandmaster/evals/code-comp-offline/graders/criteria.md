---
type: llm
weight: 1
---

The response explains uploading model weights (and code/package, plus pip wheels for any library missing from
the Kaggle image) as Kaggle datasets attached to the notebook and installing offline (e.g. pip --no-index
--find-links), loading pretrained models without internet. It addresses the hidden test: never hardcode test
size/ids, read the test files at runtime, and estimate runtime on a hidden-size (≈20x) test with a safety
margin, trimming folds/TTA if needed. Mentions of memory management, a fallback, using both GPUs, or validating
the output file are a plus. Fail if it ignores the internet-off constraint or the hidden-test scale.
