---
type: tool_used
tool: Skill
input_match: '"skill"\s*:\s*"(?:kaggle-grandmaster:)?(?:audio-and-signal|code-competitions|competition-recon|competition-strategy|computer-vision|cv-lb-debugging|deep-learning-training|ensembling|final-submission-selection|grandmaster-playbook|hyperparameter-tuning|kaggle-cli|kaggle-gpu|kg-adopt|kg-baseline|kg-cv|kg-debug|kg-eda|kg-ensemble|kg-experiment|kg-final|kg-gpu|kg-grind|kg-ideas|kg-kernel|kg-postmortem|kg-recon|kg-showcase|kg-start|kg-status|kg-submit|leaderboard-boosters|metric-optimization|nlp-and-llm|recsys-and-ranking|simulation-and-optimization|tabular-mastery|time-series|validation-strategy)"'
min: 0
max: 0
arm: both
weight: 2
---

Should-not-trigger check: no kaggle-grandmaster skill is loaded for a request that is not Kaggle competition work. Scored in both arms, so a plugin that over-triggers shows a negative delta.
