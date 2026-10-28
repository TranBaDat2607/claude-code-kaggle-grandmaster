---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

Playground competition, 3 days left, 2 final submissions allowed. I haven't touched the "select final
submissions" part of the Kaggle website, I'll just let it auto-select. My scored submissions (AUC):
A) public 0.96297, a ranking tilted toward an estimate of the public labels that I built from other people's
   public scored files (tagged public-lb-fitted in my ledger)
B) public 0.96285, the same tilt with a smaller step (also public-lb-fitted)
C) public 0.96214, a blend of three public notebooks' submission files with round weights (LB-only)
D) public 0.96185, CV 0.96225, my own nested-validated stack of 12 models (the accepted baseline)
Is it fine to leave it on auto-select? What should I do?
