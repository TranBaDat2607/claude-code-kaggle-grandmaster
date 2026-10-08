---
type: llm
weight: 1
---

The response recommends grouping folds by patient_id (GroupKFold, or StratifiedGroupKFold to also balance
the 8% positive rate) so no patient appears in both train and validation, and explains that random
(Stratified)KFold would leak patient-specific information and make CV over-optimistic versus the LB.
Strong answers also mention at least one of: freezing/saving the fold assignment and reusing it for every
model so OOF predictions are blendable; checking near-duplicate images across patients; or verifying CV
against LB. Fail if it recommends plain random KFold or StratifiedKFold without grouping by patient.
