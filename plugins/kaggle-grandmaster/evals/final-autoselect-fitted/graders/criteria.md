---
type: llm
---

PASS when all three hold:

1. It sees that Kaggle's auto-selection takes the best public scores when nothing is selected, that here those
   are A and B, and that both are public-LB fitted (tilted toward the public labels), so leaving it on
   auto-select is not fine.
2. It tells the user to select the finals themselves on the Kaggle website.
3. It recommends D (the honest, CV-backed stack) as a pick, with C at most as the hedge, and does not recommend
   A or B as a final pick, not even as the hedge.

A plus, not required: saying that the assistant cannot do the selection (there is no API or command for it), and
checking the selection afterwards (for example with `python -m kgkit final`).

FAIL if it says auto-select is fine, picks A or B as a final, or tells the user it can select the finals for them.
