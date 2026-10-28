---
type: llm
weight: 1
---

The response sees that Kaggle's auto-selection takes the best public scores when nothing is selected, that
here those are A and B, and that both are public-LB fitted (tilted toward the public labels), so leaving it on
auto-select would make two files whose public score says little about private the final picks. It tells the user
to select the finals themselves on the Kaggle website before the deadline (and notes that there is no API or
command to do it for them), and recommends D (the honest, CV-backed stack) as a pick, with C at most as the
hedge. It does not recommend A or B as a final pick, not even as the hedge. A good answer also says to check the
selection afterwards (for example with `python -m kgkit final`). Fail if it says auto-select is fine, if it
picks A or B as a final, or if it tells the user it can select the finals for them.
