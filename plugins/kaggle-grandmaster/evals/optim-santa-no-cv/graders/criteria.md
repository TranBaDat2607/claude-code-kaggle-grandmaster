---
type: llm
---

PASS when the response recognises that ML validation does not apply to the submitted solution:

1. The score is a deterministic objective of the submitted file, so there is no generalisation gap to
   cross-validate and no sampling-based public/private shake-up to fear for a fixed instance; the leaderboard
   score is the cost of the file. Advising to confirm in the rules that the private score uses the same instance
   is fine.
2. It redirects the effort to optimisation: a fast exact scorer checked against the official one (or the sample
   submission), a good initial solution (greedy, problem-specific heuristics or an exact solver such as MIP/CP),
   local search (moves such as swaps/2-opt, simulated annealing, large neighbourhood search) with fast delta
   evaluation, tuning the annealing parameters directly on the objective, many restarts, and keeping the best
   solution file versioned. Naming several of these is enough.

Optionally tuning the annealing parameters across several generated instances, presented as a secondary idea
for parameters that transfer, is acceptable.

FAIL if it recommends k-fold CV or a holdout over the items/rows of the competition's instance, or treats
public/private shake-up as a major risk for a fixed instance.
