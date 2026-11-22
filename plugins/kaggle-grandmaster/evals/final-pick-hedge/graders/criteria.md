---
type: llm
---

PASS when all three hold:

1. It says 0042 + 0044 is a poor pair: at a correlation of 0.998 their private scores rise and fall together, so
   the second pick adds almost nothing, since Kaggle scores the team by the better of the two.
2. It recommends a second pick with errors of its own, most naturally 0038_nn_mlp (another model line, correlation
   0.91), because the expected best of the pair is what counts, not the second-best CV. The 0.0015 CV gap is
   small next to the private noise of about 6,000 rows.
3. It suggests measuring the choice rather than guessing: `kgkit final pick` (simulates the private leaderboard
   on the OOF and ranks pairs by the expected best score), or an equivalent resampling of the OOF at the private
   size, before selecting on the website.

A plus, not required: noting that the simulation cannot see a shift between the training rows and the private
test, or that `kgkit final` confirms the selection afterwards.

FAIL if it endorses 0042 + 0044 as the finals, picks the pair only by CV order, or suggests scoring candidates on
the vault's labels.
