---
type: llm
---

PASS when the plan treats wall time, not GPU-hours, as the binding limit and changes the schedule because of it,
with all three of these:

1. It says (in any words) that quota is not the constraint here (six accounts, hours that would expire unused)
   and that the time to get each result (~12 h per session, only a handful of cycles in 4 days) is.
2. It proposes training one model on both T4s (DDP / `torchrun --nproc_per_node 2` / data parallel, or an
   equivalent such as accelerate or the HF Trainer on two GPUs) to get each idea's result in roughly half the
   time, and/or running ideas in parallel sessions across the accounts, instead of two averaged copies of one
   recipe per session.
3. It keeps extra copies or seeds for the ideas that win, not for every screen.

FAIL if it keeps "two copies per session, one per GPU" for every idea without weighing the time it costs, or if
it rations GPU-hours as if they were scarce.
