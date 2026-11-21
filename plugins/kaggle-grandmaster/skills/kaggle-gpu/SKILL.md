---
name: kaggle-gpu
description: "Training on Kaggle's free GPUs and TPUs within the weekly quotas: kgkit gpu remote runs, one-fold screens, TPU via torch_xla. Use when planning GPU/TPU hours, choosing an accelerator, or when a remote run was slow, idle, queued or timed out."
---

# Training on Kaggle GPUs

Two things run out: GPU-hours (a weekly quota per account) and wall time (the days to the deadline, at ~12 h per
session). Find out which one binds before you plan (`gpu plan` prints it):
- **Quota-bound** (the usual case with one account): optimise score per GPU-hour.
- **Time-bound**: the quota across *all* registered accounts is more than the planned runs can spend before the
  reset or the deadline, or one experiment fills most of a session and only a few decide-then-run cycles remain.
  Optimise time to the next result instead (section 3, item 10). Unused quota expires at the weekly reset, so hours
  you don't spend are lost anyway.

Re-check the regime whenever an account is added, the quota resets, or the deadline gets close.

## Guardrails

- `kgkit gpu push` (and raw `kaggle kernels push` with GPU or TPU) spends the user's quota and is
  outward-facing. Confirm with the user unless they authorised remote runs for this task.
- Never print credentials. The username comes from `kgkit whoami` (`KAGGLE_USERNAME`, the
  `username` field of kaggle.json, or `kaggle config view` for token logins). It never reads the key.
- Deleting a kernel (the only way to drop a queued TPU job) destroys every version and output: the user
  decides.
- Keep kernels private (the default) unless the competition rules require publishing.

## 1. Facts to work from (verify live, they change)

- `kaggle quota` (CLI ≥ 2.2): GPU hours (30 h/week) and **TPU hours (20 h/week, a separate quota)**
  used/remaining and `refreshAt` (weekly reset, UTC). `kgkit gpu quota` caches both in
  `.kaggle-gm/gpu_quota.json`, re-checks the runs it has as in flight (finished ones release their
  reservation), lists what still holds hours, and warns when hours are about to expire unused. The live call is
  tried twice; when it still fails the report says `NOT LIVE`, why, and how old the cached value is, and counts the
  runs logged since against it (it cannot see sessions started elsewhere, such as the website or another machine:
  run `gpu quota` again before committing hours on a cached figure). With several accounts registered in the Kaggle
  pane (`~/.kaggle/kg_accounts.json`), `kgkit gpu quota --all` reads each one's hours (as that account, never
  printing a token) and the session brief lists them; runs still go to the active account, which the person switches
  in the pane.
- Per-session limit 12 h (GPU/CPU), 9 h (TPU). The push timeout `-t` (`--hours`) caps a run below that, and a
  capped run never burns more than its cap. A push without `-t` reserves the full session limit until the run is
  seen finished.
- **`-t` is a hard kill, not only a quota reservation.** Kaggle ends the session at the cap wherever the script is
  (status `CANCEL_ACKNOWLEDGED`, i.e. `cancelled`) and keeps only the files already in `/kaggle/working`. A run
  that ends early is charged only the hours it ran, so a margin costs nothing: cap a kernel of your own at
  ~1.3-1.5x its measured runtime. `gpu build` kernels stop themselves a few minutes before the cap with their
  checkpoints saved.
- Batch sessions at once per account: **2 GPU, 1 TPU**, across all competitions. A push beyond that
  prints `Maximum batch GPU session count of 2 reached` and exits 0. `kgkit gpu push` checks your busy
  sessions first.
- **A GPU job starts within a minute or two of its push.** Queued for more than 5 minutes with the session
  slots free, it is usually stuck, not waiting: `gpu wait` and `gpu status` say so. Deleting and pushing it
  again starts it at once (`gpu repush <kernel> --yes`; deleting is irreversible, so ask the user first), or
  build it for the P100 pool. TPU jobs are different: they queue for 10-24 h by design.
- **A 2×T4 push sometimes lands on a bad machine**: one T4 (or none) visible, often several times slower. A
  `gpu build` kernel, and a hand-written one `gpu push` added the hardware monitor to, stop within seconds with
  a `KG_BAD_MACHINE` line; `gpu wait` reads a running GPU kernel's log in its first minutes and says so (also
  for `No CUDA GPUs are available`). The first log lines should list both GPUs. Pushing again usually gets a
  healthy machine at once: `gpu repush <kernel> --yes`, after asking the user (deleting is irreversible).
  `KG_EXPECT_GPUS=0` / `KGMON_EXPECT_GPUS = 0` turns the check off for code that needs one GPU only.
- **Inputs are bound at push time.** `kernel_sources` resolve to the outputs that exist when the kernel is
  pushed, not when it starts. A job queued behind a run that is still going starts without that run's new
  output. `kgkit gpu push` refuses when an attached kernel is still queued or running.
- `kaggle kernels logs <user>/<kernel> -f` streams the log of a running GPU kernel (CLI 2.2.4); outputs
  appear when the session ends. For TPU sessions the log stays empty until the end.
- Accelerators (`--accelerator` / `machine_shape`): **`NvidiaTeslaT4`** = 2× T4 16 GB (Turing sm_75,
  fp16 tensor cores, no bf16); **`NvidiaTeslaP100`** = 1× P100 16 GB (sm_60, faster fp32/HBM, no tensor
  cores, no Triton → no `torch.compile`). The quota is charged per session-hour either way, so
  **T4×2 is the default**. One quota hour buys two GPUs, but only if both are busy. Choose P100 only for
  fp32-bound or single-process jobs that cannot be split (verify by measuring step time). Kaggle's PyTorch of
  November 2026 (2.11, CUDA 13) no longer ships `sm_60` kernels, so PyTorch jobs cannot use the P100 there: check
  `torch.cuda.get_arch_list()` first. Kaggle answers a
  shape it does not take with a bare 400, e.g. `NvidiaTeslaT4Highmem`, which `kernels pull -m` can record for
  a past session. `kgkit gpu push` checks the shape before pushing (`--force` if Kaggle has added a new one).
- **A kernel source of 1 MB or more is refused with a bare 400 too** (a script or a notebook, its outputs and
  pasted images included). Data goes in datasets, not in the source. `kgkit gpu build` warns and names the biggest
  parts of the script, and `kgkit gpu push` stops before sending one that big.
- GPU sessions have few CPU cores (typically 4) and ~30 GB RAM. JPEG/DICOM decoding plus heavy CPU augmentation
  starves the GPUs.
- An interactive GPU session burns quota while it is open, even when idle. Batch runs (`kaggle kernels
  push`, "Save Version") stop charging when the script ends. CPU-only sessions don't touch the GPU quota.
- Output: `/kaggle/working` is saved, about 20 GB. A kernel's output can be attached as input to another
  kernel (`kernel_sources`), which is how resume and train→inference chaining work.

## 2. The pipeline (`kgkit gpu ...`)

`quota` / `plan` (hours, regime, reservations) -> `build` (a kernel of your `src/`, kgkit, competition.json and the
frozen folds) -> `push` (quota, input and session checks, `-t` cap, run log) -> `wait` (in the background) ->
`collect` (ledger import, GPU-hour accounting, diagnosis); `status`, `release`, `repush`, `cancel`, `log` around them.
Your command runs unchanged on Kaggle with the same relative paths and frozen folds, one process per (job, fold), one
per GPU. Before the first `gpu build`, read [references/pipeline.md](references/pipeline.md): the script contract,
input layouts, a kernel of your own (caps, pruning, log echoes), resuming.

## 3. Grandmaster protocol for the most leaderboard per GPU-hour

1. **Nothing gets debugged on Kaggle GPUs.** Run `--smoke` locally (CPU is fine) before every push.
   A crash after 20 minutes of pip install and data linking is pure waste.
2. **Preprocess once, off-quota.** Decode DICOM, resize to the training resolution(s), tokenise,
   and cache as uint8 PNG/npy, either locally or in a CPU kernel. Upload the result as a private dataset and attach it with
   `--dataset`. This is usually the single biggest speed-up (2-5× on image tasks).
3. **Screen, then promote.** Test ideas on **one fold** at reduced cost (lower resolution,
   fewer epochs, or a smaller backbone), against the baseline *run under the same reduced settings on
   the same fold* (a "control" screen, run once per screening setting and reused). The baseline's
   full-run fold score is not a fair reference for a cheaper run.
   - Pack two screens into one session so both T4s work:
     `gpu build --name idea-a --folds 0 --cmd "...idea-a... --folds {fold}" --extra-job idea-b "...idea-b... --folds {fold}"`.
   - `--prune-against baseline` stops a screen that trails the accepted baseline's learning curve, compared at the
     same fraction of the schedule, by more than the baseline's fold std. Two cases: its best score so far trails (an
     idea bad from the start), or, when the run keeps its last epoch (the templates' default), its current score trails
     and has been below its own peak for two evaluations (it tied early, then declined: #119). With `--select best`
     only the first case counts. The margin is deliberately generous, so only clearly losing runs stop, and they then
     cost 30-40% of a run instead of 100%. Stopping only discards the run, so the OOF stays honest. (`best` is
     accepted as an alias of `baseline`.)
   - **Write the stop rule next to the promote rule before the push**, e.g. "stop: pruned against the control at
     margin 0.002 (last mode); promote: z >= 2 on fold 0". A kernel of your own gets the same rule from
     `kgkit.budget` ([references/pipeline.md](references/pipeline.md)): `gpu push` warns when a kernel capped above 6 h has none.
   - Promote on a paired test of that fold's OOF against the control:
     `kgkit ledger compare artifacts/<idea> artifacts/<control> --truth data/train.csv:<target>
     --folds data/folds.csv:fold --fold 0` (`--groups` for patients/sessions). z ≥ 2 → promote;
     a positive gain below that → a second fold or seed first. The bootstrap does not see seed noise,
     so treat a z that only just clears 2 on one seed with suspicion. Comparing against the baseline's fold std
     is the wrong test: one fold has no fold-to-fold variance, and the fold std overstates the noise
     of a paired difference.
   - Record `--screen-gain` and, after the full run, `--full-gain` on the backlog item. `kgkit backlog
     fidelity` then tells you whether these screens rank ideas the way full CV does.
4. **Full CV only for promoted ideas.** Run all folds so the ensemble has OOFs on the frozen split. Cap at
   ~1.3× the estimate (per-fold time from `gpu log` × folds ÷ 2 GPUs + ~15 min overhead).
5. **Progressive resizing.** Train most epochs at a lower resolution and fine-tune the last 20-30% at
   full resolution. This gives most of the full-resolution score for 40-60% of the compute. Choose the
   final resolution from screen curves, not by habit.
6. **Diversity beats seeds.** For the final ensemble, a second backbone family or resolution adds more than
   extra seeds of the same model. Spend late quota on diverse members, then blend on OOF
   (`/kg-ensemble`). Seeds are for the last spare hours.
7. **Resume, don't restart.** A run that hits the deadline checkpoints (`fold<k>_resume.pt`, curve,
   status). Rebuild with `--resume-from <user>/<kernel>` (new slug, e.g. `--slug <name>-train-r2`):
   finished folds are skipped and timed-out folds continue mid-schedule.
8. **Chain, don't download.** Attach the training kernel's output to the inference kernel
   (`kernel_sources`) instead of downloading weights and re-uploading them. `collect` skips weights by default.
9. **Use it or lose it.** Hours left at the weekly reset are gone. Plan the current week's remainder
   first and fill it completely before the reset: screens, then the first promotion or a full-fold run
   of the current best (always useful as a baseline OOF and ensemble member). In later weeks, top up
   the last hours with queued, already-validated work (extra folds/seeds of ensemble members, the
   final-resolution variant, pseudo-label rounds). Never start untested exploration just to burn hours.
10. **Time-bound: one model on both T4s (DDP).** When wall time binds, screen each idea as one model on both
    GPUs (1.6-1.9x sooner, same GPU-hours): [references/plan-and-diagnose.md](references/plan-and-diagnose.md#time-bound-ddp).

## 4. Budget, diagnose, TPU

- **Budget to the deadline:** `gpu plan` (hours left, resets, final-week reserve, measured costs, the regime). Plan
  ~4-6 screens per promotion; re-plan after every `collect`.
  [references/plan-and-diagnose.md](references/plan-and-diagnose.md).
- **A run that disappoints:** act on each `collect` finding (idle T4, input-bound, deadline, cap, pruned):
  [references/plan-and-diagnose.md](references/plan-and-diagnose.md#when-a-run-disappoints). Speed: `compute-efficiency`.
- **TPU v5e-8 when the GPU quota is gone:** a separate 20 h/week quota, ~5 T4s per chip, but a 10-24 h queue and a
  torch_xla port validated before queueing: [references/tpu-xla.md](references/tpu-xla.md).
