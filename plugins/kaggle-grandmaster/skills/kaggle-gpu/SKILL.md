---
name: kaggle-gpu
description: Training models on Kaggle's free GPUs and TPUs with limited weekly quotas — T4x2 vs P100 vs TPU v5e-8 choice, quota, session and concurrency limits, remote training kernels (kgkit gpu build/push/wait/collect/release), screening ideas on one fold with learning-curve pruning, packing two jobs per 2xT4 session, resuming past the 12h limit, porting to torch_xla and the TPU queue, CPU kernels for preprocessing, input-pipeline throughput, and budgeting accelerator hours until the deadline. Use when training on Kaggle notebooks/kernels, planning or rationing GPU/TPU quota, choosing an accelerator, or when a remote run was slow, idle, queued, timed out or wasted hours.
---

# Training on Kaggle GPUs

The quota is the scarce resource: score gained per GPU-hour is what you optimise, not wall time.
Unused quota expires at the weekly reset, so hours you don't spend are also lost.

## 1. Facts to work from (verify live, they change)

- `kaggle quota` (CLI ≥ 2.2): GPU hours (30 h/week) and **TPU hours (20 h/week, a separate quota)**
  used/remaining and `refreshAt` (weekly reset, UTC). `python -m kgkit gpu quota` caches both in
  `.kaggle-gm/gpu_quota.json`, re-checks the runs it has as in flight (finished ones release their
  reservation), lists what still holds hours, and warns when hours are about to expire unused.
- Per-session limit 12 h (GPU/CPU), 9 h (TPU). The push timeout `-t` caps a run below that, and a capped
  run never burns more than its cap. A push without `-t` reserves the full session limit until the run is
  seen finished.
- Batch sessions at once per account: **2 GPU, 1 TPU**, across all competitions. A push beyond that
  prints `Maximum batch GPU session count of 2 reached` and exits 0. `kgkit gpu push` checks your busy
  sessions first.
- **Inputs are bound at push time.** `kernel_sources` resolve to the outputs that exist when the kernel is
  pushed, not when it starts. A job queued behind a run that is still going starts without that run's new
  output. `kgkit gpu push` refuses when an attached kernel is still queued or running.
- `kaggle kernels logs <user>/<kernel> -f` streams the log of a running GPU kernel (CLI 2.2.4); outputs
  appear when the session ends. For TPU sessions the log stays empty until the end.
- Accelerators (`--accelerator` / `machine_shape`): **`NvidiaTeslaT4`** = 2× T4 16 GB (Turing sm_75,
  fp16 tensor cores, no bf16); **`NvidiaTeslaP100`** = 1× P100 16 GB (sm_60, faster fp32/HBM, no tensor
  cores, no Triton → no `torch.compile`). The quota is charged per session-hour either way, so
  **T4×2 is the default**. One quota hour buys two GPUs, but only if both are busy. Choose P100 only for
  fp32-bound or single-process jobs that cannot be split (verify by measuring step time).
- GPU sessions have few CPU cores (typically 4) and ~29 GB RAM. JPEG/DICOM decoding plus heavy CPU augmentation
  starves the GPUs.
- An interactive GPU session burns quota while it is open, even when idle. Batch runs (`kaggle kernels
  push`, "Save Version") stop charging when the script ends. CPU-only sessions don't touch the GPU quota.
- Output: `/kaggle/working` is saved, about 20 GB. A kernel's output can be attached as input to another
  kernel (`kernel_sources`), which is how resume and train→inference chaining work.

## 2. The pipeline (`python -m kgkit gpu ...`)

```
quota / plan        ->  how many hours (GPU and TPU), until when, what to reserve, what holds a reservation
build               ->  kernels/<name>-train/: runner + your src/ + kgkit + competition.json + data/folds.csv
push                ->  input-kernel and busy-session checks, quota check, -t cap, --accelerator, run log
                        (.kaggle-gm/gpu_runs.jsonl). Also any other kernel folder: `gpu push <dir> --hours H`
                        reads kernel-metadata.json (notebooks, inference and holdout-diagnostic kernels)
wait                ->  polls every 90 s; prints the log tail on error   (run it in the background)
status              ->  re-checks every run still marked in flight; finished ones free their hours
collect             ->  outputs (no weights by default) -> ledger import, artefacts, GPU/TPU-hour accounting,
                        diagnostics (idle GPU, input-bound, deadline hit, pruned)
release <kernel>    ->  drop a reservation by hand (a run that was deleted or never started)
cancel <kernel>     ->  how to stop it: running = website "Stop session"; queued = only `kaggle kernels
                        delete` frees the slot (irreversible: ask the user)
log                 ->  every remote run with its cost and result
```
Raw `kaggle kernels push` calls are logged by the plugin's hook too (every push in a compound command,
each from the folder its `cd` steps lead to), and a re-push of the same kernel replaces its older
reservation. After a `cd` it cannot trace (a variable, a loop) the hook reports the push unlogged: give
`-p` an absolute path there. A push in a command that runs in the background (say, a loop that retries until
a session slot frees up) is logged as pending: its whole cap stays reserved until Kaggle lists a version of
the kernel run after it, which `gpu status` and `gpu wait <kernel>` check (`gpu release <kernel>` if the
push never went through). Your username (for `"id"` in kernel-metadata.json): `python -m kgkit whoami`.

The runner links every attached input into `./data` and points `./artifacts` and `./subs` into
`/kaggle/working`. Your training command then runs unchanged, with the same relative paths and the same frozen
folds (so `folds_hash` matches the local ledger). It runs one process per (job, fold), one per GPU,
sets `KG_DEADLINE` so the templates checkpoint before the session ends, and assembles each finished job once.

```bash
# full 5-fold run of a promoted idea: folds 0-4 spread over both T4s
python -m kgkit gpu build --name cnx384 --hours 7 --dataset me/knee-png-512 \
    --cmd "python src/train_image.py --name cnx384 --img-size 384 --folds {fold}"
python -m kgkit gpu push kernels/cnx384-train          # outward-facing: confirm with the user first
python -m kgkit gpu wait me/cnx384-train --collect     # background
```

Custom scripts work if they follow the contract: accept `--folds <k>` and `--assemble`, write
`artifacts/<name>/fold<k>_oof.npy` (+ `_idx.npy`, `_test.npy`), and use `kgkit.budget.TrainBudget`
for deadline, resume and pruning. The image and transformer templates already do all of this.

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
   - `--prune-against baseline` stops a screen whose best-so-far trails the accepted baseline's
     learning curve, compared at the same fraction of the schedule, by more than the baseline's fold
     std. This margin is deliberately generous, so only clearly bad ideas are stopped, and they then
     cost 30-40% of a run instead of 100%. (`best` is accepted as an alias.)
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

## 4. Budgeting until the deadline (`python -m kgkit gpu plan`)

- Total ≈ hours left now + resets before the deadline × weekly quota. Count the resets from
  `refreshAt` in 7-day steps; one that lands before the deadline counts, even on the last day.
  `gpu plan` does this arithmetic.
- Reserve the **final week's quota** for final-model training plus inference kernels (code competitions
  run GPU inference notebooks too), with 1-2 sessions of slack for failures.
- Exploration budget = the rest. With measured cost per fold c, a screen costs ≈ c/2 quota-hours
  (2 per session) and a full K-fold promotion ≈ c·K/2. Plan about 4-6 screens per promotion.
- Re-plan after every `collect`. Costs are measured, not guessed.

## 5. When a run disappoints

`collect` prints the diagnosis. Act on it:

| Finding | Fix |
|---|---|
| second T4 idle | ≥ 2 tasks per session (2 folds, or `--extra-job`), or DDP |
| GPU util < 60 % | input-bound: pre-resize, uint8, GPU augmentation, fewer CPU transforms, `persistent_workers` |
| too short to diagnose | a smoke/dev run (< 20 min) is mostly start-up; judge throughput on a longer run |
| deadline hit | `--resume-from`; next time raise `--hours` or lower epochs/res |
| pruned | discard or change the idea materially; don't re-run it unchanged |
| fold failed | read the log tail; reproduce with `--smoke` locally before re-pushing |

Throughput tuning on T4 (AMP, channels_last, compile, batch size, dataloaders, DDP vs fold-parallel)
is covered in [references/throughput.md](references/throughput.md).

## 6. TPU v5e-8 when the GPU quota is gone

The TPU quota (20 h/week) is separate and expires at the same reset, so it is free hours once the GPU
quota is spent. Measured on a bandwidth-bound ConvNeXt-small (Oct 2026): one v5e chip ≈ **5 T4s** per
step, bf16 losses matching the T4 run to 3-4 decimals. The catch is the queue:

- Batch TPU jobs wait **10-24 h** (12 h 49 min measured) before they start, plus ~20 min of start-up. One
  batch TPU session per account; a queued job holds that slot for its whole wait.
- Expected finish = now + ~13 h queue + 20 min + (T4 time / 5). Use the TPU when that beats the GPU
  finish: for training longer than a GPU session, or when the GPU quota is gone. For runs under ~12 h
  with GPU hours left, the GPU usually wins on wall time. Queue about half a day before you need it,
  and keep a GPU fallback scheduled for after the reset.
- `machine_shape`: ask for `TpuV5E8`. `TpuV5E1`/`TpuV5E4`/`TpuV6E1` are silently stored as `TpuV5E8`;
  `TpuV6E8` is a separate pool that often has no capacity.
- A queued job cannot be cancelled through the API (it has no session id yet). The only way to free the
  slot is `kaggle kernels delete` (irreversible; ask the user), then a re-push, which starts the queue
  again from zero. So **validate before queueing**: run the XLA path locally on PJRT CPU, then the same
  code on a ~5 min 2×T4 dev run. A bug found after the queue costs a whole queue cycle.
- `kgkit gpu build --accelerator TpuV5E8 --hours <= 9` runs one process per chip for `{fold}` tasks (PJRT
  `TPU_VISIBLE_CHIPS=<k>` with single-chip process bounds), so each fold/seed keeps the per-process batch
  it had on one GPU. A command without `{fold}` gets the whole TPU (`PJRT_DEVICE=TPU`). Hours are
  accounted as TPU-hours, not GPU-hours.
- Porting a CUDA trainer to torch_xla (bf16, static shapes, `mark_step`, `xm.save`, timm checkpointing,
  resuming CUDA checkpoints): [references/tpu-xla.md](references/tpu-xla.md).

## Guardrails

- `kgkit gpu push` (and raw `kaggle kernels push` with GPU or TPU) spends the user's quota and is
  outward-facing. Confirm with the user unless they authorised remote runs for this task.
- Never print credentials. The username comes from `python -m kgkit whoami` (`KAGGLE_USERNAME`, the
  `username` field of kaggle.json, or `kaggle config view` for token logins). It never reads the key.
- Deleting a kernel (the only way to drop a queued TPU job) destroys every version and output: the user
  decides.
- Keep kernels private (the default) unless the competition rules require publishing.
