---
name: kaggle-gpu
description: Training models on Kaggle's free GPUs with a limited weekly quota — T4x2 vs P100 choice, quota and session limits, remote training kernels (kgkit gpu build/push/wait/collect), screening ideas on one fold with learning-curve pruning, packing two jobs per 2xT4 session, resuming past the 12h limit, CPU kernels for preprocessing, input-pipeline throughput, and budgeting GPU hours until the deadline. Use when training on Kaggle notebooks/kernels, planning or rationing GPU quota, choosing an accelerator, or when a remote run was slow, idle, timed out or wasted hours.
---

# Training on Kaggle GPUs

The quota is the scarce resource: score gained per GPU-hour is what you optimise, not wall time.
Unused quota expires at the weekly reset, so hours you don't spend are also lost.

## 1. Facts to work from (verify live, they change)

- `kaggle quota` (CLI ≥ 2.2): GPU hours used/remaining and `refreshAt` (weekly reset, UTC).
  `python -m kgkit gpu quota` caches it in `.kaggle-gm/gpu_quota.json`, adjusts for kernels still running,
  and warns when hours are about to expire unused.
- Per-session limit 12 h (GPU/CPU). The push timeout `-t` caps a run below that, and a capped run never
  burns more than its cap.
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
quota / plan        ->  how many hours, until when, what to reserve
build               ->  kernels/<name>-train/: runner + your src/ + kgkit + competition.json + data/folds.csv
push                ->  quota check, -t cap, --accelerator, logs the run in .kaggle-gm/gpu_runs.jsonl
wait                ->  polls every 90 s; prints the log tail on error   (run it in the background)
collect             ->  outputs (no weights by default) -> ledger import, artefacts, GPU-hour accounting,
                        diagnostics (idle GPU, input-bound, deadline hit, pruned)
log                 ->  every remote run with its cost and result
```

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
| deadline hit | `--resume-from`; next time raise `--hours` or lower epochs/res |
| pruned | discard or change the idea materially; don't re-run it unchanged |
| fold failed | read the log tail; reproduce with `--smoke` locally before re-pushing |

Throughput tuning on T4 (AMP, channels_last, compile, batch size, dataloaders, DDP vs fold-parallel)
is covered in [references/throughput.md](references/throughput.md).

## Guardrails

- `kgkit gpu push` (and raw `kaggle kernels push` with GPU) spends the user's quota and is
  outward-facing. Confirm with the user unless they authorised remote runs for this task.
- Never print credentials. The username comes from `KAGGLE_USERNAME` or the `username` field of kaggle.json.
- Keep kernels private (the default) unless the competition rules require publishing.
