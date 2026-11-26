# Budget, regime and diagnosis for Kaggle GPU runs

Part of the `kaggle-gpu` skill (section 4).

## Budgeting until the deadline

- Total ≈ hours left now + resets before the deadline × weekly quota. Count the resets from
  `refreshAt` in 7-day steps; one that lands before the deadline counts, even on the last day.
  `gpu plan` does this arithmetic.
- Reserve the **final week's quota** for final-model training plus inference kernels (code competitions
  run GPU inference notebooks too), with 1-2 sessions of slack for failures.
- Exploration budget = the rest. With measured cost per fold c, a screen costs ≈ c/2 quota-hours
  (2 per session) and a full K-fold promotion ≈ c·K/2. Plan about 4-6 screens per promotion.
- Re-plan after every `collect`. Costs are measured, not guessed.
- **Which regime** (the skill's section 1): count the hours of every registered account, not only the active one
  (`gpu quota --all`). It is **time-bound** when those hours exceed what the planned runs will spend before the reset
  (they would expire), or when one run takes >= ~6 h on one T4 and the days left allow only a handful of such
  cycles. `gpu plan` prints the regime from every account's quota (read live; `--cached` uses the last reading), the
  measured run lengths and the deadline, and names any account whose quota it could not read.

## Several accounts
- A run counts against the account that owns its kernel (`<owner>/<slug>`): `gpu quota` lists other accounts' runs
  apart, and `gpu status`, `wait` and `collect` ask Kaggle about each run with its owner's token.
- `gpu push` of a `gpu build` folder picks the account itself: of those whose plannable hours cover the cap, the one
  with the fewest (a 10 h run goes to an account with 11 h left, keeping a full week whole for a longer run), that
  has joined the competition and has a batch session free. It runs as that account without switching, sets the
  kernel id to `<its user>/<slug>`, and says which account it chose. A kernel reading another account's kernel
  output stays with that account; a private dataset of another account is copied, private, to the one that runs it.
- Any other kernel folder (a submission or inference notebook) goes to the active account unless
  `--account <alias>` (or `auto`) is given. Submissions always use the active account, which the person switches in
  the Kaggle pane; the submit hook asks when it differs from the competition's earlier submissions.


## When a run disappoints

`collect` prints the diagnosis. Act on it:

| Finding | Fix |
|---|---|
| second T4 idle | ≥ 2 tasks per session (2 folds, or `--extra-job`), or DDP |
| one result per ~12 h session, deadline close | time-bound: DDP for screens (time-bound DDP, below) |
| GPU util < 60 % | input-bound: pre-resize, uint8, GPU augmentation, fewer CPU transforms, `persistent_workers`. Confirm with the script's `--bench` before the next push |
| too short to diagnose | a smoke/dev run (< 20 min) is mostly start-up; judge throughput on a longer run |
| deadline hit | `--resume-from`; next time raise `--hours` or lower epochs/res |
| stopped by the `--hours` cap (`cancelled`, log ends at the cap) | `gpu collect <kernel>` for what it saved; re-push with a margin (~1.3-1.5x the runtime) and save the model before the long phase. Same cap again stops it at the same point |
| pruned | discard or change the idea materially; don't re-run it unchanged |
| fold failed | read the log tail; reproduce with `--smoke` locally before re-pushing |

Making each quota hour do more work (measuring input-bound vs compute-bound, AMP, channels_last, compile, data
loaders, fold-parallel vs DDP, GBDTs and RAPIDS on the GPU, inference speed) is the `compute-efficiency` skill. With no
local GPU, a short bench session (`gpu build --name bench --cmd "python src/train_image.py --bench" --hours 0.3`)
shows whether a script is input-bound for minutes of quota; `gpu collect` adds its record to `reports/bench.jsonl`
with the speed-up against the previous bench.


## Time-bound DDP

**Time-bound: one model on both T4s.** When a run trains for ~6 h or more on one T4 and wall time binds,
    screen each idea as one model on both GPUs (DDP, `torchrun --nproc_per_node 2` in a `--cmd` without `{fold}`).
    It costs the same GPU-hours as two one-GPU runs side by side, but the result comes 1.6-1.9x sooner (~6-7 h
    instead of ~12 h), so about twice as many ideas get screened before the deadline.
    - Keep the effective batch (micro-batch x accumulation x GPUs) and the schedule of the one-GPU recipe: halve
      the accumulation (or, without accumulation, the per-GPU batch), not the learning rate, so the screen stays
      comparable with earlier runs.
    - Compare against a one-copy control run the same way, never against an average of two copies (averaging
      removes seed noise the screen still has).
    - Spend the extra copies or seeds (two one-GPU runs in one session, as before) only on promoted ideas.
    - Put the trade-off to the user when you plan recurring runs, with the numbers: "two copies per session:
      steadier result, ~12 h" vs "one copy on both GPUs: ~6-7 h, noisier, twice the ideas before the deadline".
    - The training loop must be DDP-safe (checklist in `compute-efficiency`,
      [training-loop.md](../../compute-efficiency/references/training-loop.md#ddp-checklist-for-your-own-loop)).
      `TrainBudget` agrees across ranks by itself; save, log and write OOFs on rank 0 only (`tb.is_main`).
      The plugin's image and transformer templates run one process per fold, not DDP.

