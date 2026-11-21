# A local GPU: the same build, on this machine

Part of the `kaggle-gpu` skill (section 2).

```
kgkit gpu build --name <exp> --cmd "python src/train.py --folds {fold}" --hours 4   # the same build as for Kaggle
kgkit gpu push .kaggle-gm/build/<exp>-train --backend local [--hours H]          # starts it, prints the pid and log
kgkit gpu wait local/<exp>-train --collect                                       # in the background
kgkit gpu cancel local/<exp>-train                                               # stops it; finished folds stay
```

What is the same as on Kaggle: the runner checks the fold split before anything runs, runs one fold per GPU
(`CUDA_VISIBLE_DEVICES`), skips folds a `--resume-from` run finished, sets `KG_DEADLINE` from the cap, runs the
assemble command, and writes `kg_run.json`. `collect` imports into the ledger with the same `folds_hash` check.

What differs:

- **No quota, no queue, no 12 h session limit.** `--hours` goes up to 168 and only sets the run's own deadline. The
  run is logged with resource LOCAL, which quota, reservations and `gpu plan`'s run lengths never count.
- **Inputs are links to your files**, under `.kaggle-gm/local/<slug>/v<N>/input/` laid out like `/kaggle/input`:
  the competition is everything in `data/` except `folds.csv`; a dataset is `data/ext/<slug>/` (or
  `data/<slug>/`); a kernel output (`--kernel`, `--resume-from`) is the latest local run of that slug, else what
  `gpu collect` downloaded. Push refuses when one is missing and prints the command that fetches it (`--force`
  runs without it). On Windows without the symlink privilege, folders become junctions and files hard links:
  nothing is copied.
- **One run at a time.** Two runs on one GPU share its memory, so push refuses while a local run is still
  running (`--force` overrides).
- **Speed differs from a T4.** Per-fold cost measured locally does not carry over to Kaggle; budget Kaggle hours
  from Kaggle runs (`gpu log`).
- **The run outlives the conversation.** It is a detached process; `gpu status` and `gpu wait` find it by its pid
  and its `kg_run.json`, and `session.log` beside `working/` has everything it printed.

When to use which: a run that fits the local card's memory goes local first (it costs nothing and starts now).
Kaggle is for runs that need more GPU memory than the card has (a T4 has 16 GB), for running in parallel with a
local run, and for code-competition inference, which must run on Kaggle anyway.

If the session brief says `LOCAL GPU UNUSED`, the installed torch is a CPU build and every local run would train on
the CPU: offer the CUDA build of the same torch version (ask first, it changes their environment).
