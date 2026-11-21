# kgkit gpu: the pipeline in detail

Part of the `kaggle-gpu` skill (section 2).

```
quota / plan        ->  how many hours (GPU and TPU), until when, what to reserve, what holds a reservation
build               ->  .kaggle-gm/build/<name>-train/ (ignored): runner + your src/ + kgkit + competition.json + data/folds.csv
                        (packed column by column: 700k rows of id,fold take ~0.3 MB of the script, not 2.4 MB)
push                ->  input-kernel and busy-session checks, quota check, -t cap, --accelerator, run log
                        (.kaggle-gm/gpu_runs.jsonl). Also any other kernel folder: `gpu push <dir> --hours H`
                        reads kernel-metadata.json (notebooks, inference and holdout-diagnostic kernels)
wait                ->  polls every 90 s; on cancelled/error prints the log tail, says whether the --hours cap
                        stopped it, and lists the files the session saved (still downloadable with
                        `gpu collect`)   (run it in the background)
status              ->  re-checks every run still marked in flight; finished ones free their hours
collect             ->  outputs (no weights by default) -> ledger import, artefacts, GPU/TPU-hour accounting,
                        diagnostics (idle GPU, input-bound, deadline hit, pruned). For a kernel `gpu build` did
                        not make (no kg_run.json): Kaggle's status is the outcome, the files go to
                        data/ext/kout-<slug>/, the session's hours are accounted, the ledger is yours
                        (`ledger import` / `ledger add`)
release <kernel>    ->  drop a reservation by hand (a run that was deleted or never started)
cancel <kernel>     ->  how to stop it: running = website "Stop session"; queued = only `kaggle kernels
                        delete` frees the slot (irreversible: ask the user)
repush <kernel>     ->  delete a stuck kernel and push its folder again; prints the plan, --yes does it
                        (ask the user first). A deleted kernel's runs stay in the log as deleted
log                 ->  every remote run with its cost and result
```
Raw `kaggle kernels push` calls are logged by the plugin's hook too (every push in a compound command,
each from the folder its `cd` steps lead to), and a re-push of the same kernel replaces its older
reservation. After a `cd` it cannot trace (a variable, a loop) the hook reports the push unlogged: give
`-p` an absolute path there. A push in a command that runs in the background (say, a loop that retries until
a session slot frees up) is logged as pending: its whole cap stays reserved until Kaggle lists a version of
the kernel run after it, which `gpu status` and `gpu wait <kernel>` check (`gpu release <kernel>` if the
push never went through). Your username (for `"id"` in kernel-metadata.json): `kgkit whoami`.

The runner links the attached inputs into `./data` and points `./artifacts` and `./subs` into
`/kaggle/working`. Your training command then runs unchanged, with the same relative paths and the same frozen
folds (so `folds_hash` matches the local ledger). It runs one process per (job, fold), one per GPU,
sets `KG_DEADLINE` so the templates checkpoint before the session ends, and assembles each finished job once.

The frozen split reaches the session cell for cell. `build` packs each column of `data/folds.csv` in the cheapest
exact form: an integer range costs nothing, the folds take a byte per row, other columns go as text, and a big text
column (hash-like ids) is read from `data/train.csv` in the session when that file has it row for row. The runner
writes the file back and checks it against yours; if they differ (a `data/train.csv` that is not the competition's),
it stops before anything runs. If the split is too big even packed (a few million rows), put `data/folds.csv` in a
private dataset (ask the user first) and build with `--split-from <owner>/<slug>`: the run links it from there,
checked byte for byte.

```bash
# full 5-fold run of a promoted idea: folds 0-4 spread over both T4s
kgkit gpu build --name cnx384 --hours 7 --dataset me/knee-png-512 \
    --cmd "python src/train_image.py --name cnx384 --img-size 384 --folds {fold}"
kgkit gpu push .kaggle-gm/build/cnx384-train # outward-facing: confirm with the user first
kgkit gpu wait me/cnx384-train --collect     # background
```

Custom scripts work if they follow the contract: accept `--folds <k>` and `--assemble`, write
`artifacts/<name>/fold<k>_oof.npy` (+ `_idx.npy`, `_test.npy`), and use `kgkit.budget.TrainBudget`
for deadline, resume and pruning. The image and transformer templates already do all of this.

A kernel of your own (`gpu push <dir> --hours H`) has no runner and no deadline: Kaggle kills it at H h (see
section 1). So size H from a measured runtime with a margin, and write the kernel so that a kill costs one phase,
not the run: save the model to `/kaggle/working` right after training, before a long inference or scoring phase;
write predictions in chunks (or per fold) as they are made, so a restart can skip what exists; and print progress
with the elapsed time and an ETA, so the log tail `gpu wait` prints shows where the kill came. When a cap does stop
it, `gpu wait` says so and lists the files saved before the kill.

It has no pruning either, unless its loop asks for it. Attach or embed the control's `curve_fold<k>.json` and call
`TrainBudget(out_dir, fold, epochs, greater_is_better, baseline=<that folder>, margin=<fold std>, mode="last")` each
epoch (`end_epoch` returns `"prune"`), or `kgkit.budget.update_curve` + `should_stop(curve, control, margin=...)` on the
curve file. Copy `curve_fold<k>.json` back when a run resumes in a new session, so the rule sees the whole curve. A
driver that echoes the training log should echo every key line since its last echo (evaluations, epoch done,
scored), not only the latest line: the decline of #119 was in the lines between two echoes.

Where the inputs go (match your local layout, so the command finds them):
- `--input-layout flat` (the default): competition data, `--dataset` and `--kernel` contents side by side in `data/`.
  A name two inputs share (`oof_predictions.parquet`, `submission.csv`, `README.md`) is linked from the first
  only. The run log warns about each one, and `collect` lists them.
- `--input-layout nested`: each input in its own folder, `data/ext/<slug>/` (datasets) and
  `data/ext/kout-<slug>/` (kernel outputs). Competition data stays in `data/`. Use it whenever several inputs
  can share file names, e.g. a stack over public OOF libraries.
- `--dataset <owner>/<slug>=data/<folder>` (or `--kernel`) puts one input where your scripts expect it, e.g.
  the folder you gave `kgkit datasets pull`. `=data` puts it beside the competition data.

An input Kaggle did not mount (not attached, or a kernel still running at push time) is a warning in the log
and a `collect` finding, and so is a missing `--resume-from` kernel, whose folds would be trained again.

```bash
# CPU stack over many OOF libraries (no GPU quota): every library keeps its own files
kgkit gpu build --name stack --hours 2 --accelerator none --input-layout nested \
    --dataset megayak/s6e10-oof mitudru/s6e10-oof=data/ext/mitudru-oof --kernel thisray/s6e10-stack \
    --cmd "python src/stack.py"
```

