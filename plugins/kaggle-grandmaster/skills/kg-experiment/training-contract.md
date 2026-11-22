# The training contract: a pipeline of your own

The templates (`train_gbdt.py`, `train_image.py`, `train_transformer.py`) follow it. A pipeline of your own (multi-stage,
DICOM, 3D, an LLM harness, a game agent) follows it too, and then every kgkit tool works with it: the paired verdict
when it logs, `kgkit recheck`, screens and sweeps, Kaggle GPU runs.

## Folds
- Read the frozen split from `os.environ.get("KG_FOLDS_FILE", "data/folds.csv")`: `kgkit recheck` points it at a
  re-drawn split. Never draw folds in the script.
- Nothing about the model is chosen on the fold it predicts: no early stopping, epoch or checkpoint picked on it.
  Hold a slice of the training part out for that, and log `oof_selection="independent"`.

## Command line
- Run it from the workspace root as `python src/<script>.py ...`: the record keeps that command, so
  `kgkit screen control`, `kgkit recheck` and sweeps can run it again.
- `--name`, `--parent <baseline id>` (its lineage, and what it is compared with), `--smoke` (one fold, a few steps,
  nothing logged). One change per flag, never a copy of the script.
- `--folds <k ...>` trains those folds only and writes `artifacts/<name>/fold<k>_oof.npy`, `fold<k>_idx.npy` (the rows
  it predicted) and `fold<k>_test.npy`; the run that completes the set assembles and logs, and `--assemble` does only
  that. `kgkit gpu build` runs one fold per GPU this way (`kaggle-gpu` skill).
- A fold of more than ~30 min: `kgkit.budget.TrainBudget` for the session deadline, resume checkpoints and pruning
  against the baseline's learning curve.

## Logging
```python
from kgkit.experiment import Ledger, report_comparison

rec = Ledger().log(args.name, cv, fold_scores=fold_scores, oof=oof, test_pred=test_pred, parent=args.parent,
                   params=vars(args), seeds=[args.seed], oof_selection="independent")
report_comparison(rec, parent=args.parent, truth="data/train.csv:<target>")  # prints the verdict, never raises
```
- `oof` has one row per row of `data/train.csv`, in its order, NaN where no fold predicted.
- A metric that is not one array of row predictions (Dice or IoU per image, F-beta over the particles of a tomogram,
  mAP, accuracy per problem, the results of matches) is logged as a **unit table**: one row per unit with what the
  metric sums or averages, its fold, and the group it belongs to. The comparison then resamples units, or whole
  groups:

```python
from kgkit import units as U

u = U.make(tomo_ids, counts={"tp": tp, "fp": fp, "fn": fn}, agg="fbeta", params={"beta": 4},
           fold=tomo_fold, group=tomo_site)       # or score=<per-unit score> with agg="mean"
rec = Ledger().log_units(args.name, u, parent=args.parent)
report_comparison(rec, parent=args.parent)
```
  Aggregates: `mean`, `ratio`, `iou`, `dice`, `fbeta` (per class with `tp_<c>` columns), `precision`, `recall`,
  `custom` (`params={"fn": "module:function"}`). A table written to a file is logged with
  `kgkit ledger add <name> auto --units table.csv --units-agg fbeta --units-param beta=4`.

## Screens (`KG_SCREEN`)
A screen is a cheap run on a frozen subset (`kgkit screen new`), compared with a control at the same fidelity:

```python
from kgkit import screen as SC

s = SC.active()                                  # None for a full run; ValueError for an unknown screen
keep = s.mask(ids) if s is not None else None    # ids as in the screen's source (the id column, or --ids units)
folds_to_run = (s.folds if s is not None and s.folds else None) or all_folds
out_dir = Path("artifacts") / (args.name + SC.run_tag())   # apart from a full run of the same name
```
- Train and score only the `keep` rows of the screen's folds; NaN in the OOF elsewhere; no test predictions and no
  submission (a screen is never submitted).
- `KG_FIDELITY='{"epochs": 3}'`: the cheaper settings, recorded as given. Take them from your flags (a sweep passes the
  rung's settings in `{rung_args}`: `--epochs 3`) or read them with `SC.settings()`.
- `KG_CONTROL_OF` and the `KG_SWEEP*` variables need nothing from the script: `Ledger.log` records them.

## Seeds (`KG_SEED`)
Train with `seeds = run_seeds(args.seeds)` (`from kgkit.experiment import run_seeds`; an int or a list). Under
`KG_SEED` it returns new seeds, as many as configured, and `Ledger.log` records the run as a seed replica of
`KG_REPLICA_OF`: the re-run `kgkit ledger seed-noise` prints, which measures the training noise every paired
comparison then adds. `SC.run_tag()` keeps its files apart from the record it re-runs.

## Output
- Print ASCII only: Windows consoles are often cp1252.
- Print progress with the elapsed time per epoch and fold, so a log tail shows where a long run is.
