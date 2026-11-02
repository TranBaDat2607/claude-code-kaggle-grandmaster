#!/usr/bin/env bash
# Builds a small kgkit competition workspace in the current directory, for the eval cases tagged `shell`.
# A case's setup.sh calls it:  bash "$(dirname "$0")/../_fixtures/make_workspace.sh" [options]
#
#   (always)       competition `kgeval-binary-auc` (AUC, 5 subs/day, deadline in 20 days), data/, frozen folds,
#                  ledger 0001_lgbm-base (baseline) -> 0002_lgbm-f2 (keep, public LB 0.7012), 3 backlog ideas,
#                  submissions log: sub_0001 yesterday, sub_0002 today (1 of 5 used today)
#   --candidate    also logs 0003_lgbm-f3: CV +0.0034 over 0002 (fold std 0.018) but better on 5/5 folds (paired
#                  t ~20: a real gain), with no decision recorded
#   --budget-used  four more logged submissions today: 5 of 5 used
#   --creds        a fake ~/.kaggle/kaggle.json in the run's temporary HOME (never anywhere else)
#
# It runs as you, outside the agent's sandbox (`claude plugin eval --scaffold`), with the Python on your PATH, and
# builds the state with this plugin's own kgkit so the fixture always matches the current file formats.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGIN="$(cd "$HERE/../.." && pwd)"

# Only ever inside an eval run's temporary workspace: never in a real project or home folder.
case "$PWD" in
  *claude-eval-*) ;;
  *) echo "make_workspace.sh: refusing to run outside a claude plugin eval workspace (cwd: $PWD)" >&2; exit 1 ;;
esac

CANDIDATE=0 BUDGET_USED=0 CREDS=0
for arg in "$@"; do
  case "$arg" in
    --candidate) CANDIDATE=1 ;;
    --budget-used) BUDGET_USED=1 ;;
    --creds) CREDS=1 ;;
    *) echo "make_workspace.sh: unknown option $arg" >&2; exit 2 ;;
  esac
done

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
kg() { PYTHONPATH="$PLUGIN" "$PY" -m kgkit "$@" >/dev/null; }
FOLDS=(--truth data/train.csv:target --folds data/folds.csv:fold)

"$PY" "$HERE/gen_data.py"
DEADLINE="$("$PY" -c 'import datetime as d; print((d.date.today() + d.timedelta(days=20)).isoformat())')"
kg init kgeval-binary-auc --metric auc --task tabular --target target --id-col id --daily-subs 5 --deadline "$DEADLINE"

kg ledger add lgbm_base auto --pred preds/oof_base.csv:pred "${FOLDS[@]}" --notes "LightGBM, f1 only"
kg ledger decide 0001 baseline "first model"
kg ledger add lgbm_f2 auto --pred preds/oof_f2.csv:pred "${FOLDS[@]}" --parent 0001 --notes "adds f2"
kg ledger decide 0002 keep "better on 5/5 folds"
kg ledger lb 0001 0.6790
kg ledger lb 0002 0.7012
if [ "$CANDIDATE" = 1 ]; then
  kg ledger add lgbm_f3 auto --pred preds/oof_f3.csv:pred "${FOLDS[@]}" --parent 0002 --notes "adds f3"
fi

kg backlog add "Target-encode f1 x f2 quantile bins (OOF, frozen folds)" --gain 3 --prob 0.4 --cost 1
kg backlog add "CatBoost on the same features for blend diversity" --gain 2 --prob 0.6 --cost 1
kg backlog add "Tune LightGBM num_leaves / min_child_samples" --gain 1 --prob 0.5 --cost 2

# Submission log in the format hooks/post_tool.py writes (the duplicate and daily-budget checks read it).
"$PY" - "$BUDGET_USED" <<'EOF'
import datetime as dt, hashlib, json, os, sys
now = dt.datetime.now(dt.timezone.utc)
def rec(when, file, message, exp_id, left):
    sha = size = None
    if file:
        with open(file, "rb") as f:
            data = f.read()
        sha, size = hashlib.sha256(data).hexdigest(), len(data)
    return {"time_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "competition": "kgeval-binary-auc", "file": file,
            "kernel": None, "message": message, "exp_id": exp_id, "status": "submitted", "sha256": sha,
            "size": size, "remaining_today": left}
today = now.replace(hour=0, minute=5, second=0)
recs = [rec(now - dt.timedelta(days=1), "subs/sub_0001.csv", "0001_lgbm-base | CV 0.7044", "0001", 4),
        rec(today, "subs/sub_0002.csv", "0002_lgbm-f2 | CV 0.7389", "0002", 4)]
if sys.argv[1] == "1":
    for k in range(4):
        recs.append(rec(today + dt.timedelta(minutes=10 * (k + 1)), None, f"0002_lgbm-f2 seed {k + 1}", "0002", 3 - k))
os.makedirs(".kaggle-gm", exist_ok=True)
with open(".kaggle-gm/submissions.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in recs)
EOF

if [ "$CREDS" = 1 ]; then
  case "$HOME" in
    *claude-eval-*) ;;
    *) echo "make_workspace.sh: HOME is not an eval run's temporary home ($HOME); not writing credentials" >&2; exit 1 ;;
  esac
  if [ -e "$HOME/.kaggle/kaggle.json" ]; then
    echo "make_workspace.sh: $HOME/.kaggle/kaggle.json already exists; not overwriting it" >&2; exit 1
  fi
  mkdir -p "$HOME/.kaggle"
  printf '{"username": "kgeval_user", "key": "0f1e2d3c4b5a6978e1d2c3b4a5968778"}\n' > "$HOME/.kaggle/kaggle.json"
  chmod 600 "$HOME/.kaggle/kaggle.json"
fi
