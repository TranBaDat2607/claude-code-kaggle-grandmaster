#!/bin/sh
# Launch a kaggle-grandmaster hook script with a working Python 3.
# - Tool hooks (pre/post) bail out before starting Python unless the command mentions kaggle,
#   and post_write.py unless the written file is a .py file,
#   so ordinary shell commands pay no measurable cost. A wrapper script that submits (`bash go.sh x.csv`) shows
#   neither in its command, so a payload that mentions a submit, or Kaggle's "N submissions remaining today" reply,
#   passes too. post_tool.py also starts for a command that writes a .log/.out file (to say logs go in work/).
# - On Windows `python3` may be the Microsoft Store stub, so every candidate is test-run.
# - Hooks fail open: any problem here exits 0 and never blocks the user.
DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$1"
INPUT="$(cat)"

case "$SCRIPT" in
  pre_tool.py)
    case "$INPUT" in
      *kaggle*|*KAGGLE*|*Kaggle*|*submit*|*Submit*|*SUBMIT*|*"remaining today"*) ;;
      *) exit 0 ;;
    esac
    ;;
  post_tool.py)
    case "$INPUT" in
      *kaggle*|*KAGGLE*|*Kaggle*|*submit*|*Submit*|*SUBMIT*|*"remaining today"*) ;;
      *'.log '*|*'.log"'*|*'.log;'*|*'.log&'*|*'.log|'*|*'.out '*|*'.out"'*|*'.out;'*|*'.out&'*|*'.out|'*) ;;
      *) exit 0 ;;
    esac
    ;;
  post_write.py)
    # only a written Python file can be model code
    case "$INPUT" in
      *'.py"'*|*'.PY"'*) ;;
      *) exit 0 ;;
    esac
    ;;
esac

for py in python3 python py; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1; then
    printf '%s' "$INPUT" | "$py" "$DIR/$SCRIPT"
    exit $?
  fi
done
exit 0
