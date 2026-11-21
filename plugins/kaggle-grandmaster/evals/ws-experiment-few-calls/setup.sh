#!/usr/bin/env bash
# Seeds the run's workspace with the shared kgkit fixture (see ../_fixtures/make_workspace.sh): 0003 is logged and
# compared, and backlog idea #4 (the f3 feature it tests) is running.
set -euo pipefail
bash "$(dirname "$0")/../_fixtures/make_workspace.sh" --candidate --idea
