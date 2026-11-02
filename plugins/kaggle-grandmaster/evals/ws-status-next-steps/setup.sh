#!/usr/bin/env bash
# Seeds the run's workspace with the shared kgkit fixture (see ../_fixtures/make_workspace.sh).
set -euo pipefail
bash "$(dirname "$0")/../_fixtures/make_workspace.sh"
