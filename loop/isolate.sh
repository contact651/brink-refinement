#!/usr/bin/env bash
# Weekly loop, step 2 of 3: the isolated practice step.
# Copies ONLY the practice inputs (to Dec 2014), the reviewed runner and the queue into a fresh folder with no .git,
# deletes the checkout, the downloaded data and every prep-time copy, then runs the canary and the runner with no
# network (own network namespace), as the normal runner user, with an emptied environment.
set -euo pipefail
WS="$GITHUB_WORKSPACE"; ISO="$RUNNER_TEMP/iso"; INP="$RUNNER_TEMP/practice_inputs"
PY="$(command -v python)"; PYDIR="$(dirname "$PY")"; ME="$(id -un)"
rm -rf "$ISO"; mkdir -p "$ISO/cycle" "$ISO/loop" "$ISO/queue" "$ISO/inputs" "$ISO/data/clean" "$ISO/out"
cp "$WS/src/brink_claude.py" "$WS/src/brink_grok.py" "$WS/src/brink_features.py" "$ISO/"
cp "$WS/src/cycle/practice.py" "$WS/src/cycle/misses.py" "$ISO/cycle/"
cp "$WS/src/loop/runner.py" "$WS/src/loop/canary.py" "$WS/src/loop/working_model.json" "$ISO/loop/"
cp "$WS"/src/queue/*.json "$ISO/queue/" 2>/dev/null || true
cp "$RUNNER_TEMP/done_ids.json" "$ISO/done_ids.json"
cp "$INP/panel.pkl" "$INP/manifest.json" "$ISO/inputs/"
cp "$INP"/clean/*.parquet "$ISO/data/clean/"
rm -rf "$WS/src" "$WS/ledger_ro" "$INP" "$HOME/.cache/pip"
sudo unshare --net -- runuser -u "$ME" -- env -i HOME="$ISO" PATH="$PYDIR:/usr/bin:/bin" LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  bash -c "cd '$ISO' && '$PY' loop/canary.py --ws '$WS' && '$PY' loop/runner.py --out out/result.json"
cp "$ISO/out/result.json" "$RUNNER_TEMP/result.json"
rm -rf "$ISO"
