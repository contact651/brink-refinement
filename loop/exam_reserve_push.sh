#!/usr/bin/env bash
# Exam wall: reserve one exam check and make it durable BEFORE any exam outcome is built.
# Usage: exam_reserve_push.sh LEDGER_DIR PRIOR_JSON EXPERIMENT_ID QUEUE_DIR COMMIT [NOW_UTC]
# Exit 0 = reserved and pushed. 3 = refused (the refusal line is pushed). 2 = ledger push failed (nothing reserved).
# Any other non-zero = stopped before writing. Used by .github/workflows/exam.yml and by the fake-exam self-test.
set -uo pipefail
L="$1"; HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/exam_reserve.py" reserve --ledger "$L" --prior "$2" --experiment "$3" --queue "$4" --commit "$5" ${6:+--now "$6"}
code=$?
if ! git -C "$L" diff --quiet -- exam_ledger.jsonl; then
  git -C "$L" -c user.name=brink-exam -c user.email=41898282+github-actions[bot]@users.noreply.github.com \
    commit -q -m "Exam ledger, run ${GITHUB_RUN_ID:-local}" -- exam_ledger.jsonl || exit 1
  git -C "$L" push -q origin HEAD:ledger || { echo "STOP: the ledger push failed; nothing is reserved and nothing will be built"; exit 2; }
fi
exit $code
