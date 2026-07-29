#!/bin/bash
# probe/70_pilot_watch_eval.sh
#
# Autonomous tail of the novel-arm pilot pipeline. Polls until all three arms
# launched by probe/69_pilot_real.sh (baseline/ccip/perc) have written a
# step_1500 checkpoint, then runs the gate eval (probe/68_pilot2_eval.py) exactly
# once -> data/pilot2_summary.json. Launched under setsid alongside 69 so the
# whole pilot->gate sequence finishes unattended (arms ~6h; the harness kills
# tracked background at ~1h, so detachment is mandatory). Idempotent via a marker.
#
# NOTE on interpretation (advisor, 2026-07-18): the eval reports CCIP-torch
# cosine for every arm, but the ccip arm was TRAINED on that exact metric -- its
# CCIP rise is Goodhart, NOT evidence. Read the ccip arm on Gemini same_character
# + defect only. perc arm's job is defect-down (the real gap vs krea2). This
# script only PRODUCES the numbers; the by-axis reading happens on review.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img
LOG=out/trainlight
WATCHLOG=$LOG/pilot_watch_eval.log
MARKER=$LOG/_pilot_gate_eval_done
STAMP(){ date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ -f "$MARKER" ]; then
  echo "[$(STAMP)] gate eval already done (marker exists) -> exit" >> "$WATCHLOG"; exit 0
fi
echo "[$(STAMP)] watcher start; polling for 3x step_1500" >> "$WATCHLOG"
while true; do
  ok=1
  for name in baseline ccip perc; do
    [ -d "$LOG/run_pilot_${name}/step_1500" ] || ok=0
  done
  [ "$ok" = "1" ] && break
  sleep 120
done
echo "[$(STAMP)] all 3 step_1500 present; settling 60s then running gate eval" >> "$WATCHLOG"
sleep 60
"$PY" probe/68_pilot2_eval.py >> "$WATCHLOG" 2>&1
rc=$?
echo "[$(STAMP)] gate eval exited rc=$rc; report at data/pilot2_summary.json" >> "$WATCHLOG"
[ $rc -eq 0 ] && touch "$MARKER"
