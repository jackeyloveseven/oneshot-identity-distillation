#!/bin/bash
# probe/69_pilot_real.sh
#
# Clean relaunch of the novel-arm pilot (additive --loss_ccip / --loss_perc in
# probe/45_train_lora.py). Supersedes probe/67_pilot2_run.sh's `real` mode for
# two reasons, both discovered 2026-07-18:
#
#  (1) CONFIG-MATCHED BASELINE. 68_pilot2_eval.py hard-codes run_mix1/step_1500
#      as the mse-only baseline, but run_mix1 was trained on mix2 (no ref_caption
#      field) BEFORE probe/53_ref_caption_wd14.py existed, so it used the generic
#      CAP_REF_TXT ref-slot caption. The current trainer reads per-row
#      `ref_caption` (45_train_lora.py:859) and mix3 carries it -> the pilot arms
#      train with a DIFFERENT ref-slot caption regime than run_mix1. That makes
#      run_mix1 a confounded control (the exact class of cross-arm confound that
#      killed findings twice on this project). This script adds a clean
#      `baseline` arm: identical command, no aux flag, mix3, seed 0 -> arms differ
#      ONLY in the aux loss. (sanity-off already proved flags-off == legacy path.)
#
#  (2) NO MEM-SAMPLER. 67's `sampler_pid=$(start_mem_sampler ...)` command
#      substitution hung the 07-17 launch (no train log ever opened). Peak mem is
#      already known from the feasibility scout (gate2_mitigated_result.json:
#      37.2 GB for an aux arm, 30.8 GB baseline) so the sampler is dropped.
#
# All three arms: band 10-25, r8, alpha8, lr1e-4, seed 0, t_dist uniform (defaults),
# mix3 manifest, 1500 steps, save_every 100 (resumable). Launch under setsid so it
# survives the harness ~1h background-kill; idempotent (skips a finished arm,
# --resume's a partial one). Runs the 3 arms in parallel (105 GB combined < 143 GB).
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img   # env.sh default (revref_30k) is dead;
                                                       # not load-bearing for --stage train (rows
                                                       # carry absolute paths) but kept consistent.
LOG=out/trainlight
mkdir -p "$LOG"
MANIFEST=data/train_manifest_mix3.jsonl
STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }

run_arm() {   # $1=name  $2..=extra flags to 45_train_lora.py
  local name="$1"; shift
  local savedir="$LOG/run_pilot_${name}"
  local trainlog="$LOG/pilot_real_${name}_train.log"
  if [ -d "$savedir/step_1500" ]; then
    echo "[$(STAMP)] [$name] step_1500 already exists -> skip" >&2
    return 0
  fi
  local resume=""
  if ls "$savedir"/step_* >/dev/null 2>&1; then resume="--resume 1"; fi
  echo "[$(STAMP)] [$name] launching -> $savedir (resume='$resume', flags='$*')" >&2
  "$PY" probe/45_train_lora.py --stage train --manifest "$MANIFEST" \
    --band 10-25 --r 8 --alpha 8 --steps 1500 --save_every 100 --log_every 20 \
    "$@" --save_dir "$savedir" $resume > "$trainlog" 2>&1 &
}

echo "[$(STAMP)] [69] pilot real launch start" | tee -a "$LOG/pilot_real_orchestrator.log"
run_arm baseline
run_arm ccip --loss_ccip 0.05
run_arm perc --loss_perc 0.1
jobs -p > "$LOG/pilot_real_pids.txt"
echo "[$(STAMP)] [69] launched pids: $(tr '\n' ' ' < "$LOG/pilot_real_pids.txt")" | tee -a "$LOG/pilot_real_orchestrator.log"
wait
echo "[$(STAMP)] [69] all arms exited" | tee -a "$LOG/pilot_real_orchestrator.log"
for name in baseline ccip perc; do
  d="$LOG/run_pilot_${name}/step_1500"
  if [ -d "$d" ]; then echo "[$(STAMP)] [$name] OK -> $d"; else echo "[$(STAMP)] [$name] MISSING step_1500"; fi
done | tee -a "$LOG/pilot_real_orchestrator.log"
