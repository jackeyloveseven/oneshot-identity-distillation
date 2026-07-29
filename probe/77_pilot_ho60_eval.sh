#!/bin/bash
# probe/77_pilot_ho60_eval.sh
#
# Bounded, non-speculative re-eval (advisor-endorsed 2026-07-20): the three
# step_1500 pilot arms (baseline=recon / ccip / perc) were only judged at n=24
# (devprobe12 x 2 seeds) -- CIs huge, all overlapping, and the point estimates
# trend the WRONG way (ccip/perc worse than baseline). Before betting hours on
# extend-to-convergence, get a less-noisy read: re-judge the SAME frozen
# step_1500 checkpoints on the full holdout30 (30 x 2 = 60/arm). ZERO new
# training, zero new confound surface.
#
# Discipline: fresh ho60_* link namespaces so the existing n=24 gens/judges are
# NEVER touched (namespace bleed is the exact class of confound that killed
# findings twice here). Idempotent: skips finished gens/judges.
#
# Decision rule (locked): if ccip/perc clean-yes stays <= baseline at n=60,
# identity-metric supervision is NOT the fix at this recipe -> the user's
# "was it the objective?" is answered honestly, no convergence run. Escalate to
# convergence ONLY if a supervised arm >= baseline (at least a tie) at n=60.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img
export DEVPROBE_TAGS=holdout30_tags.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TL=out/trainlight
DP=out/devprobe
LOG=$TL/ho60_eval; mkdir -p "$LOG"
ARMS=(baseline ccip perc)

# 1. fresh ho60_ ckptlinks -> the frozen step_1500 checkpoints
for arm in "${ARMS[@]}"; do
  ln -sfn "$TL/run_pilot_${arm}/step_1500" "$TL/ckptlink_ho60_pilot_${arm}"
done

# 2. gen holdout30 x 2 seeds per arm, ALL THREE IN PARALLEL (free H200, ~30GB/arm)
pids=()
for arm in "${ARMS[@]}"; do
  link="$TL/ckptlink_ho60_pilot_${arm}"
  out="$DP/lora_ckptlink_ho60_pilot_${arm}"
  nhave=0; [ -d "$out" ] && nhave=$(ls "$out"/*.png 2>/dev/null | wc -l)
  if [ "$nhave" -ge 60 ]; then echo "[gen $arm] $nhave/60 present -> skip"; continue; fi
  echo "[gen $arm] launching -> $out"
  "$PY" probe/46_devprobe.py --stage gen --arm lora --adapter "$link" \
      > "$LOG/gen_${arm}.log" 2>&1 &
  pids+=($!)
done
echo "[gen] waiting on ${#pids[@]} parallel arms..."
for p in "${pids[@]}"; do wait "$p"; done
echo "[gen] all arms done"
for arm in "${ARMS[@]}"; do
  out="$DP/lora_ckptlink_ho60_pilot_${arm}"
  echo "[gen $arm] $(ls "$out"/*.png 2>/dev/null | wc -l)/60 images"
done

# 3. judge each arm on the full n=60 (defect-aware, same instrument as band-ablation)
for arm in "${ARMS[@]}"; do
  dest="data/gemini_judge_ho60_lora_pilot_${arm}.json"
  if [ -f "$dest" ]; then echo "[judge $arm] $dest exists -> skip"; continue; fi
  echo "[judge $arm] judging n=60 ..."
  DEVPROBE_LORA_DIR="$DP/lora_ckptlink_ho60_pilot_${arm}" \
    "$PY" probe/38_gemini_judge.py --set devprobe_lora > "$LOG/judge_${arm}.log" 2>&1
  # 38 writes data/gemini_judge_devprobe_lora.json -> rename to ho60 dest
  if [ -f data/gemini_judge_devprobe_lora.json ]; then
    mv data/gemini_judge_devprobe_lora.json "$dest"
    echo "[judge $arm] -> $dest"
  else
    echo "[judge $arm] WARN: expected judge output not found (see $LOG/judge_${arm}.log)"
  fi
done
echo "[ho60] ALL DONE"
