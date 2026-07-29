#!/bin/bash
# Driver for the holdout30 cross-set replication of the band-position ablation.
# Mirrors probe/58_band_ablation_run.sh's invocation pattern exactly (same
# gen harness: probe/46_devprobe.py --stage gen --arm lora; same judge:
# probe/38_gemini_judge.py --set devprobe_lora), but:
#   - reads holdout30_tags.json instead of repprobe30_tags.json
#   - uses NEW checkpoint symlinks (ckptlink_ho30_band<band>_r<r>) so output
#     dirs are lora_ckptlink_ho30_band<band>_r<r> -- a namespace disjoint
#     from the published rep30 ablation's lora_ckptlink_band<band>_r<r> dirs
#   - regenerates mid too (checkpoint run_mix1/step_4000), not reusing any
#     old mid holdout30 images
#   - NO TRAINING: all four checkpoints already exist, this script only
#     calls --stage gen (never --stage train)
#
# All four checkpoints already verified to exist with pytorch_lora_weights
# .safetensors present. Symlinks already created by the calling agent before
# this script runs.
set -uo pipefail

cd "$(dirname "$0")"
cd /home/ubuntu/chengyanli/items/cherry
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img   # same override probe/58 uses (env.sh default REF is dead)

SUMMARY=data/ho30_band_replication_summary.txt
mkdir -p out/trainlight
# APPEND (not truncate): the driver is resumable. A killed run's completed
# arms are skipped via the judge-JSON existence check in run_arm, partially
# generated arms resume via 46_devprobe's own existing-image skip, and the
# summary keeps full history across restarts.
echo "=== ho30 band replication driver (re)start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$SUMMARY"
free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU free at driver start=${free0}MiB" | tee -a "$SUMMARY"

run_arm() {
  local label="$1" band="$2" r="$3" link="$4"
  local genlog="out/trainlight/ho30_${label}_gen.log"
  local judgelog="out/trainlight/ho30_${label}_judge.log"
  local lora_dir="out/devprobe/lora_$(basename "$link")"
  local outjson="data/gemini_judge_ho30_band_${band}_r${r}_step4000.json"

  echo "=== arm ${label} (band=${band} r=${r}) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"

  if [ -s "$outjson" ]; then
    echo "[skip] arm=${label} already complete ($outjson exists) -- not re-judging" | tee -a "$SUMMARY"
    return 0
  fi

  if [ ! -e "$link/pytorch_lora_weights.safetensors" ]; then
    echo "[gen] FAILED: checkpoint symlink $link missing weights file, skipping arm" | tee -a "$SUMMARY"
    return 1
  fi

  echo "[gen] arm=${label} adapter=$link tags=holdout30_tags.json (log: $genlog)" | tee -a "$SUMMARY"
  DEVPROBE_TAGS=holdout30_tags.json "$PY" probe/46_devprobe.py --stage gen --arm lora \
    --adapter "$link" > "$genlog" 2>&1
  local grc=$?
  if [ $grc -ne 0 ]; then
    echo "[gen] FAILED rc=$grc for arm=${label}, see $genlog -- skipping judge" | tee -a "$SUMMARY"
    return 1
  fi
  local n_imgs=$(ls "$lora_dir"/*.png 2>/dev/null | wc -l)
  echo "[gen] done arm=${label}: ${n_imgs} images in ${lora_dir} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
  if [ "$n_imgs" -ne 60 ]; then
    echo "[gen] WARNING: expected 60 images (30 subjects x 2 seeds), found ${n_imgs}" | tee -a "$SUMMARY"
  fi

  echo "[judge] arm=${label} lora_dir=${lora_dir} (log: $judgelog)" | tee -a "$SUMMARY"
  DEVPROBE_TAGS=holdout30_tags.json \
    DEVPROBE_LORA_DIR="$lora_dir" \
    "$PY" probe/38_gemini_judge.py --set devprobe_lora > "$judgelog" 2>&1
  local jrc=$?
  if [ $jrc -ne 0 ]; then
    echo "[judge] FAILED rc=$jrc for arm=${label}, see $judgelog" | tee -a "$SUMMARY"
    return 1
  fi
  mv data/gemini_judge_devprobe_lora.json "$outjson"
  echo "[judge] done -> $outjson $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
  return 0
}

run_arm "early_0-15_r8"  "0-15"  8 "out/trainlight/run_band_0-15_r8/ckptlink_ho30_band0-15_r8"
run_arm "mid_10-25_r8"   "10-25" 8 "out/trainlight/run_mix1/ckptlink_ho30_band10-25_r8"
run_arm "late_14-29_r8"  "14-29" 8 "out/trainlight/run_band_14-29_r8/ckptlink_ho30_band14-29_r8"
run_arm "fulldepth_0-29_r4" "0-29" 4 "out/trainlight/run_band_0-29_r4/ckptlink_ho30_band0-29_r4"

echo "=== all arms attempted, done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"
