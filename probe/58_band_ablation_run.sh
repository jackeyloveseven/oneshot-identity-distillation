#!/bin/bash
# probe/58_band_ablation_run.sh
#
# Sequential band-position-ablation runner for the training-light band-LoRA
# keystone. Trains the three missing arms (early 0-15 r8, late 14-29 r8,
# full-depth 0-29 r4) to step 4000 on train_manifest_mix3.jsonl, then
# generates + Gemini-judges each on the fixed repprobe30_tags.json dev set,
# one arm fully at a time (shared GPU, no parallel training).
#
# CORRECTION (2026-07-16): the mid 10-25 r8 reference arm actually used was
# run_mix1/step_4000 (proven by adapter= lines in devprobe/base_rep30.log),
# NOT run_mix2 as this header originally claimed. Worse, the original
# "byte-identical on the fields 45 actually reads" claim is FALSE:
# run_mix1 trained on train_manifest_mix2.jsonl, which lacks the
# ref_caption field entirely (0/5936 rows vs mix3's 5936/5936), and
# 45_train_lora.py falls back to the generic constant ref caption -- so the
# mid arm trained under a different ref-caption regime than the three arms
# it is compared against. Additionally, the published mid rep30/holdout30
# generations carried the "In an anime_2d style, " prompt prefix while the
# other three arms used raw prompt_B. Both confounds are disclosed in the
# paper; a confound-free mid (run_mix2/step_4000, mix3-trained) exists and
# was evaluated separately on 2026-07-16 (see data/cleanmid_summary.json).
#
# alpha is pinned equal to r for every arm (alpha/r=1.0 scaling, matching the
# band=10-25 r=8 alpha=8 convention already used by run_mix1/run_mix2) --
# full-depth uses r=4 (not the r=8 the 16-layer arms use) specifically to
# keep total trainable params close to the ~3.93M reference (30 layers * r4
# = 3,686,400, ~93.8%; r5 would overshoot to ~117%).
#
# Checkpoint dirs are always named .../step_4000 by 45_train_lora.py
# regardless of --save_dir, and both 45's --stage sample tag and 46's
# --arm lora output dir derive their uniqueness from
# os.path.basename(os.path.normpath(--adapter)) alone -- so passing three
# different arms' raw step_4000 dirs as --adapter would silently collide
# and overwrite each other's generated images / judge output. Fix: symlink
# each arm's checkpoint to a name that embeds the band before generating.
#
# Resilient to unattended overnight execution: set -e is NOT used at the
# top level -- each arm's train/gen/judge failure is caught, logged to
# data/band_ablation_summary.txt, and the script moves on to the next arm
# rather than aborting the whole run.
set -uo pipefail

cd "$(dirname "$0")/.."   # -> cherry/
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img   # env.sh's default REF (revref_30k) is dead; see probe/56 docstring

LOG_DIR=/mnt/local/cherry_out/trainlight
SUMMARY=data/band_ablation_summary.txt
mkdir -p "$LOG_DIR"
echo "=== band ablation orchestrator start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$SUMMARY"

wait_for_gpu() {
  local need_mib=42000
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$free" -ge "$need_mib" ]; then
      echo "[$ts] GPU free=${free}MiB >= ${need_mib}MiB, proceeding" | tee -a "$SUMMARY"
      return 0
    fi
    echo "[$ts] GPU free=${free}MiB < ${need_mib}MiB, waiting 60s..." >> "$LOG_DIR/band_ablation_orchestrator.log"
    sleep 60
  done
}

run_arm() {
  local band="$1" r="$2"
  local savedir="$LOG_DIR/run_band_${band}_r${r}"
  local ckpt="$savedir/step_4000"
  local link="$savedir/ckptlink_band${band}_r${r}"
  local trainlog="$LOG_DIR/band_${band}_r${r}_train.log"
  local genlog="$LOG_DIR/band_${band}_r${r}_gen.log"
  local judgelog="$LOG_DIR/band_${band}_r${r}_judge.log"
  local outjson="data/gemini_judge_band_${band}_r${r}_step4000.json"

  echo "=== arm band=${band} r=${r} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"

  if [ -d "$ckpt" ]; then
    echo "[train] checkpoint already exists at $ckpt, skipping training" | tee -a "$SUMMARY"
  else
    wait_for_gpu
    echo "[train] band=${band} r=${r} alpha=${r} -> $savedir (log: $trainlog)" | tee -a "$SUMMARY"
    "$PY" probe/45_train_lora.py --stage train \
      --manifest data/train_manifest_mix3.jsonl \
      --band "$band" --r "$r" --alpha "$r" \
      --steps 4000 --save_every 250 \
      --save_dir "$savedir" > "$trainlog" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -d "$ckpt" ]; then
      echo "[train] FAILED rc=$rc or checkpoint missing for band=${band} r=${r}, see $trainlog -- skipping rest of this arm" | tee -a "$SUMMARY"
      return 1
    fi
    echo "[train] done band=${band} r=${r} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
  fi

  ln -sfn "$(readlink -f "$ckpt")" "$link"

  echo "[gen] band=${band} r=${r} (log: $genlog)" | tee -a "$SUMMARY"
  DEVPROBE_TAGS=repprobe30_tags.json "$PY" probe/46_devprobe.py --stage gen --arm lora \
    --adapter "$link" > "$genlog" 2>&1
  local grc=$?
  if [ $grc -ne 0 ]; then
    echo "[gen] FAILED rc=$grc for band=${band} r=${r}, see $genlog -- skipping judge" | tee -a "$SUMMARY"
    return 1
  fi

  echo "[judge] band=${band} r=${r} (log: $judgelog)" | tee -a "$SUMMARY"
  DEVPROBE_TAGS=repprobe30_tags.json \
    DEVPROBE_LORA_DIR="/mnt/local/cherry_out/devprobe/lora_ckptlink_band${band}_r${r}" \
    "$PY" probe/38_gemini_judge.py --set devprobe_lora > "$judgelog" 2>&1
  local jrc=$?
  if [ $jrc -ne 0 ]; then
    echo "[judge] FAILED rc=$jrc for band=${band} r=${r}, see $judgelog" | tee -a "$SUMMARY"
    return 1
  fi
  mv data/gemini_judge_devprobe_lora.json "$outjson"
  echo "[judge] done -> $outjson $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
  return 0
}

run_arm "0-15" 8
run_arm "14-29" 8
run_arm "0-29" 4

echo "=== all arms attempted, done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== final stats (Wilson 95% CI) ===" | tee -a "$SUMMARY"
"$PY" probe/59_band_ablation_report.py >> "$SUMMARY" 2>&1
echo "report written, see $SUMMARY"
