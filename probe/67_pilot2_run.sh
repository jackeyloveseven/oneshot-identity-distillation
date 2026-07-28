#!/bin/bash
# probe/67_pilot2_run.sh
#
# PILOT (pre-registered smoke gate, NOT final evidence) orchestrator for the
# two additive loss upgrades added to probe/45_train_lora.py (--loss_perc,
# --loss_ccip). Mirrors probe/58_band_ablation_run.sh's wait_for_gpu +
# resilient-unattended-execution conventions.
#
# Usage:
#   probe/67_pilot2_run.sh sanity             # 3 steps, NO aux flags, scratch
#                                              # save_dir -- cheap proof the
#                                              # default (flags-off) path is
#                                              # unaffected by the new code
#   probe/67_pilot2_run.sh smoke ccip 0.05    # throwaway ~15-step validation run
#   probe/67_pilot2_run.sh smoke perc 0.1     # (scratch save_dir, NOT the real pilot arm)
#   probe/67_pilot2_run.sh real  ccip 0.05    # the real 1500-step pilot arm
#   probe/67_pilot2_run.sh real  perc 0.1     # (idempotent: --resume auto-attached
#                                              #  if a checkpoint already exists)
#
# "smoke"/"sanity" modes exist to validate the new code end-to-end AND read
# raw component-loss magnitudes + actual nvidia-smi peak memory BEFORE
# committing ~2.7h of GPU time to a "real" arm -- settles the loss-weight-
# ratio and parallel-vs-serial questions up front rather than relying on the
# mid-run adjustment allowance as the primary mechanism.
#
# GPU is shared with other (same-user, different-project) jobs on this
# machine; wait_for_gpu() polls before every launch, same threshold
# discipline as probe/58 (42000 MiB was that script's own empirically-chosen
# single-arm safety margin for a similar-memory bs=4 workload).
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export REF=/mnt/local/smoke_anime/anime_260623/img   # env.sh's default REF (revref_30k) is dead
                                                       # (does not exist); see probe/58's own header
                                                       # and probe/56's docstring. Not load-bearing
                                                       # for --stage train (manifest rows carry
                                                       # absolute paths) but kept for consistency /
                                                       # in case this script is ever extended.

LOG_DIR=/mnt/local/cherry_out/trainlight
SUMMARY=data/pilot2_summary.txt
mkdir -p "$LOG_DIR"

MODE="${1:?usage: $0 sanity | smoke ccip|perc weight | real ccip|perc weight}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUMMARY"; }

wait_for_gpu() {
  local need_mib=42000
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$free" -ge "$need_mib" ]; then
      echo "[$ts] GPU free=${free}MiB >= ${need_mib}MiB, proceeding" | tee -a "$SUMMARY"
      return 0
    fi
    echo "[$ts] GPU free=${free}MiB < ${need_mib}MiB, waiting 60s... (mode=$MODE)" >> "$LOG_DIR/pilot2_orchestrator.log"
    sleep 60
  done
}

# background GPU memory sampler: real nvidia-smi memory.used (reserved+
# context, all processes) sampled every 2s while a training subprocess runs
# -- a more representative datum for the parallel-vs-serial decision than
# torch's own allocated-only stat. start_mem_sampler <outfile>; stop with
# stop_mem_sampler <outfile> (prints the observed peak).
start_mem_sampler() {
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > "$1" ) &
  echo $!
}
stop_mem_sampler() {
  local pid="$1" outfile="$2"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  if [ -s "$outfile" ]; then
    sort -n "$outfile" | tail -1
  else
    echo "unknown"
  fi
}

if [ "$MODE" = "sanity" ]; then
  savedir="$LOG_DIR/_pilot2_sanity_off"
  trainlog="$LOG_DIR/pilot2_sanity_off_train.log"
  marker="$LOG_DIR/_pilot2_sanity_off_done"
  if [ -f "$marker" ]; then
    log "[sanity:off] already done (marker $marker exists), skipping. Delete marker to re-run."
    exit 0
  fi
  log "=== sanity off (NO aux flags, 3 steps) start ==="
  wait_for_gpu
  rm -rf "$savedir"
  "$PY" probe/45_train_lora.py --stage train \
    --manifest data/train_manifest_mix3.jsonl \
    --band 10-25 --r 8 --alpha 8 \
    --steps 3 --save_every 1000 --log_every 1 \
    --save_dir "$savedir" > "$trainlog" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "[sanity:off] FAILED rc=$rc, see $trainlog"
    exit 1
  fi
  touch "$marker"
  log "[sanity:off] done, log at $trainlog, train_log.jsonl at $savedir/train_log.jsonl"
  exit 0
fi

ARM="${2:?usage: $0 sanity | smoke ccip|perc weight | real ccip|perc weight}"
WEIGHT="${3:?usage: $0 sanity | smoke ccip|perc weight | real ccip|perc weight}"

if [ "$ARM" = "ccip" ]; then
  FLAG="--loss_ccip"
elif [ "$ARM" = "perc" ]; then
  FLAG="--loss_perc"
else
  echo "unknown arm: $ARM (expected ccip|perc)"; exit 1
fi

if [ "$MODE" = "smoke" ]; then
  savedir="$LOG_DIR/_pilot2_smoke_${ARM}"
  trainlog="$LOG_DIR/pilot2_smoke_${ARM}_train.log"
  marker="$LOG_DIR/_pilot2_smoke_${ARM}_done"
  memsample="$LOG_DIR/pilot2_smoke_${ARM}_gpumem.csv"
  if [ -f "$marker" ]; then
    log "[smoke:$ARM] already done (marker $marker exists), skipping. Delete marker to re-run."
    exit 0
  fi
  log "=== smoke $ARM ($FLAG=$WEIGHT) start ==="
  wait_for_gpu
  rm -rf "$savedir"
  : > "$memsample"
  sampler_pid=$(start_mem_sampler "$memsample")
  "$PY" probe/45_train_lora.py --stage train \
    --manifest data/train_manifest_mix3.jsonl \
    --band 10-25 --r 8 --alpha 8 \
    --steps 15 --save_every 1000 --log_every 1 \
    "$FLAG" "$WEIGHT" \
    --save_dir "$savedir" > "$trainlog" 2>&1
  rc=$?
  peak_mib=$(stop_mem_sampler "$sampler_pid" "$memsample")
  log "[smoke:$ARM] peak nvidia-smi memory.used during run: ${peak_mib} MiB"
  if [ $rc -ne 0 ]; then
    log "[smoke:$ARM] FAILED rc=$rc, see $trainlog"
    exit 1
  fi
  touch "$marker"
  log "[smoke:$ARM] done, log at $trainlog, train_log.jsonl at $savedir/train_log.jsonl, peak_mem=${peak_mib}MiB"
  exit 0

elif [ "$MODE" = "real" ]; then
  savedir="$LOG_DIR/run_pilot_${ARM}"
  trainlog="$LOG_DIR/pilot2_real_${ARM}_train.log"
  memsample="$LOG_DIR/pilot2_real_${ARM}_gpumem.csv"
  ckpt="$savedir/step_1500"
  if [ -d "$ckpt" ]; then
    log "[real:$ARM] checkpoint already exists at $ckpt, skipping (already done)"
    exit 0
  fi
  resume_flag=""
  if [ -d "$savedir" ] && ls "$savedir"/step_* >/dev/null 2>&1; then
    resume_flag="--resume 1"
    log "[real:$ARM] existing partial checkpoints found under $savedir, will --resume"
  fi
  log "=== real $ARM ($FLAG=$WEIGHT) start -> $savedir (log: $trainlog) ==="
  wait_for_gpu
  : >> "$memsample"
  sampler_pid=$(start_mem_sampler "$memsample")
  "$PY" probe/45_train_lora.py --stage train \
    --manifest data/train_manifest_mix3.jsonl \
    --band 10-25 --r 8 --alpha 8 \
    --steps 1500 --save_every 100 \
    "$FLAG" "$WEIGHT" \
    --save_dir "$savedir" $resume_flag > "$trainlog" 2>&1
  rc=$?
  peak_mib=$(stop_mem_sampler "$sampler_pid" "$memsample")
  log "[real:$ARM] peak nvidia-smi memory.used during this invocation: ${peak_mib} MiB"
  if [ $rc -ne 0 ] || [ ! -d "$ckpt" ]; then
    log "[real:$ARM] FAILED rc=$rc or checkpoint missing, see $trainlog"
    exit 1
  fi
  log "[real:$ARM] done -> $ckpt"
  exit 0

else
  echo "unknown mode: $MODE (expected sanity|smoke|real)"; exit 1
fi
