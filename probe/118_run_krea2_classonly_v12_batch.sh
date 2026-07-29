#!/usr/bin/env bash
set -euo pipefail

KREA_PYTHON=/mnt/local/krea2_edit_serve/venv/bin/python
PROBE=/home/ubuntu/chengyanli/items/cherry/probe/115_krea2_identity_tsd.py
IMAGE_ROOT=/mnt/local/smoke_anime/anime_260623/img
OUT_DIR=/home/ubuntu/chengyanli/items/cherry/out/krea2_tsd_neutral
LORA=models/lora/krea2_identity_edit_v1_2.safetensors

COMMON=(
  --lora-path "$LORA"
  --ref-boost 4
  --out-dir "$OUT_DIR"
  --steps 12
  --inner-iters 3
  --lr 0.05
  --pseudo-step-size 1.0
  --trust-region-rms 0.10
  --local-trust-region-rms 0.085
  --adaptive-trust-ratio 3.0
  --normalize rms_token
  --residual-mode raw
  --active-end 1.0
)

run_identity() {
  local uid=$1
  local reference=$2
  local negative_reference=$3
  local width=$4
  local height=$5
  local class_word=$6
  local possessive=$7
  local object_pronoun=$8
  local seed
  local metadata

  for seed in 777 778 779; do
    metadata="$OUT_DIR/${uid}_classonly_v12_ar_s${seed}_${width}x${height}_i3_p1_g0.1_l0.085_a3_rb4_rms_token_raw_roifull_o1.json"
    if [[ -f "$metadata" ]]; then
      echo "[skip] $metadata"
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 "$KREA_PYTHON" "$PROBE" \
      --uid "${uid}_classonly_v12_ar" \
      --ref "$reference" \
      --negative-ref "$negative_reference" \
      --seed "$seed" \
      --width "$width" \
      --height "$height" \
      --scene-prompt "single adult anime ${class_word} standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, coherent anatomy, face clearly visible, no other people" \
      --teacher-prompt "Create a high-quality anime illustration of the exact adult ${class_word} shown in the reference image as the only subject. Infer every identity trait only from the reference image. Preserve ${possessive} identity without inventing a different person. Place ${object_pronoun} on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, coherent anatomy, face clearly visible, no other people, no collage, no split panel." \
      "${COMMON[@]}"
  done
}

run_identity \
  u4229 \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  "$IMAGE_ROOT/u1839_whiteref.png" \
  512 896 man his him

run_identity \
  u1839 \
  "$IMAGE_ROOT/u1839_whiteref.png" \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  512 768 woman her her

run_identity \
  u9728 \
  "$IMAGE_ROOT/u9728_whiteref.png" \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  640 640 woman her her
