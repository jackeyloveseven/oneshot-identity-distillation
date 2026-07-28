#!/usr/bin/env bash
set -euo pipefail

KREA_PYTHON=/mnt/local/krea2_edit_serve/venv/bin/python
REPORT_PYTHON=/home/ubuntu/miniconda3/bin/python
PROBE=/home/ubuntu/chengyanli/items/cherry/probe/115_krea2_identity_tsd.py
IMAGE_ROOT=/mnt/local/smoke_anime/anime_260623/img
COMMON=(
  --lr 0.05
  --pseudo-step-size 1.0
  --trust-region-rms 0.06
  --local-trust-region-rms 0.05
  --adaptive-trust-ratio 1.5
  --normalize rms_token
  --residual-mode raw
  --active-end 1.0
)

run_identity() {
  local uid=$1
  local reference=$2
  local negative_reference=$3
  local scene_prompt=$4
  local teacher_prompt=$5
  local seed
  for seed in 777 778 779; do
    CUDA_VISIBLE_DEVICES=0 "$KREA_PYTHON" "$PROBE" \
      --uid "$uid" \
      --ref "$reference" \
      --negative-ref "$negative_reference" \
      --seed "$seed" \
      --scene-prompt "$scene_prompt" \
      --teacher-prompt "$teacher_prompt" \
      "${COMMON[@]}"
  done
}

run_identity \
  u4229 \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  "$IMAGE_ROOT/u1839_whiteref.png" \
  "single anime man with a sharp jaw, light-colored spiky hair with two long strands framing his face, narrow hooded eyes, large black hoop earrings, and a buckled neck choker, standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, coherent anatomy, face clearly visible, no other people" \
  "Preserve the exact identity, face, spiky light hair with two long front strands, narrow hooded eyes, black hoop earrings, sharp jaw, and neck choker of the reference character. Create a single coherent anime character standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, face clearly visible, no other people, no collage, no split panel."

run_identity \
  u1839 \
  "$IMAGE_ROOT/u1839_whiteref.png" \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  "single anime woman with a heart-shaped face, messy shoulder-length layered dark hair, long fringe, grey eyes, silver geometric earrings, and a thick grey cable-knit turtleneck sweater, standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, coherent anatomy, face clearly visible, no other people" \
  "Preserve the exact identity, face, messy layered dark hair, long fringe, grey eyes, heart-shaped face, thick cable-knit turtleneck sweater, and silver geometric earrings of the reference character. Create a single coherent anime woman standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, face clearly visible, no other people, no collage, no split panel."

run_identity \
  u9728 \
  "$IMAGE_ROOT/u9728_whiteref.png" \
  "$IMAGE_ROOT/u4229_whiteref.png" \
  "single anime woman with messy shoulder-length dark hair, glowing purple eyes, a black peaked newsboy cap with small pointed devil horns, a black buttoned long-sleeve shirt, and a striped neck-bow with a central purple gemstone, standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, coherent anatomy, face clearly visible, no other people" \
  "Preserve the exact identity, face, messy shoulder-length dark hair, glowing purple eyes, black peaked cap with small pointed devil horns, dark buttoned shirt, striped neck-bow, and central purple gemstone of the reference character. Create a single coherent anime woman standing on a rain-soaked neon city street at night, three-quarter body, holding a transparent umbrella, cinematic reflections, face clearly visible, no other people, no collage, no split panel."

"$REPORT_PYTHON" /home/ubuntu/chengyanli/items/cherry/probe/116_krea2_publication_report.py
