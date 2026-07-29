#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v hf >/dev/null || {
  echo "Missing 'hf'. Install requirements.txt first." >&2
  exit 1
}

mkdir -p models/lora
hf download krea/Krea-2-Turbo --local-dir models/Krea-2-Turbo
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct
hf download conradlocke/krea2-identity-edit krea2_identity_edit_v1_1.safetensors \
  --local-dir models/lora
hf download conradlocke/krea2-identity-edit krea2_identity_edit_v1_2.safetensors \
  --local-dir models/lora

echo "Core Krea 2, processor, and identity LoRA weights are ready under models/."
