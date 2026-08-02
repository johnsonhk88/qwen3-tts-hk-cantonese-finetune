#!/usr/bin/env bash
set -euo pipefail

# Reproducible wrapper for the successful full 1-epoch LoRA run.
# Run from anywhere; the script switches into this directory first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/media/johnsonhk88/Big-Data-Disk/venv-qwen3-tts/bin/python}"
INIT_MODEL_PATH="./Qwen3-TTS-12Hz-0.6B-Base"
TRAIN_JSONL="./train_prepared.jsonl"
OUTPUT_MODEL_PATH="./output/lora-full-1epoch-bs6"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

for path in "$PYTHON_BIN" "$INIT_MODEL_PATH" "$TRAIN_JSONL"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_MODEL_PATH")"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" sft_12hz_lora_mlflow.py \
  --init_model_path "$INIT_MODEL_PATH" \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --batch_size 1 \
  --gradient_accumulation_steps 6 \
  --num_epochs 1 \
  --lr 2e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --attn_implementation sdpa \
  --mlflow_tracking_uri "$MLFLOW_TRACKING_URI"
