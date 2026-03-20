#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Configuration — override via environment variables
# ============================================================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ============================================================
# HaluEval QA
# ============================================================
echo "Running HaluEval QA with model: $MODEL_PATH"

nohup python scripts/compute_icr_halueval.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_DIR/HaluEval/qa_data.json" \
    --task qa \
    --pairing random \
    --seed 42 \
    --attn_implementation eager \
    --output_path "$OUTPUT_DIR/icr_halu_eval_random_qwen2.5.jsonl" \
    > "$LOG_DIR/icr_halu_eval_qwen2.5.log" 2>&1 &

echo "HaluEval job started (PID $!). Logs: $LOG_DIR/icr_halu_eval_qwen2.5.log"

# ============================================================
# SQuAD2
# ============================================================
# Uncomment to also run SQuAD2:
#
# echo "Running SQuAD2 with model: $MODEL_PATH"
# python scripts/compute_icr_squad2.py \
#     --model_name_or_path "$MODEL_PATH" \
#     --data_path "$DATA_DIR/SQuAD2.0/dev-v2.0.json" \
#     --task squad2 \
#     --pairing random \
#     --seed 42 \
#     --attn_implementation eager \
#     --output_path "$OUTPUT_DIR/icr_squad2_random_qwen2.5.jsonl"
