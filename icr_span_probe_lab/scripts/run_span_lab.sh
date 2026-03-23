#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_DIR="${ROOT_DIR}/icr_span_probe_lab"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-minimal}"
DEVICE="${DEVICE:-cpu}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

with_optional_max_samples() {
  local -a cmd=("$@")
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  run_cmd "${cmd[@]}"
}

prepare_span_ready() {
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/prepare_span_ready_data.py"
}

build_tokenizer_route() {
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_tokenizer_windows.py"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_silver_span_labels.py" \
    --span_path "${LAB_DIR}/data/span_candidates/tokenizer_windows.jsonl"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_span_dataset.py" \
    --labeled_span_path "${LAB_DIR}/data/span_labels/tokenizer_windows_silver_labels.jsonl"
}

build_spacy_route() {
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_spacy_spans.py" --spacy_model "${SPACY_MODEL}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_silver_span_labels.py" \
    --span_path "${LAB_DIR}/data/span_candidates/spacy_spans.jsonl"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_span_dataset.py" \
    --labeled_span_path "${LAB_DIR}/data/span_labels/spacy_spans_silver_labels.jsonl"
}

train_minimal() {
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_baseline_mlp.py" \
    --dataset_path "${LAB_DIR}/data/datasets/tokenizer_windows_dataset.jsonl" \
    --device "${DEVICE}"
}

train_full_for_dataset() {
  local dataset_path="$1"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_baseline_mlp.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_discrepancy.py" \
    --dataset_path "${dataset_path}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_temporal_conv.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_change_point.py" \
    --dataset_path "${dataset_path}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_trajectory_encoder.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
}

generate_figures() {
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/generate_default_figures.py"
}

show_usage() {
  cat <<EOF
Usage:
  bash icr_span_probe_lab/scripts/run_span_lab.sh [minimal|full|figures-only]

Environment variables:
  PYTHON_BIN=python3
  DEVICE=cpu|cuda
  MAX_SAMPLES=1000
  SPACY_MODEL=en_core_web_sm

Examples:
  bash icr_span_probe_lab/scripts/run_span_lab.sh minimal
  DEVICE=cuda bash icr_span_probe_lab/scripts/run_span_lab.sh full
  bash icr_span_probe_lab/scripts/run_span_lab.sh figures-only
EOF
}

case "${MODE}" in
  minimal)
    prepare_span_ready
    build_tokenizer_route
    train_minimal
    generate_figures
    ;;
  full)
    prepare_span_ready
    build_tokenizer_route
    build_spacy_route
    train_full_for_dataset "${LAB_DIR}/data/datasets/tokenizer_windows_dataset.jsonl"
    train_full_for_dataset "${LAB_DIR}/data/datasets/spacy_spans_dataset.jsonl"
    generate_figures
    ;;
  figures-only)
    generate_figures
    ;;
  *)
    show_usage
    exit 1
    ;;
esac
