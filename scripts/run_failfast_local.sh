#!/usr/bin/env bash
set -euo pipefail

# Generic launcher for local or remote non-Slurm machines.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/project_env.sh"
: "${DLLM_DIR:?Set DLLM_DIR to your Fast_dLLM_v2_1.5B directory}"
bash "${ROOT_DIR}/scripts/validate_model_dir.sh" "${DLLM_DIR}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-math}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/local_run}"
WANDB_MODE="${WANDB_MODE:-offline}"

NUM_QUESTIONS="${NUM_QUESTIONS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SPEC_LEN="${SPEC_LEN:-8}"

NUM_DRAFTERS="${NUM_DRAFTERS:-1}"
TARGET_GPU="${TARGET_GPU:-0}"
DRAFTER_GPUS_RAW="${DRAFTER_GPUS:-0}"

read -r -a DRAFTER_GPUS_ARR <<< "${DRAFTER_GPUS_RAW}"

python failfast.py \
  --dataset_name "${DATASET}" \
  --target_model_name "${TARGET_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dllm_dir "${DLLM_DIR}" \
  --num_questions "${NUM_QUESTIONS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --spec_len "${SPEC_LEN}" \
  --run_dllm_sf \
  --baseline_sweep \
  --multi_gpu \
  --num_drafters "${NUM_DRAFTERS}" \
  --target_gpu "${TARGET_GPU}" \
  --drafter_gpus "${DRAFTER_GPUS_ARR[@]}" \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
