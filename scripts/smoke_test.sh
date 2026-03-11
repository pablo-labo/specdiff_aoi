#!/usr/bin/env bash
set -euo pipefail

# Minimal deployment health check.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/project_env.sh"
: "${DLLM_DIR:?Set DLLM_DIR to your Fast_dLLM_v2_1.5B directory}"
"${ROOT_DIR}/scripts/validate_model_dir.sh" "${DLLM_DIR}"

python failfast.py \
  --dataset_name math \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./outputs/smoke \
  --dllm_dir "${DLLM_DIR}" \
  --num_questions 1 \
  --max_new_tokens 64 \
  --spec_len 4 \
  --run_dllm_sf \
  --baseline_sweep \
  --multi_gpu \
  --num_drafters 1 \
  --target_gpu 0 \
  --drafter_gpus 0 \
  --wandb_mode disabled
