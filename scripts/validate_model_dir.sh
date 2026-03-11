#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[ERROR] Usage: ./scripts/validate_model_dir.sh /path/to/model_dir" >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Model directory does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

mapfile -t WEIGHT_FILES < <(find "${MODEL_DIR}" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) | sort)

if [[ "${#WEIGHT_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No model weight files found under: ${MODEL_DIR}" >&2
  exit 1
fi

for file in "${WEIGHT_FILES[@]}"; do
  if [[ ! -s "${file}" ]]; then
    echo "[ERROR] Empty model weight file: ${file}" >&2
    exit 1
  fi

  if [[ "${file}" == *.safetensors ]]; then
    if [[ "$(wc -c < "${file}")" -lt 1048576 ]]; then
      header="$(head -c 256 "${file}" || true)"
      if [[ "${header}" == version\ https://git-lfs.github.com/spec/v1* ]]; then
        echo "[ERROR] Git LFS pointer detected instead of real weights: ${file}" >&2
        exit 1
      fi
      echo "[ERROR] Suspiciously small safetensors file: ${file}" >&2
      exit 1
    fi
  fi
done

echo "[OK] Local model weights look sane: ${MODEL_DIR}"
