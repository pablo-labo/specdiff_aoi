#!/usr/bin/env bash
set -euo pipefail

# Project-scoped runtime/cache directories.
# Keep large downloads on the project volume instead of container ephemeral disk.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${ROOT_DIR}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${ROOT_DIR}/.cache/huggingface/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT_DIR}/.cache/pip}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/.cache/wandb}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"

mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${WANDB_DIR}" "${TMPDIR}"
