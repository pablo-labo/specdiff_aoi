#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script:
# 1) prepare Fast-dLLM customized repo
# 2) install dependencies
# 3) optionally run smoke test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLLM_DIR="${ROOT_DIR}/third_party/Fast_dLLM_v2_1.5B"
PYTHON_BIN="python3"
SKIP_INSTALL=0
RUN_SMOKE=0

usage() {
  cat <<EOF
Usage: ./scripts/start.sh [options]

Options:
  --dllm-dir PATH       Target directory for Fast-dLLM clone/update
                        (default: ./third_party/Fast_dLLM_v2_1.5B)
  --python BIN          Python executable for dependency installation
                        (default: python3)
  --skip-install        Skip "pip install -r requirements.txt"
  --run-smoke           Run smoke test after initialization
  -h, --help            Show this help

Examples:
  ./scripts/start.sh
  ./scripts/start.sh --run-smoke
  ./scripts/start.sh --dllm-dir /data/Fast_dLLM_v2_1.5B --python python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dllm-dir)
      DLLM_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --run-smoke)
      RUN_SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git not found" >&2
  exit 1
fi

echo "[1/4] Prepare Fast-dLLM repository: ${DLLM_DIR}"
mkdir -p "$(dirname "${DLLM_DIR}")"

if [[ -d "${DLLM_DIR}/.git" ]]; then
  echo "      Existing repo detected, updating remotes and pulling latest."
else
  if [[ -e "${DLLM_DIR}" ]]; then
    echo "[ERROR] ${DLLM_DIR} exists but is not a git repo." >&2
    exit 1
  fi
  git clone https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B "${DLLM_DIR}"
fi

git -C "${DLLM_DIR}" remote set-url origin https://github.com/ruipeterpan/Fast_dLLM_v2_1.5B.git
git -C "${DLLM_DIR}" pull origin

DLLM_DIR="$(cd "${DLLM_DIR}" && pwd)"
echo "      Fast-dLLM ready at: ${DLLM_DIR}"

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  echo "[2/4] Install Python dependencies"
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" -m pip install -r requirements.txt
else
  echo "[2/4] Skip dependency installation"
fi

echo "[3/4] Export runtime env hints"
echo "      export DLLM_DIR=\"${DLLM_DIR}\""
echo "      export WANDB_MODE=offline"

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  echo "[4/4] Run smoke test"
  cd "${ROOT_DIR}"
  DLLM_DIR="${DLLM_DIR}" ./scripts/smoke_test.sh
else
  echo "[4/4] Skip smoke test (use --run-smoke to enable)"
fi

echo
echo "[DONE] Initialization completed."
echo "Next:"
echo "  DLLM_DIR=\"${DLLM_DIR}\" ./scripts/run_failfast_local.sh"
