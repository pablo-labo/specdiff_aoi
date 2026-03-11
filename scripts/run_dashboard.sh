#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "${ROOT_DIR}"
python3 ./scripts/dashboard_server.py --host "${HOST}" --port "${PORT}"
