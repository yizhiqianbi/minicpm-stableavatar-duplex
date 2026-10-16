#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m duplex_avatar.server --config "${1:-configs/default.yaml}"
