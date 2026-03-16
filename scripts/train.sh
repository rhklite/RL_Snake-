#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

nohup conda run --no-capture-output -n snake python train.py "$@" > /dev/null 2>&1 &
PID=$!
disown "$PID"

echo "Training started in background (PID $PID)"
echo "Logs: check the latest runs/*/train.log"
echo "Stop: kill $PID"
