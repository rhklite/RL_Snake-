#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

nohup conda run --no-capture-output -n snake python train.py --track "$@" > /dev/null 2>&1 &
PID=$!
disown "$PID"

echo "Training started in background (PID $PID)"
echo "Logs: check the latest runs/*/train.log"
echo "Stop: kill $PID"

sleep 10

LATEST_LOG=$(ls -t runs/*/train.log 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    URL=$(grep -oE 'https://wandb\.ai/[^ ]+' "$LATEST_LOG" | tail -1 || true)
    if [[ -n "$URL" ]]; then
        echo "Opening W&B dashboard: $URL"
        open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true
    else
        echo "W&B URL not found yet. Check $LATEST_LOG"
    fi
fi
