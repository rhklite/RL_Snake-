#!/usr/bin/env bash
# Auto-retry training wrapper: halves num_envs on memory_limit stop, down to MIN_ENVS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

NUM_ENVS="${NUM_ENVS:-256}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
MIN_ENVS=16
EXTRA_ARGS=("$@")

while true; do
    echo ""
    echo "=== Starting training: num_envs=$NUM_ENVS num_minibatches=$NUM_MINIBATCHES ==="

    # Run training in foreground so we can detect exit and inspect logs
    conda run --no-capture-output -n snake python train.py --track \
        --training hybrid_64x64 \
        "training.num_envs=${NUM_ENVS}" \
        "ppo.num_minibatches=${NUM_MINIBATCHES}" \
        "${EXTRA_ARGS[@]}" || true

    # Find the most recent metrics.jsonl and check stop reason
    LATEST_METRICS=$(ls -t runs/*/metrics.jsonl 2>/dev/null | head -1)
    STOP_REASON=""
    if [[ -n "$LATEST_METRICS" ]]; then
        STOP_REASON=$(python3 -c "
import json, sys
with open('$LATEST_METRICS') as f:
    lines = [json.loads(l) for l in f if l.strip()]
for line in reversed(lines):
    if 'stop_reason' in line:
        print(line['stop_reason'])
        sys.exit(0)
print('')
" 2>/dev/null || true)
    fi

    echo "Stop reason: '${STOP_REASON}'"

    if [[ "$STOP_REASON" == "memory_limit" ]]; then
        if [[ "$NUM_ENVS" -le "$MIN_ENVS" ]]; then
            echo "Memory limit hit at minimum envs ($MIN_ENVS). Stopping."
            exit 1
        fi
        NUM_ENVS=$(( NUM_ENVS / 2 ))
        NUM_MINIBATCHES=$(( NUM_MINIBATCHES / 2 ))
        if [[ "$NUM_MINIBATCHES" -lt 1 ]]; then
            NUM_MINIBATCHES=1
        fi
        echo "Memory limit hit — retrying with num_envs=$NUM_ENVS num_minibatches=$NUM_MINIBATCHES"
    else
        echo "Training finished (reason: '${STOP_REASON:-normal}'). Done."
        exit 0
    fi
done
