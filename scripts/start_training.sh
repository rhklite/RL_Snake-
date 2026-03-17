#!/usr/bin/env bash
# ── Configurable variables (Cursor agent updates these) ──────────────
TRAINING_CONFIG="reachability"
TRACK="--track"
EXTRA_ARGS=""
TMUX_SESSION="snake-train"
# ─────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/miniconda3/envs/snake/bin/python"
PID_FILE="$REPO_DIR/runs/.active_training_pid"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    exec tmux attach -t "$TMUX_SESSION"
fi

cd "$REPO_DIR" || exit 1

echo "n" | nohup "$PYTHON" train.py --training "$TRAINING_CONFIG" $TRACK $EXTRA_ARGS \
    >/dev/null 2>&1 &
TRAIN_PID=$!
mkdir -p runs
echo "$TRAIN_PID" > "$PID_FILE"

sleep 3

LATEST_RUN=$(ls -td runs/*/metrics.jsonl 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    LATEST_RUN="runs/*/metrics.jsonl"
fi

tmux new-session -d -s "$TMUX_SESSION" \
    "$PYTHON $REPO_DIR/scripts/watch_metrics.py $LATEST_RUN"

osascript -e "display notification \"PID $TRAIN_PID — config: $TRAINING_CONFIG\" with title \"Snake Training Started\""

echo "Training started (PID $TRAIN_PID), tmux session: $TMUX_SESSION"
echo "Reattach: tmux attach -t $TMUX_SESSION"
