Watch the latest training run's metrics live in the terminal with a progress bar and rolling table.

Find the most recently modified `metrics.jsonl` under `runs/`. If none exists, respond with "No metrics.jsonl found under runs/".

Otherwise, output the following as a fenced bash code block for the user to copy-paste into their terminal — do NOT run it:

```bash
python3 scripts/watch_metrics.py <path_to_metrics.jsonl>
```

Also state which run it points to (the parent directory name).
