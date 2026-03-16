Run Snake PPO training with hypothesis tracking and W&B dashboard.

1. Read the `log-hypothesis` skill at `.cursor/skills/log-hypothesis/SKILL.md` and follow its steps.
2. The skill handles: hypothesis capture, slug confirmation, launching `bash scripts/train_wandb.sh` with the slug and any user-provided config overrides, and reporting the PID, run directory, and W&B URL.
3. Forward any arguments the user provides after `/train-wandb` as config overrides.
