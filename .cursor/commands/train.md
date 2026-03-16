Run Snake PPO training locally (no W&B).

Execute `bash scripts/train.sh` in a background terminal. Forward any arguments the user provides after `/train` (e.g. `/train --game big_grid --training fast_debug model.arch=mlp`).

The script runs `python train.py` directly inside the container. Report which terminal the training is running in and the PID.
