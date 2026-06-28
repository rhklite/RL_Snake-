"""Single-process PPO training for the Snake environment (CleanRL style)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from model import make_agent
from snake_env import SnakeEnv

CONFIG_DIR = Path(__file__).resolve().parent / "config"

_SELF_PROCESS = psutil.Process(os.getpid())


def _process_footprint_gb(pid: int) -> float:
    """Return process memory footprint in GB using macOS ``footprint`` tool.

    Includes MPS/GPU unified memory. Falls back to psutil RSS if the
    ``footprint`` command is unavailable or fails.
    """
    try:
        out = subprocess.check_output(
            ["footprint", str(pid)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if "Footprint:" in line:
                for token in line.split():
                    if token.replace(".", "", 1).isdigit():
                        val = float(token)
                        if "GB" in line:
                            return val
                        return val / 1024
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return psutil.Process(pid).memory_info().rss / 1e9


class _Tee:
    """Mirror writes to both a stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._file = log_file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()


def parse_args() -> tuple[DictConfig, bool, Path | None, Path | None]:
    """Load game + training YAML configs, merge CLI overrides.

    Returns (cfg, track, resume_dir, init_from).
    """
    parser = argparse.ArgumentParser(description="PPO Snake Training")
    parser.add_argument(
        "--game", type=str, default="default", help="Game config variant name"
    )
    parser.add_argument(
        "--training", type=str, default="default", help="Training config variant name"
    )
    parser.add_argument("--track", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing run directory to resume training from",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "Warm-start a FRESH run from a checkpoint (.pt) or run dir: loads agent "
            "weights ONLY (strict), with a fresh optimizer and start_update=0. Use for "
            "curriculum transfer across grid sizes (needs a size-agnostic encoder, "
            "i.e. model.adaptive_pool_size set). Mutually exclusive with --resume."
        ),
    )
    args, overrides = parser.parse_known_args()

    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    resume_dir: Path | None = None
    if args.resume:
        resume_dir = Path(args.resume)
        cfg = OmegaConf.load(resume_dir / "config.yaml")
    else:
        game_cfg = OmegaConf.load(CONFIG_DIR / "game" / f"{args.game}.yaml")
        train_cfg = OmegaConf.load(CONFIG_DIR / "training" / f"{args.training}.yaml")
        cfg = OmegaConf.merge(game_cfg, train_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    init_from = Path(args.init_from) if args.init_from else None
    return cfg, args.track, resume_dir, init_from


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the best checkpoint to resume from inside ckpt_dir.

    Priority: agent_final.pt > highest agent_N.pt > best.pt.
    """
    final = ckpt_dir / "agent_final.pt"
    if final.exists():
        return final

    numbered = sorted(
        ckpt_dir.glob("agent_*.pt"),
        key=lambda p: (
            int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1
        ),
    )
    numbered = [p for p in numbered if p.stem.split("_")[1].isdigit()]
    if numbered:
        return numbered[-1]

    best = ckpt_dir / "best.pt"
    if best.exists():
        return best

    return None


def make_env(rank: int, cfg: DictConfig) -> callable:
    def _init() -> gym.Env:
        env = SnakeEnv(
            rows=cfg.game.rows,
            cols=cfg.game.cols,
            obs_type=cfg.model.obs_type,
            render_mode=None,
            dist_shaping_alpha=cfg.training.get("dist_shaping_alpha", 0.0),
            step_penalty=cfg.training.get("step_penalty", -0.025),
            win_bonus=cfg.training.get("win_bonus", 0.0),
            cycle_beta=cfg.training.get("cycle_beta", 0.0),
            mask_mode=cfg.training.get("mask_mode", "none"),
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=cfg.training.seed + rank)
        return env

    return _init


def _gather_action_masks(envs, device: torch.device) -> torch.Tensor:
    """Per-env legal-action masks as a bool ``(num_envs, 4)`` tensor, for the CURRENT obs.

    Works for both the batched ``VectorizedSnakeEnv`` (one array op) and a
    ``SyncVectorEnv`` (stack each sub-env's ``legal_action_mask()``). Aligns with
    ``next_obs`` after a reset/step (auto-reset envs return the reset state's mask).
    """
    if hasattr(envs, "legal_action_mask"):  # batched env exposes it directly
        m = envs.legal_action_mask()
    else:
        m = np.stack([env.unwrapped.legal_action_mask() for env in envs.envs])
    return torch.as_tensor(m, dtype=torch.bool, device=device)


def _build_experiment_name(cfg: DictConfig) -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%m%d_%H")
    name = ts
    slug = OmegaConf.select(cfg, "training.hypothesis_slug", default="")
    if slug:
        name = f"{name}_{slug}"
    return name


def _ensure_clean_git(cfg: DictConfig) -> str:
    """Prompt user to commit uncommitted changes. Return current commit hash."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: git not available. Skipping reproducibility check.")
        return "unknown"

    if status:
        print("Uncommitted changes detected:\n" + status)
        resp = input("Commit before training? [Y/n] ").strip().lower()
        if resp != "n":
            subprocess.run(["git", "add", "-A"], check=True)
            slug = OmegaConf.select(cfg, "training.hypothesis_slug", default="")
            msg = (
                f"<chore> Pre-training snapshot: {slug}"
                if slug
                else "<chore> Pre-training snapshot"
            )
            subprocess.run(["git", "commit", "-m", msg], check=True)
        else:
            print(
                "WARNING: Running with uncommitted changes. Reproducibility not guaranteed."
            )

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _setup_experiment_dir(cfg: DictConfig) -> Path:
    """Create a uniquely named experiment directory and save the config snapshot."""
    base_dir = Path(cfg.experiments_dir)
    name = _build_experiment_name(cfg)
    exp_dir = base_dir / name

    if exp_dir.exists():
        version = 2
        while (base_dir / f"{name}_v{version}").exists():
            version += 1
        exp_dir = base_dir / f"{name}_v{version}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "videos").mkdir(exist_ok=True)

    OmegaConf.save(cfg, exp_dir / "config.yaml")

    slug = OmegaConf.select(cfg, "training.hypothesis_slug", default="")
    if slug:
        hypothesis_path = exp_dir / "hypothesis.md"
        hypothesis_path.write_text(
            f"# Hypothesis\n\n"
            f"**Slug**: {slug}\n\n"
            f"## What I'm testing\n"
            f"<!-- fill in before the run -->\n\n"
            f"## Expected outcome\n"
            f"<!-- fill in before the run -->\n\n"
            f"## Key config changes\n"
            f"<!-- e.g. ppo.ent_coef: 0.01 → 0.03 -->\n\n"
            f"## Result (fill in after run)\n"
            f"<!-- outcome, actual metrics, next steps -->\n"
        )

    return exp_dir


def _log_metrics(
    metrics: dict,
    metrics_file: object,
    track: bool,
) -> None:
    """Write one JSON line to the local metrics file and optionally log to W&B."""
    metrics_file.write(json.dumps(metrics) + "\n")
    metrics_file.flush()
    if track:
        import wandb

        wandb.log(metrics)


def _obs_to_device(
    obs_np, cfg: DictConfig, device: torch.device
) -> dict | torch.Tensor:
    """Convert raw numpy obs (dict or array) to tensors on device."""
    if cfg.model.obs_type == "hybrid":
        return {
            "grid": torch.tensor(obs_np["grid"], device=device, dtype=torch.float32),
            "food": torch.tensor(obs_np["food"], device=device, dtype=torch.float32),
        }
    dtype = torch.float32 if cfg.model.obs_type == "features" else torch.int8
    return torch.tensor(obs_np, device=device, dtype=dtype)


def _draw_overlay(
    frame: np.ndarray,
    cumulative_reward: float,
    step_count: int,
    snake_length: int,
    cause_of_death: str | None,
    update: int | None = None,
) -> np.ndarray:
    """Draw a HUD overlay in the bottom-right ~1/3 of the frame."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    H, W = frame.shape[:2]
    panel_x = 2 * W // 3
    panel_y = 2 * H // 3

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", size=max(12, H // 24)
        )
    except Exception:
        font = ImageFont.load_default()

    lines: list[str] = []
    if update is not None:
        lines.append(f"Update: {update}")
    lines += [
        f"Length: {snake_length}",
        f"Return: {cumulative_reward:.2f}",
        f"Steps:  {step_count}",
    ]
    if cause_of_death:
        lines.append(f"Death:  {cause_of_death}")

    margin = max(4, H // 80)
    line_height = max(14, H // 20)
    for i, line in enumerate(lines):
        color = (255, 80, 80) if (cause_of_death and i == len(lines) - 1) else (0, 0, 0)
        draw.text(
            (panel_x + margin, panel_y + margin + i * line_height),
            line,
            fill=color,
            font=font,
        )

    return np.array(img)


def _record_episode(
    agent: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    video_path: Path,
    update: int | None = None,
) -> None:
    """Run one episode with the current policy and save as mp4 with HUD overlay."""
    target_px = 720
    cell = min(
        cfg.video.render_cell_size,
        max(1, target_px // max(cfg.game.rows, cfg.game.cols)),
    )
    env = SnakeEnv(
        rows=cfg.game.rows,
        cols=cfg.game.cols,
        obs_type=cfg.model.obs_type,
        render_mode="rgb_array",
        render_cell_size=cell,
    )
    obs, _ = env.reset()
    cumulative_reward = 0.0
    step_count = 0
    cause_of_death = None
    done = False

    with imageio.get_writer(str(video_path), fps=cfg.video.fps) as writer:
        writer.append_data(
            _draw_overlay(
                env.render(),
                cumulative_reward,
                step_count,
                len(env.snake),
                None,
                update=update,
            )
        )
        while not done:
            obs_t = _obs_to_device(
                (
                    {k: v[np.newaxis] for k, v in obs.items()}
                    if isinstance(obs, dict)
                    else obs[np.newaxis]
                ),
                cfg,
                device,
            )
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            obs, reward, terminated, truncated, info = env.step(action.item())
            cumulative_reward += reward
            step_count += 1
            done = terminated or truncated
            if done:
                cause_of_death = info.get("cause_of_death")
            writer.append_data(
                _draw_overlay(
                    env.render(),
                    cumulative_reward,
                    step_count,
                    len(env.snake),
                    cause_of_death if done else None,
                    update=update,
                )
            )


def main() -> None:
    cfg, track, resume_dir, init_from = parse_args()

    if cfg.model.arch == "cnn" and cfg.model.obs_type != "grid":
        raise ValueError("CNN architecture requires obs_type='grid'")
    if cfg.model.arch in ("hybrid", "hybrid_asym") and cfg.model.obs_type != "hybrid":
        raise ValueError("Hybrid architecture requires obs_type='hybrid'")
    if cfg.model.num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    if not resume_dir:
        git_commit = _ensure_clean_git(cfg)
        cfg = OmegaConf.merge(cfg, {"git": {"commit": git_commit}})

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print("--- Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------")

    if resume_dir:
        exp_dir = resume_dir
        print(f"Resuming from: {exp_dir}")
    else:
        exp_dir = _setup_experiment_dir(cfg)
    log_file = open(exp_dir / "train.log", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    try:
        metrics_file = open(exp_dir / "metrics.jsonl", "a", encoding="utf-8")
        print(f"Experiment directory: {exp_dir}")

        if track:
            import wandb

            wandb.init(project="snake-ppo", config=config_dict)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {device}")

        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)

        use_vec_env = cfg.training.get("vectorized_env", False)
        if use_vec_env:
            if cfg.model.obs_type != "hybrid":
                raise ValueError("vectorized_env requires obs_type='hybrid'")
            from vec_snake_env import VectorizedSnakeEnv

            envs = VectorizedSnakeEnv(
                num_envs=cfg.training.num_envs,
                rows=cfg.game.rows,
                cols=cfg.game.cols,
                max_steps_factor=200,  # SnakeEnv default; make_env doesn't override it
                step_penalty=cfg.training.get("step_penalty", -0.025),
                win_bonus=cfg.training.get("win_bonus", 0.0),
                cycle_beta=cfg.training.get("cycle_beta", 0.0),
                mask_mode=cfg.training.get("mask_mode", "none"),
                seed=cfg.training.seed,
            )
        else:
            envs = gym.vector.SyncVectorEnv(
                [make_env(i, cfg) for i in range(cfg.training.num_envs)]
            )

        agent = make_agent(
            arch=cfg.model.arch,
            obs_type=cfg.model.obs_type,
            rows=cfg.game.rows,
            cols=cfg.game.cols,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            activation=cfg.model.activation,
            adaptive_pool_size=cfg.model.get("adaptive_pool_size", None),
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

        num_envs = cfg.training.num_envs
        num_steps = cfg.training.num_steps
        batch_size = num_envs * num_steps
        minibatch_size = batch_size // cfg.ppo.num_minibatches
        num_updates = cfg.training.get("num_updates", None)
        if num_updates is None:
            num_updates = cfg.training.total_timesteps // batch_size
        unlimited = num_updates == 0

        start_update = 0
        best_avg_return = -float("inf")
        # Select best.pt on this metric. "coverage" is reward-invariant, so it
        # stays comparable when a shaping term (e.g. cycle_beta) shifts avg_return.
        best_metric = cfg.training.get("best_metric", "avg_return")
        best_score = -float("inf")
        if resume_dir:
            ckpt_path = _find_latest_checkpoint(exp_dir / "checkpoints")
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {exp_dir / 'checkpoints'}"
                )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "agent" in ckpt:
                agent.load_state_dict(ckpt["agent"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_update = ckpt.get("update", 0)
                best_avg_return = ckpt.get("best_avg_return", -float("inf"))
                # Restore the best-selection threshold so the first post-resume
                # eval can't clobber best.pt. Pre-v10 checkpoints lack best_score;
                # fall back to best_avg_return (correct for the default metric).
                best_score = ckpt.get("best_score", best_avg_return)
            else:
                agent.load_state_dict(ckpt)
            print(
                f"Resumed from {ckpt_path.name} at update {start_update} "
                f"(best_avg_return={best_avg_return:.2f})"
            )
        elif init_from:
            init_ckpt = init_from
            if init_ckpt.is_dir():
                init_ckpt = _find_latest_checkpoint(
                    init_ckpt / "checkpoints"
                ) or _find_latest_checkpoint(init_ckpt)
                if init_ckpt is None:
                    raise FileNotFoundError(
                        f"No checkpoint found under {init_from}"
                    )
            ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
            state = (
                ckpt["agent"]
                if isinstance(ckpt, dict) and "agent" in ckpt
                else ckpt
            )
            # strict load: a shape mismatch (e.g. warm-starting across grid sizes
            # without a size-agnostic encoder) MUST fail visibly, not silently.
            if hasattr(agent, "load_actor_weights"):
                # Asymmetric agent: the checkpoint is a shared-trunk HybridActorCritic,
                # so load it into the actor pathway only; the privileged critic stays fresh.
                agent.load_actor_weights(state)
                print(
                    f"Warm-started ACTOR weights from {init_ckpt} "
                    f"(asymmetric critic fresh, fresh optimizer, start_update=0)"
                )
            else:
                agent.load_state_dict(state)
                print(
                    f"Warm-started agent weights from {init_ckpt} "
                    f"(fresh optimizer, start_update=0, best_avg_return reset)"
                )

        is_hybrid = cfg.model.obs_type == "hybrid"

        mask_mode = cfg.training.get("mask_mode", "none")
        use_mask = mask_mode != "none"
        if use_mask and not is_hybrid:
            raise ValueError(
                f"mask_mode={mask_mode!r} currently requires obs_type='hybrid' "
                f"(action masking is wired through HybridActorCritic only)."
            )

        if is_hybrid:
            rows, cols = cfg.game.rows, cfg.game.cols
            grid_channels = (
                4 if use_vec_env
                else envs.single_observation_space["grid"].shape[0]
            )
            obs_grid = torch.zeros(
                (num_steps, num_envs, grid_channels, rows, cols),
                dtype=torch.float32,
                device=device,
            )
            obs_food = torch.zeros(
                (num_steps, num_envs, 2), dtype=torch.float32, device=device
            )
            next_obs_grid_buf = torch.zeros(
                (num_envs, grid_channels, rows, cols),
                dtype=torch.float32,
                device=device,
            )
            next_obs_food_buf = torch.zeros(
                (num_envs, 2), dtype=torch.float32, device=device
            )
        else:
            obs_shape = envs.single_observation_space.shape
            obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)

        actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), device=device)
        values = torch.zeros((num_steps, num_envs), device=device)
        masks = torch.ones((num_steps, num_envs, 4), dtype=torch.bool, device=device)

        next_obs_np, _ = envs.reset(seed=cfg.training.seed)
        if is_hybrid:
            next_obs_grid_buf.copy_(
                torch.as_tensor(next_obs_np["grid"], dtype=torch.float32)
            )
            next_obs_food_buf.copy_(
                torch.as_tensor(next_obs_np["food"], dtype=torch.float32)
            )
            next_obs = {"grid": next_obs_grid_buf, "food": next_obs_food_buf}
        else:
            next_obs = _obs_to_device(next_obs_np, cfg, device)
        next_done = torch.zeros(num_envs, device=device)
        next_mask = _gather_action_masks(envs, device) if use_mask else None

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        snake_sizes: list[int] = []
        death_counts: dict[str, int] = {"wall": 0, "body": 0, "timeout": 0, "win": 0}
        grid_area = cfg.game.rows * cfg.game.cols
        start_time = time.time()
        train_start = time.monotonic()
        return_history: list[tuple[int, float]] = []

        max_memory_pct = cfg.training.get("max_memory_pct", 95.0)

        if not unlimited:
            ckpt_milestones = {
                round(num_updates * i / cfg.checkpointing.num_checkpoints)
                for i in range(1, cfg.checkpointing.num_checkpoints + 1)
            }
            video_milestones = {
                round(num_updates * i / cfg.video.num_videos)
                for i in range(1, cfg.video.num_videos + 1)
            }

        ckpt_interval = cfg.checkpointing.get("interval", 500)
        video_interval = cfg.video.get("interval", 1000)
        update = start_update
        nonfinite_grad_skips = 0  # cumulative PPO minibatch steps skipped on NaN/inf grad
        # LR annealing (v16 stability lever): linearly decay LR base->0 over
        # lr_anneal_updates to damp the late peak->decay of action-masked runs. Off by
        # default (lr_anneal_updates=0). Works for unlimited runs (fixed horizon, not
        # tied to num_updates); LR floors at 0 (frozen) past the horizon.
        anneal_lr = cfg.ppo.get("anneal_lr", False)
        lr_anneal_updates = cfg.ppo.get("lr_anneal_updates", 0)
        base_lr = cfg.ppo.learning_rate
        while True:
            update += 1
            if not unlimited and update > num_updates:
                break
            if anneal_lr and lr_anneal_updates > 0:
                # absolute update (NOT relative to start_update) so the schedule continues
                # correctly across --resume instead of restarting at base LR.
                frac = max(0.0, 1.0 - (update - 1) / lr_anneal_updates)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * frac
            # --- Early stopping: time limit ---
            elapsed_hours = (time.monotonic() - train_start) / 3600
            if cfg.training.max_hours > 0 and elapsed_hours >= cfg.training.max_hours:
                print(
                    f"Stopping: time limit reached ({elapsed_hours:.2f}h >= {cfg.training.max_hours}h)"
                )
                _log_metrics(
                    {"stop_reason": "time_limit", "elapsed_hours": elapsed_hours},
                    metrics_file,
                    track,
                )
                break

            # --- Early stopping: system memory guard ---
            sys_mem = psutil.virtual_memory()
            sys_mem_pct = sys_mem.percent
            sys_used_gb = sys_mem.used / 1e9
            proc_gb = _process_footprint_gb(os.getpid())
            if sys_mem_pct >= max_memory_pct:
                print(
                    f"Stopping: system memory limit reached "
                    f"({sys_mem_pct:.1f}% >= {max_memory_pct}% | "
                    f"used {sys_used_gb:.1f}GB, process {proc_gb:.1f}GB)"
                )
                _log_metrics(
                    {"stop_reason": "memory_limit", "sys_mem_pct": sys_mem_pct},
                    metrics_file,
                    track,
                )
                break

            # --- Rollout ---
            for step in range(num_steps):
                if is_hybrid:
                    obs_grid[step] = next_obs["grid"]
                    obs_food[step] = next_obs["food"]
                else:
                    obs[step] = next_obs
                dones[step] = next_done
                if use_mask:
                    masks[step] = next_mask

                with torch.no_grad():
                    mask_kw = {"action_mask": next_mask} if use_mask else {}
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs, **mask_kw
                    )
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()

                next_obs_np, reward_np, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                done_np = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(
                    reward_np, device=device, dtype=torch.float32
                )
                if is_hybrid:
                    next_obs_grid_buf.copy_(
                        torch.as_tensor(next_obs_np["grid"], dtype=torch.float32)
                    )
                    next_obs_food_buf.copy_(
                        torch.as_tensor(next_obs_np["food"], dtype=torch.float32)
                    )
                else:
                    next_obs = _obs_to_device(next_obs_np, cfg, device)
                next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
                if use_mask:
                    # Post-step (post-autoreset) mask for next_obs; stored next loop iter.
                    next_mask = _gather_action_masks(envs, device)

                if "_episode" in infos:
                    ep_mask = infos["_episode"]
                    for idx in range(num_envs):
                        if ep_mask[idx]:
                            episode_returns.append(float(infos["episode"]["r"][idx]))
                            episode_lengths.append(int(infos["episode"]["l"][idx]))
                            snake_sizes.append(int(infos["snake_length"][idx]))
                            if infos.get(
                                "_cause_of_death", np.zeros(num_envs, dtype=bool)
                            )[idx]:
                                cod = infos["cause_of_death"][idx]
                                if cod in death_counts:
                                    death_counts[cod] += 1

            # --- GAE ---
            with torch.no_grad():
                next_value = agent.get_value(next_obs).flatten()
                advantages = torch.zeros_like(rewards)
                last_gae = 0.0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + cfg.ppo.gamma * nextvalues * next_nonterminal
                        - values[t]
                    )
                    advantages[t] = last_gae = (
                        delta
                        + cfg.ppo.gamma
                        * cfg.ppo.gae_lambda
                        * next_nonterminal
                        * last_gae
                    )
                returns = advantages + values

            # --- Flatten rollout buffers ---
            if is_hybrid:
                b_obs = {
                    "grid": obs_grid.reshape(
                        -1, grid_channels, cfg.game.rows, cfg.game.cols
                    ),
                    "food": obs_food.reshape(-1, 2),
                }
            else:
                b_obs = obs.reshape((-1,) + obs_shape)

            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)  # noqa: F841
            b_masks = masks.reshape(-1, 4) if use_mask else None

            b_inds = np.arange(batch_size)
            clipfracs = []

            for _ in range(cfg.ppo.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    if is_hybrid:
                        mb_obs = {
                            "grid": b_obs["grid"][mb_inds],
                            "food": b_obs["food"][mb_inds],
                        }
                    else:
                        mb_obs = b_obs[mb_inds].float()

                    mb_mask_kw = (
                        {"action_mask": b_masks[mb_inds]} if use_mask else {}
                    )
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_obs, b_actions[mb_inds], **mb_mask_kw
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs.append(
                            ((ratio - 1.0).abs() > cfg.ppo.clip_coef)
                            .float()
                            .mean()
                            .item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - cfg.ppo.ent_coef * entropy_loss
                        + cfg.ppo.vf_coef * v_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        agent.parameters(), cfg.ppo.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        # A non-finite grad (NaN/inf) would poison every weight on step()
                        # and produce all-NaN logits next forward. Skip this minibatch
                        # instead; count it so a recurrence is visible, not silent.
                        optimizer.zero_grad(set_to_none=True)
                        nonfinite_grad_skips += 1
                        continue
                    optimizer.step()

            # --- Logging ---
            elapsed = time.time() - start_time
            global_step = update * batch_size
            sps = int(global_step / elapsed)
            metrics: dict = {
                "sps": sps,
                "policy_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "clip_frac": float(np.mean(clipfracs)),
                "grad_skips": nonfinite_grad_skips,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if episode_returns:
                recent_n = min(20, len(episode_returns))
                metrics["avg_return"] = float(np.mean(episode_returns[-recent_n:]))
                metrics["avg_length"] = float(np.mean(episode_lengths[-recent_n:]))
                metrics["avg_snake_length"] = float(np.mean(snake_sizes[-recent_n:]))
                metrics["avg_coverage"] = metrics["avg_snake_length"] / grid_area
                metrics["max_snake_length"] = float(max(snake_sizes[-recent_n:]))
                if unlimited:
                    prefix = f"Update {update}"
                else:
                    pct = 100.0 * update / num_updates
                    prefix = f"[{pct:.1f}%] Update {update}/{num_updates}"
                print(
                    f"{prefix} | "
                    f"Avg Return: {metrics['avg_return']:.2f} | "
                    f"Avg Length: {metrics['avg_length']:.1f} | "
                    f"Avg Snake Size: {metrics['avg_snake_length']:.1f} | "
                    f"SPS: {sps:,}"
                )

            # --- Death breakdown as scalar metrics ---
            total_deaths = sum(death_counts.values())
            if total_deaths > 0:
                for k, v in death_counts.items():
                    metrics[f"death/{k}_pct"] = 100.0 * v / total_deaths

            _log_metrics(metrics, metrics_file, track)

            # --- Cap episode history to avoid unbounded growth ---
            episode_returns = episode_returns[-100:]
            episode_lengths = episode_lengths[-100:]
            snake_sizes = snake_sizes[-100:]

            # --- Best-reward tracking ---
            if episode_returns:
                cand = (
                    metrics["avg_coverage"]
                    if best_metric == "coverage"
                    else metrics["avg_return"]
                )
                if cand > best_score:
                    best_score = cand
                    best_avg_return = metrics["avg_return"]  # recorded for back-compat
                    best_ckpt = exp_dir / "checkpoints" / "best.pt"
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "update": update,
                            "best_avg_return": best_avg_return,
                            "best_score": best_score,
                            "best_metric": best_metric,
                        },
                        best_ckpt,
                    )
                    best_vid = exp_dir / "videos" / "best.mp4"
                    _record_episode(agent, cfg, device, best_vid, update=update)
                    print(
                        f"  New best return {best_avg_return:.2f} -> "
                        f"saved {best_ckpt} and {best_vid}"
                    )
                    if track:
                        import wandb

                        wandb.log(
                            {
                                "best_video": wandb.Video(
                                    str(best_vid),
                                    caption=f"best return {best_avg_return:.2f}",
                                    fps=cfg.video.fps,
                                    format="mp4",
                                )
                            }
                        )

            # --- Early stopping: plateau detection ---
            if episode_returns and "avg_return" in metrics:
                return_history.append((global_step, metrics["avg_return"]))
                plateau_steps = cfg.training.plateau_steps
                return_history = [
                    (s, r)
                    for s, r in return_history
                    if global_step - s <= plateau_steps
                ]
                if global_step >= plateau_steps and len(return_history) >= 10:
                    returns_in_window = [r for _, r in return_history]
                    r_range = max(returns_in_window) - min(returns_in_window)
                    if r_range <= cfg.training.plateau_threshold:
                        print(
                            f"Stopping: plateau detected "
                            f"(return range {r_range:.3f} <= {cfg.training.plateau_threshold} "
                            f"over last {plateau_steps:,} steps)"
                        )
                        _log_metrics(
                            {"stop_reason": "plateau", "plateau_range": r_range},
                            metrics_file,
                            track,
                        )
                        break

            # --- Checkpoint + Video milestones ---
            save_ckpt = (
                (update % ckpt_interval == 0)
                if unlimited
                else (update in ckpt_milestones)
            )
            if save_ckpt:
                ckpt_path = exp_dir / "checkpoints" / f"agent_{update}.pt"
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "best_avg_return": best_avg_return,
                        "best_score": best_score,
                        "best_metric": best_metric,
                    },
                    ckpt_path,
                )
                print(f"  Saved checkpoint: {ckpt_path}")

            save_video = (
                (update % video_interval == 0)
                if unlimited
                else (update in video_milestones)
            )
            if save_video:
                vid_path = exp_dir / "videos" / f"update_{update:05d}.mp4"
                _record_episode(agent, cfg, device, vid_path, update=update)
                print(f"  Recorded video: {vid_path}")

        # --- Final saves ---
        final_ckpt = exp_dir / "checkpoints" / "agent_final.pt"
        torch.save(
            {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "best_avg_return": best_avg_return,
                "best_score": best_score,
                "best_metric": best_metric,
            },
            final_ckpt,
        )
        print(f"Training complete. Final model saved to {final_ckpt}")

        for ep_idx in range(cfg.video.final_episodes):
            vid_path = exp_dir / "videos" / f"final_{ep_idx}.mp4"
            _record_episode(agent, cfg, device, vid_path, update=update)
            if track:
                import wandb

                wandb.log(
                    {
                        f"final_video_{ep_idx}": wandb.Video(
                            str(vid_path),
                            caption=f"final episode {ep_idx}",
                            fps=cfg.video.fps,
                            format="mp4",
                        )
                    }
                )
        print(f"  Recorded {cfg.video.final_episodes} final episode videos")

        metrics_file.close()
        envs.close()

        if track:
            import wandb

            wandb.save(str(final_ckpt))
            wandb.finish()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
