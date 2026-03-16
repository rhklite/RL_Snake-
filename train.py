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


def parse_args() -> tuple[DictConfig, bool]:
    """Load game + training YAML configs, merge CLI overrides, return (cfg, track)."""
    parser = argparse.ArgumentParser(description="PPO Snake Training")
    parser.add_argument(
        "--game", type=str, default="default", help="Game config variant name"
    )
    parser.add_argument(
        "--training", type=str, default="default", help="Training config variant name"
    )
    parser.add_argument("--track", action="store_true", help="Enable W&B logging")
    args, overrides = parser.parse_known_args()

    game_cfg = OmegaConf.load(CONFIG_DIR / "game" / f"{args.game}.yaml")
    train_cfg = OmegaConf.load(CONFIG_DIR / "training" / f"{args.training}.yaml")
    cfg = OmegaConf.merge(game_cfg, train_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg, args.track


def make_env(rank: int, cfg: DictConfig) -> callable:
    def _init() -> gym.Env:
        env = SnakeEnv(
            rows=cfg.game.rows,
            cols=cfg.game.cols,
            obs_type=cfg.model.obs_type,
            render_mode=None,
            dist_shaping_alpha=cfg.training.get("dist_shaping_alpha", 0.0),
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=cfg.training.seed + rank)
        return env

    return _init


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
            print("WARNING: Running with uncommitted changes. Reproducibility not guaranteed.")

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
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

    lines = [
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
                env.render(), cumulative_reward, step_count, len(env.snake), None
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
                )
            )


def main() -> None:
    cfg, track = parse_args()

    if cfg.model.arch == "cnn" and cfg.model.obs_type != "grid":
        raise ValueError("CNN architecture requires obs_type='grid'")
    if cfg.model.arch == "hybrid" and cfg.model.obs_type != "hybrid":
        raise ValueError("Hybrid architecture requires obs_type='hybrid'")
    if cfg.model.num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    git_commit = _ensure_clean_git(cfg)
    cfg = OmegaConf.merge(cfg, {"git": {"commit": git_commit}})

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print("--- Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------")

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
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

        num_envs = cfg.training.num_envs
        num_steps = cfg.training.num_steps
        batch_size = num_envs * num_steps
        minibatch_size = batch_size // cfg.ppo.num_minibatches
        num_updates = cfg.training.total_timesteps // batch_size

        is_hybrid = cfg.model.obs_type == "hybrid"

        if is_hybrid:
            rows, cols = cfg.game.rows, cfg.game.cols
            grid_channels = envs.single_observation_space["grid"].shape[0]
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

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        snake_sizes: list[int] = []
        death_counts: dict[str, int] = {"wall": 0, "body": 0, "timeout": 0}
        start_time = time.time()
        train_start = time.monotonic()
        best_avg_return = -float("inf")
        return_history: list[tuple[int, float]] = []

        max_memory_pct = cfg.training.get("max_memory_pct", 95.0)

        ckpt_milestones = {
            round(num_updates * i / cfg.checkpointing.num_checkpoints)
            for i in range(1, cfg.checkpointing.num_checkpoints + 1)
        }
        video_milestones = {
            round(num_updates * i / cfg.video.num_videos)
            for i in range(1, cfg.video.num_videos + 1)
        }

        for update in range(1, num_updates + 1):
            # --- Early stopping: time limit ---
            elapsed_hours = (time.monotonic() - train_start) / 3600
            if elapsed_hours >= cfg.training.max_hours:
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
            rss_gb = _SELF_PROCESS.memory_info().rss / 1e9
            if sys_mem_pct >= max_memory_pct:
                print(
                    f"Stopping: system memory limit reached "
                    f"({sys_mem_pct:.1f}% >= {max_memory_pct}% | "
                    f"used {sys_used_gb:.1f}GB, process RSS {rss_gb:.1f}GB)"
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

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
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

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_obs, b_actions[mb_inds]
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
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
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
            }
            if episode_returns:
                recent_n = min(20, len(episode_returns))
                metrics["avg_return"] = float(np.mean(episode_returns[-recent_n:]))
                metrics["avg_length"] = float(np.mean(episode_lengths[-recent_n:]))
                metrics["avg_snake_length"] = float(np.mean(snake_sizes[-recent_n:]))
                pct = 100.0 * update / num_updates
                print(
                    f"[{pct:.1f}%] Update {update}/{num_updates} | "
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
                avg_return = metrics["avg_return"]
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_ckpt = exp_dir / "checkpoints" / "best.pt"
                    torch.save(agent.state_dict(), best_ckpt)
                    best_vid = exp_dir / "videos" / "best.mp4"
                    _record_episode(agent, cfg, device, best_vid)
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
                if global_step >= plateau_steps:
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

            # --- Checkpoint + Video (percentage milestones) ---
            if update in ckpt_milestones:
                ckpt_path = exp_dir / "checkpoints" / f"agent_{update}.pt"
                torch.save(agent.state_dict(), ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            if update in video_milestones:
                vid_path = exp_dir / "videos" / f"update_{update:05d}.mp4"
                _record_episode(agent, cfg, device, vid_path)
                print(f"  Recorded video: {vid_path}")

        # --- Final saves ---
        final_ckpt = exp_dir / "checkpoints" / "agent_final.pt"
        torch.save(agent.state_dict(), final_ckpt)
        print(f"Training complete. Final model saved to {final_ckpt}")

        for ep_idx in range(cfg.video.final_episodes):
            vid_path = exp_dir / "videos" / f"final_{ep_idx}.mp4"
            _record_episode(agent, cfg, device, vid_path)
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
