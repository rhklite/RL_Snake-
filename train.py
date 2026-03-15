"""Single-process PPO training for the Snake environment (CleanRL style)."""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import make_agent
from snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO Snake Training")
    parser.add_argument(
        "--obs-type", type=str, default="grid", choices=["grid", "features"]
    )
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--save-interval", type=int, default=50, help="Save checkpoint every N updates"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def make_env(rank: int, args: argparse.Namespace) -> callable:
    def _init() -> gym.Env:
        env = SnakeEnv(
            rows=args.rows,
            cols=args.cols,
            obs_type=args.obs_type,
            render_mode=None,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(i, args) for i in range(args.num_envs)])

    agent = make_agent(args.obs_type, args.rows, args.cols).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.long, device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(
        next_obs_np,
        device=device,
        dtype=torch.float32 if args.obs_type == "features" else torch.int8,
    )
    next_done = torch.zeros(args.num_envs, device=device)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # --- Rollout ---
        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.float())
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            next_obs_np, reward_np, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward_np, device=device, dtype=torch.float32)
            next_obs = torch.tensor(
                next_obs_np,
                device=device,
                dtype=torch.float32 if args.obs_type == "features" else torch.int8,
            )
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            if "_episode" in infos:
                ep_mask = infos["_episode"]
                for idx in range(args.num_envs):
                    if ep_mask[idx]:
                        episode_returns.append(float(infos["episode"]["r"][idx]))
                        episode_lengths.append(int(infos["episode"]["l"][idx]))

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs.float()).flatten()
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * next_nonterminal - values[t]
                )
                advantages[t] = last_gae = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                )
            returns = advantages + values

        # --- PPO Update ---
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)  # noqa: F841 — kept for clarity

        b_inds = np.arange(batch_size)
        clipfracs = []

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds].float(), b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- Logging ---
        elapsed = time.time() - start_time
        sps = int((update * batch_size) / elapsed)
        if episode_returns:
            recent = episode_returns[-min(20, len(episode_returns)) :]
            avg_return = np.mean(recent)
            avg_length = np.mean(episode_lengths[-min(20, len(episode_lengths)) :])
            print(
                f"Update {update}/{num_updates} | "
                f"SPS: {sps} | "
                f"Avg Return: {avg_return:.2f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Policy Loss: {pg_loss.item():.4f} | "
                f"Value Loss: {v_loss.item():.4f} | "
                f"Entropy: {entropy_loss.item():.4f} | "
                f"Clip Frac: {np.mean(clipfracs):.4f}"
            )

        if update % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"agent_{update}.pt")
            torch.save(agent.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

    # Final save
    path = os.path.join(args.checkpoint_dir, "agent_final.pt")
    torch.save(agent.state_dict(), path)
    print(f"Training complete. Final model saved to {path}")

    envs.close()


if __name__ == "__main__":
    main()
