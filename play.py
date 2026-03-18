"""Human-playable Snake using SnakeEnv with pygame rendering.

Supports AI agent playback via --agent flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch
from omegaconf import OmegaConf

from model import make_agent
from snake_env import CELL_BODY, CELL_FOOD, CELL_HEAD, DOWN, LEFT, RIGHT, UP, SnakeEnv

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load_agent(ckpt_path: Path, device: torch.device):
    """Load agent from checkpoint. Returns (agent, cfg) tuple."""
    run_dir = ckpt_path.parent.parent
    cfg = OmegaConf.load(run_dir / "config.yaml")

    agent = make_agent(
        arch=cfg.model.arch,
        obs_type=cfg.model.obs_type,
        rows=cfg.game.rows,
        cols=cfg.game.cols,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        activation=cfg.model.activation,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()
    return agent, cfg


def _obs_to_device(obs_np, obs_type: str, device: torch.device):
    """Convert numpy obs to tensor on device."""
    if obs_type == "hybrid":
        return {
            "grid": torch.tensor(
                obs_np["grid"][np.newaxis], device=device, dtype=torch.float32
            ),
            "food": torch.tensor(
                obs_np["food"][np.newaxis], device=device, dtype=torch.float32
            ),
        }
    dtype = torch.float32 if obs_type == "features" else torch.int8
    return torch.tensor(obs_np[np.newaxis], device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake")
    parser.add_argument(
        "--game", type=str, default="default", help="Game config variant name"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to checkpoint file for AI agent playback",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (agent mode)"
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    agent_model = None
    if args.agent:
        ckpt_path = Path(args.agent)
        agent_model, cfg = _load_agent(ckpt_path, device)
        obs_type = cfg.model.obs_type
    else:
        cfg = OmegaConf.load(CONFIG_DIR / "game" / f"{args.game}.yaml")
        obs_type = "grid"

    rows = cfg.game.rows
    cols = cfg.game.cols
    cell_size = cfg.gui.cell_size
    colors = cfg.gui.color

    bg_color = tuple(colors.background[:3])
    grid_color = tuple(colors.grid[:3])
    head_color = tuple(colors.snake.head)
    body_color = tuple(colors.snake.body)
    food_color = (213, 50, 80)

    env = SnakeEnv(rows=rows, cols=cols, obs_type=obs_type, render_mode=None)
    obs, _ = env.reset()

    pygame.init()
    screen_w = cols * cell_size
    screen_h = rows * cell_size
    screen = pygame.display.set_mode((screen_w, screen_h))
    caption = (
        "Snake Agent" if agent_model else cfg.gui.get("display_app_name", "Snake Game")
    )
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont(None, 28)

    key_map = {
        pygame.K_UP: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_RIGHT: RIGHT,
        pygame.K_w: UP,
        pygame.K_s: DOWN,
        pygame.K_a: LEFT,
        pygame.K_d: RIGHT,
    }

    current_action = RIGHT
    running = True
    game_over = False
    step_count = 0
    cumulative_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game_over and not agent_model:
                    obs, _ = env.reset()
                    current_action = RIGHT
                    game_over = False
                    step_count = 0
                    cumulative_reward = 0.0
                elif event.key in key_map and not game_over and not agent_model:
                    current_action = key_map[event.key]

        if not game_over:
            if agent_model:
                obs_t = _obs_to_device(obs, obs_type, device)
                with torch.no_grad():
                    action, _, _, _ = agent_model.get_action_and_value(obs_t)
                current_action = action.item()

            obs, reward, terminated, truncated, _ = env.step(current_action)
            cumulative_reward += reward
            step_count += 1
            if terminated or truncated:
                game_over = True

        # --- Draw ---
        screen.fill(bg_color)
        grid = env.grid

        for r in range(rows):
            for c in range(cols):
                x = c * cell_size
                y = r * cell_size
                rect = (x, y, cell_size - 1, cell_size - 1)

                val = grid[r, c]
                if val == CELL_HEAD:
                    pygame.draw.rect(screen, head_color, rect)
                elif val == CELL_BODY:
                    pygame.draw.rect(screen, body_color, rect)
                elif val == CELL_FOOD:
                    pygame.draw.rect(screen, food_color, rect)
                else:
                    pygame.draw.rect(screen, grid_color, rect)

        # --- HUD ---
        hud_lines = [
            f"Length: {len(env.snake)}",
            f"Score: {env.score}",
            f"Steps: {step_count}",
            f"Return: {cumulative_reward:.2f}",
        ]
        for i, line in enumerate(hud_lines):
            txt = hud_font.render(line, True, (255, 255, 255))
            screen.blit(txt, (8, 4 + i * 22))

        if game_over:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            big_font = pygame.font.SysFont(None, 48)
            if agent_model:
                msg = f"Game Over! Score: {env.score}  [Q]uit"
            else:
                msg = f"Game Over! Score: {env.score}  [R]estart  [Q]uit"
            txt = big_font.render(msg, True, (255, 255, 255))
            rect = txt.get_rect(center=(screen_w // 2, screen_h // 2))
            screen.blit(txt, rect)

        pygame.display.flip()
        clock.tick(args.fps if agent_model else 10)

        if game_over and agent_model:
            time.sleep(2)
            obs, _ = env.reset()
            game_over = False
            step_count = 0
            cumulative_reward = 0.0

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
