"""Human-playable Snake using SnakeEnv with pygame rendering.

This is the only file that imports pygame. The environment itself stays
headless — this script reads the env's grid state and draws it.
"""

from __future__ import annotations

import sys

import pygame
import yaml

from snake_env import CELL_BODY, CELL_FOOD, CELL_HEAD, DOWN, LEFT, RIGHT, UP, SnakeEnv


def load_settings(path: str = "settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    settings = load_settings()
    gui_cfg = settings["gui"]
    world = gui_cfg["world"]
    colors = gui_cfg["color"]

    rows = world["rows"]
    cols = world["columns"]
    cell_size = world["cell_size"]

    bg_color = tuple(colors["background"][:3])
    grid_color = tuple(colors["grid"][:3])
    head_color = tuple(colors["snake"]["head"])
    body_color = tuple(colors["snake"]["body"])
    food_color = (213, 50, 80)

    env = SnakeEnv(rows=rows, cols=cols, obs_type="grid", render_mode=None)
    env.reset()

    pygame.init()
    screen_w = cols * cell_size
    screen_h = rows * cell_size
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(gui_cfg.get("display_app_name", "Snake Game"))
    clock = pygame.time.Clock()

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

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game_over:
                    env.reset()
                    current_action = RIGHT
                    game_over = False
                elif event.key in key_map and not game_over:
                    current_action = key_map[event.key]

        if not game_over:
            _, _, terminated, truncated, _ = env.step(current_action)
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

        if game_over:
            font = pygame.font.SysFont(None, 48)
            txt = font.render(
                f"Game Over! Score: {env.score}  [R]estart  [Q]uit",
                True,
                (255, 255, 255),
            )
            rect = txt.get_rect(center=(screen_w // 2, screen_h // 2))
            screen.blit(txt, rect)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
