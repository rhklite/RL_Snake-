from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTION_DELTAS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv(gymnasium.Env):
    """Gymnasium-compatible Snake environment with feature-vector observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        grid_rows: int = 12,
        grid_cols: int = 12,
        render_mode: str | None = None,
        cell_size: int = 50,
    ):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.max_steps_without_food = grid_rows * grid_cols

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.snake: deque[tuple[int, int]] = deque()
        self.direction: int = RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.steps_since_food: int = 0
        self.total_steps: int = 0
        self.score: int = 0

        self._screen = None
        self._clock = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        mid_x = self.grid_cols // 2
        mid_y = self.grid_rows // 2
        self.snake = deque([(mid_x - 2, mid_y), (mid_x - 1, mid_y), (mid_x, mid_y)])
        self.direction = RIGHT
        self.steps_since_food = 0
        self.total_steps = 0
        self.score = 0

        self._place_food()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if OPPOSITE.get(action) != self.direction:
            self.direction = action

        dx, dy = DIRECTION_DELTAS[self.direction]
        head_x, head_y = self.snake[-1]
        new_head = (head_x + dx, head_y + dy)

        terminated = False
        truncated = False
        reward = 0.0

        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_cols
            or new_head[1] < 0
            or new_head[1] >= self.grid_rows
        ):
            terminated = True
            reward = -1.0
            self.total_steps += 1
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if new_head in self.snake:
            terminated = True
            reward = -1.0
            self.total_steps += 1
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        self.snake.append(new_head)

        if new_head == self.food:
            reward = 1.0
            self.score += 1
            self.steps_since_food = 0
            if len(self.snake) < self.grid_rows * self.grid_cols:
                self._place_food()
            else:
                self.total_steps += 1
                return (
                    self._get_obs(),
                    reward,
                    True,
                    False,
                    self._get_info(),
                )
        else:
            self.snake.popleft()
            self.steps_since_food += 1
            if new_dist < old_dist:
                reward = 0.1
            else:
                reward = -0.1

        self.total_steps += 1

        if self.steps_since_food >= self.max_steps_without_food:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _place_food(self) -> None:
        snake_set = set(self.snake)
        available = [
            (x, y)
            for x in range(self.grid_cols)
            for y in range(self.grid_rows)
            if (x, y) not in snake_set
        ]
        self.food = available[self.np_random.integers(len(available))]

    def _get_obs(self) -> np.ndarray:
        head_x, head_y = self.snake[-1]
        max_x = self.grid_cols - 1
        max_y = self.grid_rows - 1

        norm_head_x = head_x / max_x if max_x > 0 else 0.0
        norm_head_y = head_y / max_y if max_y > 0 else 0.0
        norm_food_x = self.food[0] / max_x if max_x > 0 else 0.0
        norm_food_y = self.food[1] / max_y if max_y > 0 else 0.0

        food_dir_x = (self.food[0] - head_x) / max_x if max_x > 0 else 0.0
        food_dir_y = (self.food[1] - head_y) / max_y if max_y > 0 else 0.0
        food_dir_x = np.clip((food_dir_x + 1.0) / 2.0, 0.0, 1.0)
        food_dir_y = np.clip((food_dir_y + 1.0) / 2.0, 0.0, 1.0)

        snake_set = set(self.snake)

        def _is_danger(dx: int, dy: int) -> float:
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= self.grid_cols or ny < 0 or ny >= self.grid_rows:
                return 1.0
            if (nx, ny) in snake_set:
                return 1.0
            return 0.0

        danger_up = _is_danger(0, -1)
        danger_down = _is_danger(0, 1)
        danger_left = _is_danger(-1, 0)
        danger_right = _is_danger(1, 0)

        max_len = self.grid_rows * self.grid_cols
        norm_length = len(self.snake) / max_len
        norm_hunger = self.steps_since_food / self.max_steps_without_food

        return np.array(
            [
                norm_head_x,
                norm_head_y,
                norm_food_x,
                norm_food_y,
                food_dir_x,
                food_dir_y,
                danger_up,
                danger_down,
                danger_left,
                danger_right,
                norm_length,
                norm_hunger,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.total_steps,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            return None

        w = self.grid_cols * self.cell_size
        h = self.grid_rows * self.cell_size

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Snake RL")
            else:
                self._screen = pygame.Surface((w, h))
            self._clock = pygame.time.Clock()

        self._screen.fill((10, 10, 40))

        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self._screen, (30, 30, 60), rect, 1)

        cs = self.cell_size
        for i, (sx, sy) in enumerate(self.snake):
            rect = pygame.Rect(sx * cs, sy * cs, cs, cs)
            if i == len(self.snake) - 1:
                pygame.draw.rect(self._screen, (4, 74, 0), rect)
            else:
                pygame.draw.rect(self._screen, (128, 189, 38), rect)

        food_rect = pygame.Rect(self.food[0] * cs, self.food[1] * cs, cs, cs)
        pygame.draw.rect(self._screen, (213, 50, 80), food_rect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self._screen is not None:
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass
            self._screen = None
            self._clock = None
