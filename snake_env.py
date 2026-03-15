"""Headless Snake gymnasium environment with grid and feature observation modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DIRECTION_VECTORS = {
    UP: np.array([0, -1]),
    DOWN: np.array([0, 1]),
    LEFT: np.array([-1, 0]),
    RIGHT: np.array([1, 0]),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

CELL_EMPTY = 0
CELL_BODY = 1
CELL_HEAD = 2
CELL_FOOD = 3


class SnakeEnv(gym.Env):
    """Snake game as a Gymnasium environment.

    Observation modes
    -----------------
    * ``"grid"``  – 2-D integer grid (rows × cols) with cell type IDs.
    * ``"features"`` – flat float vector: relative food direction (4),
      danger in three look-ahead directions (3), current direction one-hot (4).

    Actions
    -------
    Discrete(4): 0=up, 1=down, 2=left, 3=right.

    Rewards
    -------
    +1 for eating food, −1 for dying, −0.01 per step.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        rows: int = 12,
        cols: int = 12,
        obs_type: str = "grid",
        render_mode: str | None = None,
        max_steps_factor: int = 200,
    ) -> None:
        super().__init__()
        assert obs_type in ("grid", "features")
        self.rows = rows
        self.cols = cols
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.max_steps_factor = max_steps_factor

        self.action_space = spaces.Discrete(4)
        if obs_type == "grid":
            self.observation_space = spaces.Box(
                low=0, high=CELL_FOOD, shape=(rows, cols), dtype=np.int8
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )

        self._snake: list[np.ndarray] = []
        self._direction: int = RIGHT
        self._food: np.ndarray = np.zeros(2, dtype=np.intp)
        self._score: int = 0
        self._steps: int = 0
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=np.int8)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._grid[:] = CELL_EMPTY
        mid_r, mid_c = self.rows // 2, self.cols // 2
        self._snake = [np.array([mid_c, mid_r])]
        self._direction = RIGHT
        self._score = 0
        self._steps = 0
        self._place_food()
        self._update_grid()

        return self._get_obs(), {"score": self._score}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action != OPPOSITE.get(self._direction, -1):
            self._direction = action

        head = self._snake[0].copy()
        head += DIRECTION_VECTORS[self._direction]
        self._steps += 1

        terminated = False
        reward = -0.01

        if self._is_collision(head):
            terminated = True
            reward = -1.0
            self._update_grid()
            truncated = False
            return (
                self._get_obs(),
                reward,
                terminated,
                truncated,
                {"score": self._score},
            )

        self._snake.insert(0, head)

        if np.array_equal(head, self._food):
            self._score += 1
            reward = 1.0
            self._place_food()
        else:
            self._snake.pop()

        self._update_grid()

        max_steps = self.max_steps_factor * len(self._snake)
        truncated = self._steps >= max_steps

        return self._get_obs(), reward, terminated, truncated, {"score": self._score}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_collision(self, pos: np.ndarray) -> bool:
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return True
        return any(np.array_equal(pos, seg) for seg in self._snake)

    def _place_food(self) -> None:
        occupied = {tuple(s) for s in self._snake}
        free = [
            (c, r)
            for r in range(self.rows)
            for c in range(self.cols)
            if (c, r) not in occupied
        ]
        if not free:
            self._food = self._snake[0].copy()
            return
        idx = self.np_random.integers(len(free))
        self._food = np.array(free[idx], dtype=np.intp)

    def _update_grid(self) -> None:
        self._grid[:] = CELL_EMPTY
        for seg in self._snake[1:]:
            self._grid[seg[1], seg[0]] = CELL_BODY
        head = self._snake[0]
        if 0 <= head[0] < self.cols and 0 <= head[1] < self.rows:
            self._grid[head[1], head[0]] = CELL_HEAD
        self._grid[self._food[1], self._food[0]] = CELL_FOOD

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "grid":
            return self._grid.copy()
        return self._get_feature_obs()

    def _get_feature_obs(self) -> np.ndarray:
        head = self._snake[0]
        food_dir = np.zeros(4, dtype=np.float32)
        diff = self._food - head
        if diff[1] < 0:
            food_dir[UP] = 1.0
        if diff[1] > 0:
            food_dir[DOWN] = 1.0
        if diff[0] < 0:
            food_dir[LEFT] = 1.0
        if diff[0] > 0:
            food_dir[RIGHT] = 1.0

        dir_vec = DIRECTION_VECTORS[self._direction]
        left_vec = np.array([dir_vec[1], -dir_vec[0]])
        right_vec = np.array([-dir_vec[1], dir_vec[0]])

        danger = np.array(
            [
                self._is_collision(head + dir_vec),
                self._is_collision(head + left_vec),
                self._is_collision(head + right_vec),
            ],
            dtype=np.float32,
        )

        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[self._direction] = 1.0

        return np.concatenate([food_dir, danger, direction_onehot])

    def _render_rgb(self) -> np.ndarray:
        cell = 10
        img = np.zeros((self.rows * cell, self.cols * cell, 3), dtype=np.uint8)
        img[:] = [10, 10, 40]

        for r in range(self.rows):
            for c in range(self.cols):
                val = self._grid[r, c]
                if val == CELL_BODY:
                    color = [128, 189, 38]
                elif val == CELL_HEAD:
                    color = [4, 74, 0]
                elif val == CELL_FOOD:
                    color = [213, 50, 80]
                else:
                    continue
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                img[y0:y1, x0:x1] = color

        return img

    @property
    def grid(self) -> np.ndarray:
        """Expose internal grid for external renderers (e.g. ``play.py``)."""
        return self._grid

    @property
    def score(self) -> int:
        return self._score

    @property
    def snake(self) -> list[np.ndarray]:
        return self._snake

    @property
    def food(self) -> np.ndarray:
        return self._food
