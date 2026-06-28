"""Headless Snake gymnasium environment with grid, feature, and hybrid observation modes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numba as nb
import numpy as np
from gymnasium import spaces


@nb.njit(cache=True)
def _bfs_reachability(
    body_clearance: np.ndarray,
    distance: np.ndarray,
    hy: int,
    hx: int,
    rows: int,
    cols: int,
) -> None:
    """Numba-compiled BFS with forward-looking body clearance.

    Modifies ``distance`` in-place. Uses a pre-allocated numpy ring buffer
    as the BFS queue since Numba cannot handle Python deque.
    """
    max_q = rows * cols
    queue_y = np.empty(max_q, dtype=np.int32)
    queue_x = np.empty(max_q, dtype=np.int32)
    head_ptr = 0
    tail_ptr = 0

    if 0 <= hy < rows and 0 <= hx < cols:
        distance[hy, hx] = 0
        queue_y[tail_ptr] = hy
        queue_x[tail_ptr] = hx
        tail_ptr += 1

    dy = np.array([-1, 1, 0, 0], dtype=np.int32)
    dx = np.array([0, 0, -1, 1], dtype=np.int32)

    while head_ptr < tail_ptr:
        cy = queue_y[head_ptr]
        cx = queue_x[head_ptr]
        head_ptr += 1
        d = distance[cy, cx] + 1
        for k in range(4):
            ny = cy + dy[k]
            nx = cx + dx[k]
            if 0 <= ny < rows and 0 <= nx < cols and distance[ny, nx] == -1:
                if body_clearance[ny, nx] <= d:
                    distance[ny, nx] = d
                    queue_y[tail_ptr] = ny
                    queue_x[tail_ptr] = nx
                    tail_ptr += 1


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DIRECTION_VECTORS = {
    UP: np.array([0, -1]),
    DOWN: np.array([0, 1]),
    LEFT: np.array([-1, 0]),
    RIGHT: np.array([1, 0]),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

VEC2ACT = {(0, -1): UP, (0, 1): DOWN, (-1, 0): LEFT, (1, 0): RIGHT}


def _comb_order_rows_even(rows: int, cols: int) -> list[tuple[int, int]]:
    """Cell visit order for a 'comb' Hamiltonian cycle, requires rows even.

    Top row left->right, interior rows boustrophedon over columns 1..cols-1,
    then the column-0 spine bottom->top closes the loop back to (0, 0).
    """
    order = [(x, 0) for x in range(cols)]
    for y in range(1, rows):
        xs = range(cols - 1, 0, -1) if (y % 2) == 1 else range(1, cols)
        order.extend((x, y) for x in xs)
    order.extend((0, y) for y in range(rows - 1, 0, -1))
    return order


def build_cycle_order(rows: int, cols: int) -> list[tuple[int, int]]:
    """Cell visit order along the comb Hamiltonian cycle. Raises on both-odd grids."""
    if rows % 2 == 0:
        order = _comb_order_rows_even(rows, cols)
    elif cols % 2 == 0:
        order = [(y, x) for (x, y) in _comb_order_rows_even(cols, rows)]
    else:
        raise ValueError(
            f"No Hamiltonian cycle for both-odd grid {rows}x{cols} (area is odd)"
        )
    n = rows * cols
    assert len(order) == n and len(set(order)) == n, "cycle order is not a permutation"
    return order


def build_cycle_phase(rows: int, cols: int) -> np.ndarray:
    """Normalized position (0 <= p < 1) of each cell along the comb Hamiltonian cycle.

    A static, grid-only field used as a privileged CRITIC input in the asymmetric
    actor-critic: it hands the value function the canonical traversal order so it can
    value "is this state heading into a self-trap" without the actor ever seeing it.
    """
    order = build_cycle_order(rows, cols)
    n = rows * cols
    phase = np.zeros((rows, cols), dtype=np.float32)
    for i, (x, y) in enumerate(order):
        phase[y, x] = i / n
    return phase


def build_cycle_succ(rows: int, cols: int) -> np.ndarray:
    """Successor-direction field for a comb Hamiltonian cycle on an even grid.

    Returns an int8 ``(rows, cols)`` array where ``succ[y, x]`` is the action
    (UP/DOWN/LEFT/RIGHT) that advances one cell along a fixed Hamiltonian cycle.
    Raises ``ValueError`` on a both-odd grid (no cycle exists, area is odd).
    Self-asserts validity (permutation, 4-adjacency, closed tour) and fails
    visibly on any violation.
    """
    order = build_cycle_order(rows, cols)
    n = rows * cols

    succ = np.full((rows, cols), -1, dtype=np.int8)
    for i, (x, y) in enumerate(order):
        nx, ny = order[(i + 1) % n]
        dx, dy = nx - x, ny - y
        assert abs(dx) + abs(dy) == 1, f"non-adjacent cycle step {(x, y)}->{(nx, ny)}"
        succ[y, x] = VEC2ACT[(dx, dy)]
    assert (succ != -1).all(), "cycle successor field has holes"

    # Independent closed-tour walk: visit all n cells once and return to (0, 0).
    cx, cy, seen = 0, 0, set()
    for _ in range(n):
        assert (cx, cy) not in seen, "cycle revisits a cell"
        seen.add((cx, cy))
        dx, dy = DIRECTION_VECTORS[int(succ[cy, cx])]
        cx, cy = cx + dx, cy + dy
    assert (cx, cy) == (0, 0) and len(seen) == n, "cycle did not close over all cells"
    return succ


CELL_EMPTY = 0
CELL_BODY = 1
CELL_HEAD = 2
CELL_FOOD = 3


class SnakeEnv(gym.Env):
    """Snake game as a Gymnasium environment.

    Observation modes
    -----------------
    * ``"grid"``     – 2-D integer grid (rows × cols) with cell type IDs.
    * ``"features"`` – flat float vector: relative food direction (4),
      danger in three look-ahead directions (3), current direction one-hot (4).
    * ``"hybrid"``   – Dict obs with two keys:
        - ``"grid"``: 4-channel float32 grid ``(4, rows, cols)``:
          ch0 = body gradient (tail→head normalized 0→1),
          ch1 = head location (binary), ch2 = food location (binary),
          ch3 = BFS reachability (inverted distance from head, 0=unreachable).
        - ``"food"``: normalized signed food offset ``(Δx/cols, Δy/rows) float32``.

    Actions
    -------
    Discrete(4): 0=up, 1=down, 2=left, 3=right.

    Rewards
    -------
    +1 for eating food, −1 for dying, ``step_penalty`` per step (default −0.025).
    +``win_bonus`` when the snake fills the entire grid (default 0; coverage goal).
    Optional distance shaping: +alpha*(prev_dist - curr_dist) per step (default alpha=0).
    Optional Hamiltonian cycle shaping: +``cycle_beta`` for taking the on-cycle
    successor move (bonus-only, never penalized; default 0 = disabled).

    For the full-coverage goal (see ``docs/full-coverage-design.md``), set
    ``step_penalty`` near 0, ``win_bonus`` high, and leave distance shaping off.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        rows: int = 12,
        cols: int = 12,
        obs_type: str = "grid",
        render_mode: str | None = None,
        max_steps_factor: int = 200,
        render_cell_size: int = 60,  # 12x12 -> 720x720 (720p)
        dist_shaping_alpha: float = 0.0,  # distance shaping coefficient; 0 = disabled
        step_penalty: float = -0.025,  # per-step reward; ~0 for coverage goal
        win_bonus: float = 0.0,  # terminal reward for filling the whole grid
        cycle_beta: float = 0.0,  # +beta for the on-cycle move, 0 otherwise; 0 disables
        mask_mode: str = "none",  # action masking: "none" | "safety" | "cycle"
    ) -> None:
        super().__init__()
        assert obs_type in ("grid", "features", "hybrid")
        assert mask_mode in ("none", "safety", "cycle")
        self.rows = rows
        self.cols = cols
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.max_steps_factor = max_steps_factor
        self.render_cell_size = render_cell_size
        self.dist_shaping_alpha = dist_shaping_alpha
        self.step_penalty = step_penalty
        self.win_bonus = win_bonus
        self.cycle_beta = cycle_beta
        self.mask_mode = mask_mode
        # Precompute the Hamiltonian cycle successor field once (depends only on
        # grid size). None when neither shaping nor cycle-masking needs it, so
        # default behavior is unchanged.
        self._cycle_succ = (
            build_cycle_succ(rows, cols)
            if cycle_beta != 0.0 or mask_mode == "cycle"
            else None
        )

        self.action_space = spaces.Discrete(4)
        if obs_type == "grid":
            self.observation_space = spaces.Box(
                low=0, high=CELL_FOOD, shape=(rows, cols), dtype=np.int8
            )
        elif obs_type == "features":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )
        else:  # hybrid
            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0.0, high=1.0, shape=(4, rows, cols), dtype=np.float32
                    ),
                    "food": spaces.Box(
                        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                    ),
                }
            )

        self._snake: list[np.ndarray] = []
        self._direction: int = RIGHT
        self._food: np.ndarray = np.zeros(2, dtype=np.intp)
        self._score: int = 0
        self._steps: int = 0
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self._cause_of_death: str | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        self._grid[:] = CELL_EMPTY
        mid_r, mid_c = self.rows // 2, self.cols // 2
        self._snake = [np.array([mid_c, mid_r])]
        self._direction = RIGHT
        if self.mask_mode == "cycle":
            # Align the initial heading to the cycle successor at the start cell so
            # the opposite-guard never derails the first (cycle-masked) action; after
            # the first on-cycle step the successor is never the reverse of heading.
            self._direction = int(self._cycle_succ[mid_r, mid_c])
        self._score = 0
        self._steps = 0
        self._cause_of_death = None
        self._place_food()
        self._update_grid()

        return self._get_obs(), {"score": self._score}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action != OPPOSITE.get(self._direction, -1):
            self._direction = action

        head = self._snake[0].copy()

        # Record distance before move for shaping
        prev_dist = (
            abs(int(head[0]) - int(self._food[0]))
            + abs(int(head[1]) - int(self._food[1]))
            if self.dist_shaping_alpha != 0.0
            else 0
        )

        # Hamiltonian cycle-alignment bonus: read the PRE-move head and the
        # resolved heading (post opposite-guard). Added only on the surviving
        # non-win path below, so death/win signals stay clean.
        cyc_bonus = 0.0
        if self._cycle_succ is not None:
            if self._direction == int(self._cycle_succ[int(head[1]), int(head[0])]):
                cyc_bonus = self.cycle_beta

        head += DIRECTION_VECTORS[self._direction]
        self._steps += 1

        terminated = False
        reward = self.step_penalty

        if self._is_collision(head):
            terminated = True
            reward = -1.0
            # Distinguish wall vs body collision
            x, y = head
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                self._cause_of_death = "wall"
            else:
                self._cause_of_death = "body"
            self._update_grid()
            return (
                self._get_obs(),
                reward,
                terminated,
                False,
                {
                    "score": self._score,
                    "cause_of_death": self._cause_of_death,
                    "snake_length": len(self._snake),
                    "coverage": len(self._snake) / (self.rows * self.cols),
                },
            )

        self._snake.insert(0, head)

        if np.array_equal(head, self._food):
            self._score += 1
            reward = 1.0
            if len(self._snake) >= self.rows * self.cols:
                # Grid fully covered: win. No free cell remains to place food.
                terminated = True
                reward += self.win_bonus
                self._cause_of_death = "win"
            else:
                self._place_food()
        else:
            self._snake.pop()
            # Distance shaping only on non-eating steps (food position unchanged)
            if self.dist_shaping_alpha != 0.0:
                curr_dist = abs(int(head[0]) - int(self._food[0])) + abs(
                    int(head[1]) - int(self._food[1])
                )
                reward += self.dist_shaping_alpha * (prev_dist - curr_dist)

        # Surviving path only: excludes the collision early-return (reward -1.0)
        # and the win branch (terminated, reward 1.0+win_bonus). On a food-eating
        # non-win step reward becomes 1.0 + cyc_bonus.
        if not terminated:
            reward += cyc_bonus

        self._update_grid()

        max_steps = self.max_steps_factor * len(self._snake)
        truncated = (not terminated) and self._steps >= max_steps
        if truncated:
            self._cause_of_death = "timeout"

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "score": self._score,
                "cause_of_death": self._cause_of_death,
                "snake_length": len(self._snake),
                "coverage": len(self._snake) / (self.rows * self.cols),
            },
        )

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

    def legal_action_mask(self) -> np.ndarray:
        """Boolean ``(4,)`` mask over actions {UP, DOWN, LEFT, RIGHT}; True = allowed.

        * ``"none"``   – all actions allowed (identity; default behavior).
        * ``"safety"`` – mask any action whose **resolved** move (after the
          opposite-direction guard in :meth:`step`) steps into a wall or an
          occupied cell, i.e. exactly the moves the env scores as a death this
          step. If *every* action is fatal (an unavoidable trap), return
          all-allowed so the policy never faces an all-masked (NaN) distribution.
        * ``"cycle"``  – allow only the Hamiltonian-cycle successor at the head.

        The occupancy set includes the tail, matching :meth:`_is_collision` at
        collision-check time (the tail is popped only after the check), so the
        mask is consistent with the env's actual lethality.
        """
        mask = np.ones(4, dtype=bool)
        if self.mask_mode == "none":
            return mask

        head = self._snake[0]
        hx, hy = int(head[0]), int(head[1])

        if self.mask_mode == "cycle":
            mask[:] = False
            mask[int(self._cycle_succ[hy, hx])] = True
            return mask

        # safety
        occupied = {(int(s[0]), int(s[1])) for s in self._snake}
        opp = OPPOSITE.get(self._direction, -1)
        for a in range(4):
            resolved = self._direction if a == opp else a
            dx, dy = DIRECTION_VECTORS[resolved]
            nx, ny = hx + int(dx), hy + int(dy)
            fatal = (
                nx < 0
                or nx >= self.cols
                or ny < 0
                or ny >= self.rows
                or (nx, ny) in occupied
            )
            mask[a] = not fatal
        if not mask.any():
            mask[:] = True  # unavoidable death: don't emit an all-(-inf) row
        return mask

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

    def _get_obs(self) -> Any:
        if self.obs_type == "grid":
            return self._grid.copy()
        if self.obs_type == "features":
            return self._get_feature_obs()
        return self._get_hybrid_obs()

    def _get_hybrid_obs(self) -> dict[str, np.ndarray]:
        grid = np.zeros((4, self.rows, self.cols), dtype=np.float32)
        n = len(self._snake)
        for i, seg in enumerate(self._snake):
            x, y = int(seg[0]), int(seg[1])
            if 0 <= x < self.cols and 0 <= y < self.rows:
                grid[0, y, x] = 0.1 + 0.9 * (n - 1 - i) / max(n - 1, 1)
        head = self._snake[0]
        hx, hy = int(head[0]), int(head[1])
        if 0 <= hx < self.cols and 0 <= hy < self.rows:
            grid[1, hy, hx] = 1.0
        grid[2, int(self._food[1]), int(self._food[0])] = 1.0
        scale = min(1.0, max(n - 1, 0) / 50)
        grid[3] = self._compute_reachability(n, hx, hy) * scale
        food_vec = np.array(
            [
                (self._food[0] - head[0]) / max(self.cols - 1, 1),
                (self._food[1] - head[1]) / max(self.rows - 1, 1),
            ],
            dtype=np.float32,
        )
        return {"grid": grid, "food": food_vec}

    def _compute_reachability(self, n: int, hx: int, hy: int) -> np.ndarray:
        """BFS reachability from head with forward-looking body clearance.

        Delegates to a Numba-compiled BFS. Returns an inverted distance map
        normalized by the max distance observed this step so the gradient
        always spans 0.0 (farthest reachable / unreachable) to 1.0 (head).
        """
        rows, cols = self.rows, self.cols

        body_clearance = np.zeros((rows, cols), dtype=np.int32)
        for i, seg in enumerate(self._snake):
            sx, sy = int(seg[0]), int(seg[1])
            if 0 <= sx < cols and 0 <= sy < rows:
                body_clearance[sy, sx] = n - i

        distance = np.full((rows, cols), -1, dtype=np.int32)
        _bfs_reachability(body_clearance, distance, hy, hx, rows, cols)

        reachability = np.zeros((rows, cols), dtype=np.float32)
        mask = distance >= 0
        max_dist = distance[mask].max() if mask.any() else 1
        if max_dist == 0:
            max_dist = 1
        reachability[mask] = 1.0 - distance[mask].astype(np.float32) / max_dist
        return reachability

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
        cell = self.render_cell_size
        img = np.zeros((self.rows * cell, self.cols * cell, 3), dtype=np.uint8)
        img[:] = [255, 255, 255]

        for r in range(self.rows):
            for c in range(self.cols):
                val = self._grid[r, c]
                if val == CELL_BODY:
                    color = [128, 189, 38]
                elif val == CELL_HEAD:
                    color = [10, 10, 40]
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

    @property
    def cause_of_death(self) -> str | None:
        """Returns 'wall', 'body', 'timeout', or None if episode is ongoing."""
        return self._cause_of_death
