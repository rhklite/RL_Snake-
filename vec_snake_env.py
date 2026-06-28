"""Batched (vectorized) Snake environment — steps N games at once with array ops.

Behaviorally equivalent to ``snake_env.SnakeEnv`` (hybrid obs) but holds all N envs'
state as ``[N, ...]`` arrays and advances them with vectorized numpy, eliminating the
per-env Python loop (gym.vector.SyncVectorEnv) AND the per-segment Python loops in the
single env's obs construction. The snake is represented as a ``body_count[N, R, C]``
grid (0 = empty, k>0 = steps until that cell is vacated; head = length, tail = 1). That
grid IS the BFS ``body_clearance`` and makes occupancy/collision/body-gradient pure array
ops. Reachability reuses the exact single-env BFS, batched in numba (compiled, not Python).

Equivalence is enforced by scripts/verify_vec_env.py (batched vs N single envs, identical
actions + injected food -> identical obs/reward/done/mask).
"""

from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np

from snake_env import build_cycle_succ

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# (dx, dy) per action, grid indexed [y, x]; OPP[a] = reverse of a.
_DIRS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
_OPP = np.array([DOWN, UP, RIGHT, LEFT], dtype=np.int64)


@nb.njit(cache=True)
def _bfs_reachability_batched(
    body_count: np.ndarray,  # (N, R, C) int32 — = clearance (steps to vacate)
    heads: np.ndarray,       # (N, 2) int64 — (x, y) per env
    out: np.ndarray,         # (N, R, C) float32 — reachability written here
) -> None:
    """Per-env BFS reachability with forward-looking body clearance (batched in njit).

    Mirrors snake_env._bfs_reachability exactly, looping envs inside the compiled
    function. ``out`` is filled with normalized inverted distance (1.0 at head, 0.0 at
    the farthest reachable / unreachable cell).
    """
    n_envs, rows, cols = body_count.shape
    dy = np.array([-1, 1, 0, 0], dtype=np.int64)
    dx = np.array([0, 0, -1, 1], dtype=np.int64)
    for e in range(n_envs):
        distance = np.full((rows, cols), -1, dtype=np.int64)
        max_q = rows * cols
        qy = np.empty(max_q, dtype=np.int64)
        qx = np.empty(max_q, dtype=np.int64)
        head_ptr = 0
        tail_ptr = 0
        hx = heads[e, 0]
        hy = heads[e, 1]
        if 0 <= hy < rows and 0 <= hx < cols:
            distance[hy, hx] = 0
            qy[tail_ptr] = hy
            qx[tail_ptr] = hx
            tail_ptr += 1
        while head_ptr < tail_ptr:
            cy = qy[head_ptr]
            cx = qx[head_ptr]
            head_ptr += 1
            d = distance[cy, cx] + 1
            for k in range(4):
                ny = cy + dy[k]
                nx = cx + dx[k]
                if 0 <= ny < rows and 0 <= nx < cols and distance[ny, nx] == -1:
                    if body_count[e, ny, nx] <= d:
                        distance[ny, nx] = d
                        qy[tail_ptr] = ny
                        qx[tail_ptr] = nx
                        tail_ptr += 1
        # normalize: 1 - dist/max_dist over reachable cells (matches single env)
        max_dist = 0
        for y in range(rows):
            for x in range(cols):
                if distance[y, x] > max_dist:
                    max_dist = distance[y, x]
        if max_dist == 0:
            max_dist = 1
        for y in range(rows):
            for x in range(cols):
                if distance[y, x] >= 0:
                    out[e, y, x] = 1.0 - distance[y, x] / max_dist


class VectorizedSnakeEnv:
    """Batched Snake (hybrid obs only). API mirrors gym.vector.SyncVectorEnv enough for
    the PPO loop: ``reset() -> (obs, info)``, ``step(actions) -> (obs, rew, term, trunc,
    infos)`` with autoreset, plus ``legal_action_mask() -> (N, 4) bool``.
    """

    def __init__(
        self,
        num_envs: int,
        rows: int = 12,
        cols: int = 12,
        max_steps_factor: int = 200,
        step_penalty: float = -0.025,
        win_bonus: float = 0.0,
        cycle_beta: float = 0.0,
        mask_mode: str = "none",
        seed: int | None = None,
    ) -> None:
        assert mask_mode in ("none", "safety", "cycle")
        self.N = num_envs
        self.rows = rows
        self.cols = cols
        self.area = rows * cols
        self.max_steps_factor = max_steps_factor
        self.step_penalty = step_penalty
        self.win_bonus = win_bonus
        self.cycle_beta = cycle_beta
        self.mask_mode = mask_mode
        self.rng = np.random.default_rng(seed)
        self._cycle_succ = (
            build_cycle_succ(rows, cols)
            if cycle_beta != 0.0 or mask_mode == "cycle"
            else None
        )
        self._ei = np.arange(self.N)

        N, R, C = self.N, rows, cols
        self.body_count = np.zeros((N, R, C), dtype=np.int32)
        self.head = np.zeros((N, 2), dtype=np.int64)   # (x, y)
        self.direction = np.full(N, RIGHT, dtype=np.int64)
        self.food = np.zeros((N, 2), dtype=np.int64)   # (x, y)
        self.length = np.ones(N, dtype=np.int64)
        self.steps = np.zeros(N, dtype=np.int64)
        self.score = np.zeros(N, dtype=np.int64)
        self.ep_return = np.zeros(N, dtype=np.float64)
        self.ep_len = np.zeros(N, dtype=np.int64)

    # ------------------------------------------------------------------ helpers
    def _reset_envs(self, idx: np.ndarray) -> None:
        """Reset the envs in ``idx`` (1-D index array) to a fresh single-cell snake."""
        if idx.size == 0:
            return
        mid_c, mid_r = self.cols // 2, self.rows // 2
        self.body_count[idx] = 0
        self.body_count[idx, mid_r, mid_c] = 1
        self.head[idx] = (mid_c, mid_r)
        self.length[idx] = 1
        self.steps[idx] = 0
        self.score[idx] = 0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        if self.mask_mode == "cycle":
            self.direction[idx] = self._cycle_succ[mid_r, mid_c]
        else:
            self.direction[idx] = RIGHT
        self._place_food(idx)

    def _place_food(self, idx: np.ndarray) -> None:
        """Place food on a uniformly-random free cell for each env in ``idx``."""
        if idx.size == 0:
            return
        free = self.body_count[idx] == 0  # (k, R, C) ; head/body excluded (count>0)
        r = self.rng.random((idx.size, self.rows, self.cols))
        r[~free] = np.inf
        flat = r.reshape(idx.size, -1).argmin(axis=1)  # uniform free cell (argmin of iid uniforms)
        fy, fx = np.divmod(flat, self.cols)
        # if an env has no free cell (full board) argmin returns a body cell; that env has
        # already won so food is irrelevant. Keep the chosen cell.
        self.food[idx, 0] = fx
        self.food[idx, 1] = fy

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_envs(np.arange(self.N))
        return self._get_obs(), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        N = self.N
        ei = self._ei
        actions = np.asarray(actions, dtype=np.int64)

        # 1) opposite-direction guard: a reversal is ignored (keep current heading).
        keep = actions == _OPP[self.direction]
        self.direction = np.where(keep, self.direction, actions)

        hx, hy = self.head[:, 0], self.head[:, 1]
        # 2) cycle-alignment bonus: pre-move head + resolved heading (bonus-only).
        cyc_bonus = np.zeros(N, dtype=np.float64)
        if self._cycle_succ is not None:
            on_cycle = self.direction == self._cycle_succ[hy, hx]
            cyc_bonus = np.where(on_cycle, self.cycle_beta, 0.0)

        dv = _DIRS[self.direction]
        nx = hx + dv[:, 0]
        ny = hy + dv[:, 1]

        # 3) collision against OLD occupancy (tail still present, matches single env).
        wall = (nx < 0) | (nx >= self.cols) | (ny < 0) | (ny >= self.rows)
        cx = np.clip(nx, 0, self.cols - 1)
        cy = np.clip(ny, 0, self.rows - 1)
        body = (~wall) & (self.body_count[ei, cy, cx] > 0)
        dead = wall | body

        eat = (~dead) & (nx == self.food[:, 0]) & (ny == self.food[:, 1])
        alive = ~dead
        survive_noeat = alive & ~eat

        # 4) advance occupancy. non-eat survivors: tail vacates (decrement). then write head.
        dec = survive_noeat[:, None, None] & (self.body_count > 0)
        self.body_count = np.where(dec, self.body_count - 1, self.body_count)
        new_length = self.length + eat.astype(np.int64)
        live = alive
        self.body_count[ei[live], ny[live], nx[live]] = new_length[live]
        self.head[live, 0] = nx[live]
        self.head[live, 1] = ny[live]
        self.length = new_length

        win = eat & (self.length >= self.area)

        # 5) reward assembly (matches single env ordering exactly).
        reward = np.full(N, self.step_penalty, dtype=np.float64)
        reward[dead] = -1.0
        reward[eat & ~win] = 1.0
        reward[win] = 1.0 + self.win_bonus
        reward[alive & ~win] += cyc_bonus[alive & ~win]

        self.score += eat.astype(np.int64)
        self.steps += 1
        terminated = dead | win
        max_steps = self.max_steps_factor * self.length
        truncated = (~terminated) & (self.steps >= max_steps)

        # 6) cause codes: 1 wall, 2 body, 3 timeout, 4 win, 0 none.
        cause = np.zeros(N, dtype=np.int64)
        cause[wall] = 1
        cause[body & ~wall] = 2
        cause[truncated] = 3
        cause[win] = 4

        # 7) new food for survivors that ate (and didn't win).
        self._place_food(np.where(eat & ~win)[0])

        # 8) episode accounting (RecordEpisodeStatistics-equivalent).
        self.ep_return += reward
        self.ep_len += 1
        done = terminated | truncated
        infos: dict[str, Any] = {}
        if done.any():
            d_idx = np.where(done)[0]
            ep_mask = np.zeros(N, dtype=bool)
            ep_mask[d_idx] = True
            infos["_episode"] = ep_mask
            infos["episode"] = {
                "r": self.ep_return.astype(np.float64),
                "l": self.ep_len.astype(np.int64),
            }
            infos["_cause_of_death"] = ep_mask
            cod = np.array([""] * N, dtype=object)
            names = {1: "wall", 2: "body", 3: "timeout", 4: "win"}
            for i in d_idx:
                cod[i] = names.get(int(cause[i]), "")
            infos["cause_of_death"] = cod
            infos["snake_length"] = self.length.copy()
            infos["coverage"] = self.length / self.area
            # 9) autoreset done envs (return their fresh obs, like SyncVectorEnv).
            self._reset_envs(d_idx)

        return self._get_obs(), reward.astype(np.float32), terminated, truncated, infos

    # ------------------------------------------------------------------ obs / mask
    def _get_obs(self) -> dict[str, np.ndarray]:
        N, R, C = self.N, self.rows, self.cols
        grid = np.zeros((N, 4, R, C), dtype=np.float32)
        occ = self.body_count > 0
        L = np.maximum(self.length - 1, 1).astype(np.float32)[:, None, None]
        # ch0 body gradient: 0.1 (tail) -> 1.0 (head); 0.1 for a length-1 snake.
        grad = 0.1 + 0.9 * (self.body_count.astype(np.float32) - 1.0) / L
        grid[:, 0] = np.where(occ, grad, 0.0)
        ei = self._ei
        grid[ei, 1, self.head[:, 1], self.head[:, 0]] = 1.0          # ch1 head
        grid[ei, 2, self.food[:, 1], self.food[:, 0]] = 1.0          # ch2 food
        reach = np.zeros((N, R, C), dtype=np.float32)                # ch3 reachability
        _bfs_reachability_batched(self.body_count, self.head, reach)
        scale = np.minimum(1.0, np.maximum(self.length - 1, 0) / 50.0).astype(np.float32)
        grid[:, 3] = reach * scale[:, None, None]
        food_vec = np.empty((N, 2), dtype=np.float32)
        food_vec[:, 0] = (self.food[:, 0] - self.head[:, 0]) / max(C - 1, 1)
        food_vec[:, 1] = (self.food[:, 1] - self.head[:, 1]) / max(R - 1, 1)
        return {"grid": grid, "food": food_vec}

    def legal_action_mask(self) -> np.ndarray:
        """Batched (N, 4) bool mask; True = allowed. Matches SnakeEnv.legal_action_mask."""
        N = self.N
        mask = np.ones((N, 4), dtype=bool)
        if self.mask_mode == "none":
            return mask
        hx, hy = self.head[:, 0], self.head[:, 1]
        if self.mask_mode == "cycle":
            mask[:] = False
            mask[self._ei, self._cycle_succ[hy, hx]] = True
            return mask
        # safety: resolve the opposite-guard per action, then test wall/body collision.
        for a in range(4):
            resolved = np.where(a == _OPP[self.direction], self.direction, a)
            dv = _DIRS[resolved]
            nx = hx + dv[:, 0]
            ny = hy + dv[:, 1]
            wall = (nx < 0) | (nx >= self.cols) | (ny < 0) | (ny >= self.rows)
            cx = np.clip(nx, 0, self.cols - 1)
            cy = np.clip(ny, 0, self.rows - 1)
            fatal = wall | (self.body_count[self._ei, cy, cx] > 0)
            mask[:, a] = ~fatal
        # all-fatal fallback -> all-legal (no all-(-inf) row downstream).
        none_legal = ~mask.any(axis=1)
        mask[none_legal] = True
        return mask

    def close(self) -> None:  # API parity
        pass
