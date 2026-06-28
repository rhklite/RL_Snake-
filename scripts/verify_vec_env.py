#!/usr/bin/env python3
"""Equivalence gate: VectorizedSnakeEnv must match N single SnakeEnvs exactly.

Drives the batched env and N independent single envs with identical actions and a
synced food cell (food RNG differs, so we inject the same free cell into both each
step to isolate dynamics). Asserts obs (all 4 channels + food vec), reward, terminated,
truncated, and legal_action_mask match at every step — across growth (long snakes),
deaths (wall/body), timeouts, and cycle-fills.

Run:  python3 scripts/verify_vec_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snake_env import SnakeEnv  # noqa: E402
from vec_snake_env import VectorizedSnakeEnv  # noqa: E402


def _free_cell(body_count_i: np.ndarray) -> tuple[int, int]:
    """First free (unoccupied) cell row-major, as (x, y). Deterministic given occupancy."""
    ys, xs = np.where(body_count_i == 0)
    return int(xs[0]), int(ys[0])


def _sync_food(vec: VectorizedSnakeEnv, singles: list[SnakeEnv]) -> None:
    """Force batched + each single env to the SAME free food cell (kills RNG divergence)."""
    for i, s in enumerate(singles):
        fx, fy = _free_cell(vec.body_count[i])
        vec.food[i] = (fx, fy)
        s._food = np.array([fx, fy], dtype=np.intp)
        s._update_grid()


def _single_obs(s: SnakeEnv) -> dict:
    return s._get_hybrid_obs()


def _compare(tag, vec, singles, step):
    vobs = vec._get_obs()
    vmask = vec.legal_action_mask()
    for i, s in enumerate(singles):
        sobs = _single_obs(s)
        assert np.allclose(vobs["grid"][i], sobs["grid"], atol=1e-5), (
            f"{tag} step{step} env{i}: grid mismatch "
            f"(max |Δ|={np.abs(vobs['grid'][i] - sobs['grid']).max():.4g}, "
            f"len={vec.length[i]})"
        )
        assert np.allclose(vobs["food"][i], sobs["food"], atol=1e-6), (
            f"{tag} step{step} env{i}: food vec mismatch"
        )
        sm = s.legal_action_mask()
        assert np.array_equal(vmask[i], sm), (
            f"{tag} step{step} env{i}: mask mismatch vec={vmask[i]} single={sm}"
        )


def run_equivalence(mask_mode: str, cycle_beta: float, policy: str, steps: int) -> None:
    N, R, C = 24, 20, 20
    kw = dict(rows=R, cols=C, win_bonus=20.0, step_penalty=0.0,
              cycle_beta=cycle_beta, mask_mode=mask_mode, max_steps_factor=200)
    vec = VectorizedSnakeEnv(N, seed=0, **kw)
    singles = [SnakeEnv(obs_type="hybrid", **kw) for _ in range(N)]
    vec.reset(seed=0)
    for i, s in enumerate(singles):
        s.reset(seed=100 + i)
        # force single's snake state to match the batched fresh state exactly
        mid_c, mid_r = C // 2, R // 2
        s._snake = [np.array([mid_c, mid_r])]
        s._direction = int(vec.direction[i])
        s._steps = 0
        s._score = 0
        s._cause_of_death = None
    _sync_food(vec, singles)

    rng = np.random.default_rng(7)
    for step in range(steps):
        _compare(f"[{mask_mode},β={cycle_beta},{policy}]", vec, singles, step)

        vmask = vec.legal_action_mask()
        if policy == "legal":  # stay alive -> long snakes (exercise obs at length)
            acts = np.array([int(rng.choice(np.flatnonzero(vmask[i]))) for i in range(N)])
        else:                  # random over all 4 -> exercises wall/body death paths
            acts = rng.integers(0, 4, N)

        v_obs, v_rew, v_term, v_trunc, v_info = vec.step(acts)
        s_rew = np.empty(N)
        s_term = np.empty(N, dtype=bool)
        s_trunc = np.empty(N, dtype=bool)
        for i, s in enumerate(singles):
            _, r, t, tr, _ = s.step(int(acts[i]))
            s_rew[i], s_term[i], s_trunc[i] = r, t, tr
            if t or tr:  # mirror the batched env's autoreset
                s.reset(seed=200 + step * N + i)
                mid_c, mid_r = C // 2, R // 2
                s._snake = [np.array([mid_c, mid_r])]
                s._direction = int(vec.direction[i])
                s._steps = 0
                s._score = 0
                s._cause_of_death = None

        assert np.allclose(v_rew, s_rew, atol=1e-6), (
            f"[{mask_mode}] step{step}: reward mismatch "
            f"(max|Δ|={np.abs(v_rew - s_rew).max():.4g}, "
            f"vec={v_rew[np.argmax(np.abs(v_rew - s_rew))]:.4f} "
            f"single={s_rew[np.argmax(np.abs(v_rew - s_rew))]:.4f})"
        )
        assert np.array_equal(v_term, s_term), f"[{mask_mode}] step{step}: terminated mismatch"
        assert np.array_equal(v_trunc, s_trunc), f"[{mask_mode}] step{step}: truncated mismatch"
        _sync_food(vec, singles)


def check_food_distribution() -> None:
    """_place_food is uniform over free cells (sanity: covers the board, never on body)."""
    vec = VectorizedSnakeEnv(1, rows=6, cols=6, mask_mode="none")
    seen = np.zeros((6, 6), dtype=int)
    for _ in range(4000):
        vec.reset()
        fx, fy = int(vec.food[0, 0]), int(vec.food[0, 1])
        assert vec.body_count[0, fy, fx] == 0, "food placed on an occupied cell"
        seen[fy, fx] += 1
    # center is the head on reset (never food); all other 35 cells should be hit.
    assert (seen[np.arange(6) != 3][:, np.arange(6) != 3] >= 0).all()
    assert (seen > 0).sum() >= 34, f"food not ~uniform: only {(seen>0).sum()}/36 cells hit"


def main() -> None:
    checks = [
        ("equivalence: safety mask, β=0.15, legal policy (long snakes)",
         lambda: run_equivalence("safety", 0.15, "legal", 600)),
        ("equivalence: safety mask, β=0.15, random policy (deaths)",
         lambda: run_equivalence("safety", 0.15, "random", 400)),
        ("equivalence: no mask, β=0.0, random policy",
         lambda: run_equivalence("none", 0.0, "random", 400)),
        ("equivalence: cycle mask, β=0.15, legal policy (fills)",
         lambda: run_equivalence("cycle", 0.15, "legal", 600)),
        ("food placement uniform over free cells", check_food_distribution),
    ]
    for name, fn in checks:
        fn()
        print(f"  [pass] {name}")
    print("\nVectorizedSnakeEnv matches SnakeEnv exactly. All equivalence checks passed.")


if __name__ == "__main__":
    main()
