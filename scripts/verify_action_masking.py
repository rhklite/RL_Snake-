#!/usr/bin/env python3
"""Correctness gate for v15 action masking (mask_mode = none | safety | cycle).

Run before any v15 training:  python3 scripts/verify_action_masking.py
All checks raise (fail visibly) on violation; prints a per-check summary on success.

The load-bearing check is `check_safety_matches_env_death`: for many states it clones
the env, steps each action, and asserts the mask flags an action illegal IFF the env
scores that action as a wall/body death. If that holds, the mask is — by construction —
exactly the env's lethality, so masking can never remove a survivable move nor keep a
fatal one.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snake_env import (  # noqa: E402
    DIRECTION_VECTORS,
    OPPOSITE,
    SnakeEnv,
    build_cycle_succ,
)


def _is_death(env: SnakeEnv, action: int) -> bool:
    """Ground truth: does taking `action` from `env`'s state end in a wall/body death?"""
    clone = copy.deepcopy(env)
    _, _, terminated, _, _ = clone.step(action)
    return terminated and clone.cause_of_death in ("wall", "body")


def check_none_is_identity() -> None:
    """mask_mode='none' (default) always returns an all-True mask and builds no cycle."""
    env = SnakeEnv(rows=8, cols=8, obs_type="hybrid")
    env.reset(seed=0)
    assert env.mask_mode == "none"
    assert env._cycle_succ is None, "mask_mode=none must not build the cycle field"
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert env.legal_action_mask().all(), "none mask must be all-True"
        _, _, term, trunc, _ = env.step(int(rng.integers(4)))
        if term or trunc:
            env.reset(seed=0)


def check_safety_matches_env_death() -> None:
    """Safety mask flags action a illegal IFF the env scores a as a wall/body death."""
    for seed in range(6):
        env = SnakeEnv(rows=12, cols=12, obs_type="hybrid", mask_mode="safety")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(400):
            mask = env.legal_action_mask()
            for a in range(4):
                truth_fatal = _is_death(env, a)
                # all-fatal fallback intentionally breaks the equivalence (mask=all True
                # even though every move is fatal); handle it as its own check below.
                if not mask.any():
                    continue
                assert bool(mask[a]) == (not truth_fatal), (
                    f"seed={seed}: mask[{a}]={bool(mask[a])} but env-death={truth_fatal} "
                    f"at head={tuple(env.snake[0])} dir={env._direction} len={len(env.snake)}"
                )
            # survive longer (reach long-snake states) by walking a legal action.
            legal = np.flatnonzero(mask)
            a = int(rng.choice(legal)) if legal.size else int(rng.integers(4))
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset(seed=seed)


def check_opposite_guard_resolution() -> None:
    """The reverse action's legality mirrors continuing forward (guard resolves it)."""
    env = SnakeEnv(rows=10, cols=10, obs_type="hybrid", mask_mode="safety")
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(300):
        mask = env.legal_action_mask()
        d = env._direction
        opp = OPPOSITE[d]
        if mask.any():
            # forward (continue heading d) and the reverse action resolve to the same move.
            head = env.snake[0]
            fwd = head + DIRECTION_VECTORS[d]
            fwd_fatal = env._is_collision(fwd)
            assert bool(mask[opp]) == (not fwd_fatal), "reverse action not resolved to forward"
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal)) if legal.size else 0
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset(seed=3)


def check_all_fatal_fallback() -> None:
    """When every resolved move is fatal, the mask falls back to all-True (no NaN row)."""
    env = SnakeEnv(rows=5, cols=5, obs_type="hybrid", mask_mode="safety")
    env.reset(seed=0)
    # Box the head into corner (0,0): neighbors (1,0) and (0,1) are body, the other two
    # are walls -> all four resolved moves are fatal.
    env._snake = [
        np.array([0, 0]),  # head
        np.array([1, 0]),  # body blocks RIGHT
        np.array([0, 1]),  # body blocks DOWN
    ]
    env._direction = 3  # RIGHT (arbitrary; all moves fatal regardless)
    mask = env.legal_action_mask()
    assert mask.all(), f"all-fatal state must fall back to all-True, got {mask}"
    # And the ground truth confirms every action really is fatal here.
    for a in range(4):
        assert _is_death(env, a), f"constructed trap: action {a} should be fatal"


def check_cycle_mask_single_and_fills() -> None:
    """Cycle mask exposes exactly one legal action; following it fills the grid (win)."""
    for rows, cols in [(6, 6), (8, 8), (4, 4)]:
        env = SnakeEnv(
            rows=rows, cols=cols, obs_type="hybrid",
            step_penalty=0.0, win_bonus=10.0, mask_mode="cycle",
        )
        env.reset(seed=0)
        succ = build_cycle_succ(rows, cols)
        won = False
        # Filling needs up to ~N steps per food eaten (one near-full lap), so ~N^2 total.
        for _ in range((rows * cols) ** 2 + rows * cols):
            mask = env.legal_action_mask()
            assert mask.sum() == 1, f"{rows}x{cols}: cycle mask must allow exactly 1 action"
            a = int(np.flatnonzero(mask)[0])
            hx, hy = int(env.snake[0][0]), int(env.snake[0][1])
            assert a == int(succ[hy, hx]), "cycle mask action != cycle successor"
            _, _, term, trunc, _ = env.step(a)
            if term and env.cause_of_death == "win":
                won = True
                break
            assert not term, f"{rows}x{cols}: cycle-follow died (cause={env.cause_of_death})"
            assert not trunc, f"{rows}x{cols}: cycle-follow timed out"
        assert won, f"{rows}x{cols}: cycle mask did not reach a win"


def check_model_masked_distribution() -> None:
    """Masked logits -> finite entropy/logprob, ~0 prob on illegal, samples always legal."""
    import torch  # heavy; local import

    from model import make_agent

    rows = cols = 12
    agent = make_agent(
        arch="hybrid", obs_type="hybrid", rows=rows, cols=cols,
        hidden_size=128, num_layers=2, activation="relu", adaptive_pool_size=4,
    )
    b = 256
    rng = np.random.default_rng(0)
    obs = {
        "grid": torch.rand(b, 4, rows, cols),
        "food": torch.rand(b, 2) * 2 - 1,
    }
    # random masks with at least one legal action per row
    m = torch.from_numpy(rng.integers(0, 2, size=(b, 4)).astype(bool))
    no_legal = ~m.any(dim=1)
    m[no_legal, 0] = True

    with torch.no_grad():
        action, logprob, entropy, _ = agent.get_action_and_value(obs, action_mask=m)

    assert torch.isfinite(entropy).all(), "masked entropy is non-finite (NaN/inf)"
    assert torch.isfinite(logprob).all(), "masked logprob is non-finite"
    # sampled actions must always be legal
    assert m[torch.arange(b), action].all(), "sampled an illegal (masked) action"
    # entropy must not exceed log(#legal) for any row
    n_legal = m.sum(dim=1).float()
    assert (entropy <= torch.log(n_legal) + 1e-4).all(), "entropy exceeds log(#legal)"
    # illegal actions carry ~0 probability
    from torch.distributions import Categorical

    logits = agent.actor if False else None  # noqa: F841 (silence linter on unused)
    feats = agent._encode(obs)
    masked_logits = agent.actor(feats).masked_fill(~m, float("-inf"))
    probs = Categorical(logits=masked_logits).probs
    assert probs[~m].abs().max().item() < 1e-6, "illegal actions have non-zero probability"


def check_rollout_update_consistency() -> None:
    """Same obs+mask+action -> identical logprob in 'rollout' and 'update' calls (ratio=1)."""
    import torch

    from model import make_agent

    rows = cols = 12
    agent = make_agent(
        arch="hybrid", obs_type="hybrid", rows=rows, cols=cols,
        hidden_size=128, num_layers=2, activation="relu", adaptive_pool_size=4,
    )
    agent.eval()
    b = 128
    rng = np.random.default_rng(1)
    obs = {"grid": torch.rand(b, 4, rows, cols), "food": torch.rand(b, 2) * 2 - 1}
    m = torch.from_numpy(rng.integers(0, 2, size=(b, 4)).astype(bool))
    m[~m.any(dim=1), 0] = True

    with torch.no_grad():
        action, lp_roll, _, _ = agent.get_action_and_value(obs, action_mask=m)
        _, lp_upd, _, _ = agent.get_action_and_value(obs, action, action_mask=m)
    assert torch.allclose(lp_roll, lp_upd, atol=1e-6), (
        "rollout vs update logprob mismatch under identical inputs (ratio would be wrong)"
    )


def check_masked_backward_finite_grads() -> None:
    """The BACKWARD pass through the masked dist yields finite grads (catches -inf NaN).

    The entropy term's gradient is the trap: with a -inf fill, masked actions compute
    0 * log(0) = 0 * -inf = NaN, which poisons the weights after some updates. Stress it
    with the worst case (every row exactly ONE legal action) and the entropy term in the
    loss, exactly as the PPO update does.
    """
    import torch

    from model import make_agent

    rows = cols = 12
    for trial in range(5):
        agent = make_agent(
            arch="hybrid", obs_type="hybrid", rows=rows, cols=cols,
            hidden_size=128, num_layers=2, activation="relu", adaptive_pool_size=4,
        )
        rng = np.random.default_rng(trial)
        b = 512
        obs = {"grid": torch.rand(b, 4, rows, cols), "food": torch.rand(b, 2) * 2 - 1}
        # Worst case: exactly one legal action per row (max entropy-gradient stress).
        only = torch.from_numpy(rng.integers(0, 4, size=b))
        m = torch.zeros(b, 4, dtype=torch.bool)
        m[torch.arange(b), only] = True

        action, logprob, entropy, value = agent.get_action_and_value(obs, action_mask=m)
        # PPO-shaped loss: policy term (logprob) + entropy bonus + value — all backprop.
        loss = -logprob.mean() - 0.03 * entropy.mean() + 0.5 * value.mean()
        assert torch.isfinite(loss), f"trial {trial}: loss is non-finite"
        loss.backward()
        for name, p in agent.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"trial {trial}: non-finite grad in {name} "
                    f"(the -inf-mask NaN-gradient bug)"
                )


def check_warm_start_strict_load() -> None:
    """The 16x16 (C2) checkpoint loads strict into the unchanged hybrid agent at 20x20."""
    ckpt = (
        Path(__file__).resolve().parent.parent
        / "runs" / "0610_22_coverage-16x16-b15-w20" / "checkpoints" / "best.pt"
    )
    if not ckpt.exists():
        print(f"  [skip] warm-start load: {ckpt} not found")
        return
    import torch

    from model import make_agent

    agent = make_agent(
        arch="hybrid", obs_type="hybrid", rows=20, cols=20,
        hidden_size=128, num_layers=2, activation="relu", adaptive_pool_size=4,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state["agent"] if isinstance(state, dict) and "agent" in state else state
    agent.load_state_dict(state)  # strict; raises on any shape/key mismatch


def main() -> None:
    checks = [
        ("none mask is identity (default, no cycle built)", check_none_is_identity),
        ("safety mask == env death rule (clone-and-step)", check_safety_matches_env_death),
        ("opposite-guard resolution (reverse mirrors forward)", check_opposite_guard_resolution),
        ("all-fatal trap -> all-True fallback (no NaN row)", check_all_fatal_fallback),
        ("cycle mask: single legal action, fills to win", check_cycle_mask_single_and_fills),
        ("model: finite entropy/logprob, legal-only samples", check_model_masked_distribution),
        ("rollout/update logprob consistency (ratio=1)", check_rollout_update_consistency),
        ("masked BACKWARD finite grads (single-legal-action)", check_masked_backward_finite_grads),
        ("warm-start strict load (16x16 -> 20x20 agent)", check_warm_start_strict_load),
    ]
    for name, fn in checks:
        fn()
        print(f"  [pass] {name}")
    print("\nAll v15 action-masking checks passed.")


if __name__ == "__main__":
    main()
