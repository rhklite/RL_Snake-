#!/usr/bin/env python3
"""Correctness gate for the v11 asymmetric actor-critic (privileged critic).

Run before any v11 training:  python3 scripts/verify_asym_critic.py
All checks raise (fail visibly) on violation. The load-bearing guarantee is that the
ACTOR never sees the cycle (so the policy is deployable without it and warm-starts cleanly),
while the CRITIC does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import AsymmetricHybridActorCritic, HybridActorCritic  # noqa: E402
from snake_env import build_cycle_phase  # noqa: E402


def _obs(rows: int, cols: int, b: int = 8, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "grid": torch.rand(b, 4, rows, cols, generator=g),
        "food": torch.rand(b, 2, generator=g) * 2 - 1,
    }


def check_shapes() -> None:
    for rows, cols in [(10, 10), (12, 12)]:
        m = AsymmetricHybridActorCritic(rows, cols, num_layers=2, adaptive_pool_size=4)
        obs = _obs(rows, cols)
        a, lp, ent, v = m.get_action_and_value(obs)
        assert a.shape == (8,) and lp.shape == (8,) and ent.shape == (8,)
        assert v.shape == (8, 1), f"value shape {v.shape}"
        assert m.get_value(obs).shape == (8, 1)
        assert m.cycle_field.shape == (1, rows, cols)


def check_actor_blind_critic_sees() -> None:
    """Perturbing the cycle field must NOT change actor logits, but MUST change the value."""
    m = AsymmetricHybridActorCritic(10, 10, num_layers=2, adaptive_pool_size=4)
    obs = _obs(10, 10)
    # actor logits and value before
    feats = m.actor_net._encode(obs)
    logits1 = m.actor_net.actor(feats).detach().clone()
    v1 = m.get_value(obs).detach().clone()
    # zero the privileged channel
    m.cycle_field.zero_()
    logits2 = m.actor_net.actor(m.actor_net._encode(obs)).detach()
    v2 = m.get_value(obs).detach()
    assert torch.equal(logits1, logits2), "ACTOR logits changed with the cycle field (leak!)"
    assert not torch.allclose(v1, v2), "CRITIC value unchanged by the cycle field (not used)"


def check_actor_warmstart_exact() -> None:
    """load_actor_weights from a HybridActorCritic reproduces that net's actor exactly."""
    h = HybridActorCritic(10, 10, num_layers=2, adaptive_pool_size=4)
    obs = _obs(10, 10, seed=3)
    h_logits = h.actor(h._encode(obs)).detach()

    m = AsymmetricHybridActorCritic(10, 10, num_layers=2, adaptive_pool_size=4)
    # fresh asym actor should differ from h before loading
    pre = m.actor_net.actor(m.actor_net._encode(obs)).detach()
    assert not torch.allclose(pre, h_logits), "fresh actor already matches (test vacuous)"
    m.load_actor_weights(h.state_dict())
    post = m.actor_net.actor(m.actor_net._encode(obs)).detach()
    assert torch.allclose(post, h_logits, atol=1e-6), "actor warm-start did not reproduce source"


def check_real_checkpoint_load() -> None:
    """The 10x10 beta=0.08 checkpoint actor-loads into a 12x12 asymmetric agent (adaptive pool)."""
    ckpt_dir = (
        Path(__file__).resolve().parent.parent
        / "runs" / "0604_15_coverage-10x10-cycle-b08" / "checkpoints"
    )
    ckpt = None
    for name in ("agent_4000.pt", "best.pt"):
        if (ckpt_dir / name).exists():
            ckpt = ckpt_dir / name
            break
    if ckpt is None:
        print(f"  [skip] real checkpoint load: none found in {ckpt_dir}")
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state["agent"] if isinstance(state, dict) and "agent" in state else state
    m = AsymmetricHybridActorCritic(12, 12, num_layers=2, adaptive_pool_size=4)
    m.load_actor_weights(state)  # strict; raises on any shape/key mismatch


def check_cycle_phase_field() -> None:
    for rows, cols in [(10, 10), (12, 12), (8, 5)]:
        p = build_cycle_phase(rows, cols)
        assert p.shape == (rows, cols)
        assert p.min() >= 0.0 and p.max() < 1.0, f"{rows}x{cols}: phase out of [0,1)"
        # one distinct phase per cell (it is a permutation index / n)
        assert len(np.unique(p)) == rows * cols, f"{rows}x{cols}: phases not all distinct"


def main() -> None:
    checks = [
        ("shapes at 10x10 and 12x12", check_shapes),
        ("actor blind to cycle, critic uses it", check_actor_blind_critic_sees),
        ("actor warm-start reproduces source exactly", check_actor_warmstart_exact),
        ("real 10x10 beta=0.08 checkpoint actor-loads at 12x12", check_real_checkpoint_load),
        ("cycle-phase field valid", check_cycle_phase_field),
    ]
    for name, fn in checks:
        fn()
        print(f"  [pass] {name}")
    print("\nAll v11 asymmetric-critic checks passed.")


if __name__ == "__main__":
    main()
