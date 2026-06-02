# v1 — Baseline / fix-zigzag-hack

- **Date:** 2026-03-15
- **Run(s):** `runs/archive/0315_14/`, `0315_22/` (slug `fix-zigzag-hack`)
- **Grid:** 32×32 · **Obs:** single int8 grid (0=empty,1=body,2=head,3=food) + food-offset MLP
- **Outcome:** partial
- **Decisions introduced:** [D7](../design-packet.md) (delta-Manhattan shaping), [D8](../design-packet.md) (step penalty −0.01→−0.025)

## Hypothesis

A baseline hybrid model (CNN over the grid + MLP over the food offset, 2 conv layers, hidden=128)
with shaped reward will learn food-seeking and survival. Raising the step penalty from −0.01 to
−0.025 (2.5× the shaping signal) should suppress the zigzag/lawnmower reward-hacking seen in
prototypes. The single int8 grid was kept pending evidence it bottlenecks learning.

## Changes vs. prior

- Established the PPO pipeline, `ppo/` package, gymnasium env, YAML configs (commit `9ec6e3b`).
- Single int8 grid observation; 2-layer CNN + MaxPool, hidden=128, 256 envs.
- Reward: **+1** food, **−1** death, **−0.025**/step, **+0.01·Δmanhattan** shaping.

## Outcome & metrics

| Metric | Value |
|--------|-------|
| Best return | 27.82 (update 4229) |
| Best snake length | 45.8 |
| Final return | ~21–24 |
| Death split (final) | 42.5% wall / **57.4% body** / 0.07% timeout |
| Entropy | 1.386 → ~0.45 (collapsed by update ~100) |
| SPS | 9,100 → 6,533 |
| Eating rate | ~1 pellet / 22–25 steps |

## Learnings

- Entropy collapsed early into a conservative zigzag/lawnmower sweep.
- Body deaths overtook wall deaths (~update 2500) and dominated by the end (57.4%).
- The 2.5× penalty:shaping ratio did **not** kill zigzag, and the single int8 channel forces the
  CNN to disentangle head/body/food from one shared value space.
- Shelved-ideas item 1 revisit condition (body deaths >30% at ≥20% progress) was tripped (57.4% at
  14.1%), motivating per-element channels.

## Kept → v2

Reward structure (D7/D8), hybrid CNN+MLP head, distance-shaping coefficient.

## Discarded

Single int8 grid encoding (→ per-element channels in v2); 2-layer CNN depth (→ 4 layers in v2).

## Next

Attack zigzag and body-death dominance from the **observation** side: multi-channel encoding with a
body gradient + separate head/food channels, deeper CNN.

## Sources

- [0315_14_GE.md](../postmortem/0315_14_GE.md), [0315_2100_OP.md](../postmortem/0315_2100_OP.md)
- [shelved-ideas.md](../shelved-ideas.md), [design-packet.md D8](../design-packet.md)
- commits `9ec6e3b`, `2a6608a`
