# v3 — BFS reachability + MaxPool removed (FAILED)

- **Date:** 2026-03-16 → 17
- **Run(s):** `runs/archive/0316_21_bfs-reachability/` (postmortem [0317_0015_OP](../postmortem/0317_0015_OP.md))
- **Grid:** 32×32 · **Obs:** 4-channel (body gradient, head, food, **BFS**) + food MLP
- **Outcome:** **failure** (catastrophic regression — agent never ate)
- **Decisions introduced:** [D3](../design-packet.md) (BFS channel), [D4](../design-packet.md) (inverted-distance), [D5](../design-packet.md) (area scaling), [D9](../design-packet.md) (Numba JIT)

## Hypothesis

A precomputed BFS reachability channel (inverted distance from head, forward-looking body
clearance, scaled by (n−1)/(rows·cols−1)) gives the CNN explicit topology it can't derive from a
finite receptive field. To preserve 1-cell resolution for the BFS signal, **remove all MaxPool**
and add a 1×1 conv bottleneck (128→4) before flatten. Numba JIT recovers BFS throughput.

## Changes vs. prior

- Added 4th channel: BFS reachability, forward-looking clearance, inverted-distance, area-scaled
  (D3, D4, D5).
- **Removed all MaxPool** + added **1×1 bottleneck** (128→4) (commit `f19399f`).
- Numba JIT BFS (D9, commit `1522271`, ~70× vs deque BFS); memory guard 95→98%.
- Same reward as v2.

## Outcome & metrics

| Metric | Value |
|--------|-------|
| Best return | −1.22 (update 386) |
| Final return | −1.33 |
| Avg snake size | stuck at 1.0–1.1 (never ate) |
| Episode length | flat at **16** (center→wall, straight line) |
| Death split | **99.97% wall**, 0% body |
| Entropy | 1.386 → 0.55 |
| Clip fraction | ~0 for most of training (gradients too weak to escape) |
| SPS | 1,750 (−78% vs v2) |

## Learnings

- **Confounded experiment:** MaxPool removal + 1×1 bottleneck + BFS channel changed three things at
  once, so the cause of collapse can't be cleanly attributed.
- The agent never ate — walked straight up ~16 steps and died on the wall, for 391 updates. Clip
  fraction ~0 means the policy never escaped its initial local minimum.
- Area scaling (D5) makes BFS ≈0 at length 1 — no signal during the critical food-finding bootstrap,
  yet near-uniform noisy values when slightly longer.
- SPS dropped 4.5× vs v2, so the same wall-clock yielded far fewer gradient updates.

## Kept → v4

BFS channel concept (D3), inverted-distance encoding (D4), Numba JIT (D9).

## Discarded

**MaxPool removal + 1×1 bottleneck** (reverted in v4, commit `fbd589b`). D5 area-scaling flagged
for revision (later superseded by D11).

## Next

**Decouple the confounds:** restore the proven 4-layer MaxPool CNN, keep BFS as the *only*
independent variable vs v2, and fix BFS normalization/gating so the channel doesn't poison early
training. → v4.

## Sources

- [0317_0015_OP.md](../postmortem/0317_0015_OP.md), [design-packet.md](../design-packet.md) (confounded-experiment note)
- commits `f19399f`, `1522271`
