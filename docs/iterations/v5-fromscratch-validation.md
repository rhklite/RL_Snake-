# v5 — From-scratch validation of the D10/D11 design

- **Date:** 2026-03-20
- **Run(s):** `runs/0320_09_bfs-reachability/`
- **Grid:** 32×32 · **Obs:** 4-channel (body gradient w/ D10, head, food, BFS w/ D11) + food MLP, MaxPool CNN
- **Outcome:** partial (reproduced the plateau cleanly; exposed the *objective* as the real ceiling)
- **Decisions introduced:** none (validation run)

## Hypothesis

Train the full v4 design from scratch to confirm the mid-run D10/D11 fix is not a fragile artifact
of the 0317_22 trajectory, and characterize the steady-state plateau on 32×32.

## Changes vs. prior

- From-scratch run on the post-D10/D11 codebase (4-channel hybrid, MaxPool, num_layers=4).
- **Scaled up: 1024 envs** (vs 256 in v1–v4), num_steps=128, num_updates=0 (unlimited).
- ent_coef=0.03, dist_shaping_alpha=0.01, max_memory_pct=98, plateau_steps=1e6.
- Same reward (+1/−1/−0.025/+0.01 shaping).

## Outcome & metrics

| Metric | Value |
|--------|-------|
| Best avg return | 63.99 |
| Best avg snake length | **107** |
| Final avg return | 52.5 |
| Final avg snake length | 84.1 |
| Death split (final) | 44.3% wall / 55.6% body / 0.1% timeout |
| Updates | 4381 (~24 h) |
| SPS | ~6,700 |

## Learnings

- **Reproduced the v4 plateau profile** from scratch (best len 107, return 64, same ~44/56
  wall/body split) — D10/D11 are robust, not trajectory artifacts.
- The design still **saturates at ~10% grid fill** on 32×32. Wall and body deaths are roughly
  balanced, so the agent is no longer obviously bottlenecked by tail visibility or BFS scaling.
- **Key conclusion:** the remaining bottleneck is the **objective itself** — food-seeking with a
  −0.025 step penalty cannot reach high coverage on a large grid in practical training time. No
  observation enrichment fixes this. This directly motivated the v6 pivot.

## Kept → v6

The full v4 observation/architecture stack (4-channel obs, D10, D11, MaxPool CNN, hybrid head).

## Discarded

The implicit assumption that food-seeking + step penalty can reach high coverage on 32×32.

## Next

**Reframe the objective** from food maximization to explicit grid coverage; run a Phase 0
acceptance gate on a small even-sided grid. → v6.

## Sources

- `runs/0320_09_bfs-reachability/` — `config.yaml`, `train.log`, `metrics.jsonl`
