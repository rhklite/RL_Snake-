# v6 — Full-coverage pivot on 6×6 (Phase 0 gate)

- **Date:** 2026-05-26
- **Run(s):** `runs/0526_23_coverage-6x6/`
- **Grid:** **6×6** · **Obs:** 4-channel hybrid (carried from v4/v5) + food MLP, **num_layers=2**
- **Outcome:** **success** — Phase 0 acceptance gate PASSED
- **Decisions introduced:** [D12](../full-coverage-design.md) (coverage goal), [D13](../full-coverage-design.md) (even grids), [D14](../full-coverage-design.md) (coverage reward), [D15](../full-coverage-design.md) (win detection + metrics), [D16](../full-coverage-design.md) (per-grid CNN depth)

## Hypothesis

Reframe the end goal from food-seeking (plateaued at ~9% fill on 32×32, see [v5](v5-fromscratch-validation.md))
to **100% grid coverage**. Test whether a coverage-oriented reward (step penalty 0, shaping off,
**+10 win bonus** on full fill, +1 food, −1 death) lets pure RL learn complete space-filling on a
small **even-sided** grid (6×6) with **no structural prior**. Even grids only — Hamiltonian cycles
require an even cell count. Reduce CNN depth to 2 layers because 4 layers + MaxPool collapses a 6×6
grid below 1×1.

## Changes vs. prior

- **Objective:** 100% coverage, not food maximization (D12).
- **Curriculum constraint:** even-sided grids; 6×6 for Phase 0 (D13).
- **Reward:** step_penalty **0.0**, dist_shaping_alpha **0.0**, **win_bonus 10.0**, +1 food, −1 death (D14).
- **Win detection:** `len(snake)==rows*cols` → terminate with cause `"win"`; added `avg_coverage`,
  `max_snake_length`, win-rate metrics (D15).
- **num_layers=2** for the 6×6 config to avoid spatial-dim collapse (D16).
- Channel-encoding refinements: body gradient offset to [0.1,1.0], BFS scale capped at len 51
  (commit `019a44f`); plateau detection requires ≥10 samples (commit `6c3310c`); gated 6×6 config
  added (commit `fc35a6e`).

## Outcome & metrics

| Metric | Value |
|--------|-------|
| **Avg coverage** | **~0.98** (frequently 1.0) |
| **Win rate** (`death/win_pct`) | **67%** |
| Best avg snake length | 36 / 36 (full grid) |
| Best avg return | 45.0 |
| Death split (final) | 22% wall / 11% body / **67% win** |
| Entropy | settled ~0.28 (no collapse) |
| Clip fraction | ~0.025 (healthy) |
| SPS | ~13,500 |
| Gate reached | ~update 245 (best.pt); run stopped ~update 2393 |

## Learnings

- **Gate PASSED:** the coverage reward alone is sufficient for pure RL to fill a 6×6 grid — no
  structural prior or curriculum needed at this scale. Wins dominate (67%), entropy stable.
- The entire v4/v5 observation + network stack carried over **unchanged**; only the reward, win
  signal, grid size, and CNN depth changed. This confirms v5's conclusion that the *objective* was
  the ceiling, not the network.
- **`best.pt` gotcha:** best-checkpoint tracking uses `avg_return`, which **saturates** once
  win_bonus is hit (~46 cap). So `best.pt` froze at update ~245 while the policy kept improving in
  *consistency* to update ~2393. `best.mp4` is one stochastic rollout from that early moment — it
  can show a short death even though aggregate coverage is ~0.98. **Watch the latest milestone
  video** (`videos/update_02000.mp4`), and for coverage runs track best-by-coverage, not return.

## Kept → v7 (Phase 1)

Coverage objective + reward (D12/D14), even-sided curriculum (D13), win detection & metrics (D15),
per-grid CNN depth (D16), 4-channel hybrid obs + HybridActorCritic.

## Discarded

Food/survival framing as the primary objective; −0.025 step penalty and +0.01 shaping (for coverage
mode).

## Next

- **Phase 1:** curriculum to larger even grids (8×8, 10×10, …) with a **size-agnostic encoder**
  (adaptive/global pooling head) so one policy transfers across sizes.
- **Phase 2:** structural prior (Hamiltonian cycle / spiral) for large grids where pure RL is
  expected to fail.
- Fix best-checkpoint criterion to track coverage/win-rate for coverage runs.

## Sources

- [full-coverage-design.md](../full-coverage-design.md) (design + §11 result)
- `runs/0526_23_coverage-6x6/` — `config.yaml`, `train.log`, `metrics.jsonl`
- commits `019a44f`, `6c3310c`, `fc35a6e`
