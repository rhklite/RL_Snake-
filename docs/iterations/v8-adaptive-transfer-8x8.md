# v8 — Size-agnostic encoder + curriculum transfer (6×6 → 8×8)

- **Date:** 2026-06-02
- **Run(s):** seed `runs/0602_02_coverage-6x6-v8/` (6×6 adaptive) · transfer `runs/0602_04_coverage-8x8-v8/` (8×8 ←seed)
- **Grid:** 6×6 then 8×8 · **Obs:** 4-channel hybrid + food MLP, **num_layers=2**, **adaptive_pool_size=4**
- **Outcome:** **success** — encoder transfers across grid sizes; warm-start beats from-scratch on 8×8
- **Decisions introduced:** [D17](../full-coverage-design.md) (size-agnostic adaptive-pool encoder), [D18](../full-coverage-design.md) (warm-start curriculum via `--init-from`)

## Hypothesis

The CNN encoder was size-locked (flatten dim computed from a fixed grid), the blocker for the whole
curriculum (open-Q#2). Make it **grid-invariant** with an `AdaptiveMaxPool2d(N)` head, then test
whether **one policy's weights transfer** across even grid sizes: train a 6×6 seed, warm-start 8×8
from it, and compare against the v7 8×8 from-scratch baseline (56% win, 88% coverage). Transfer
should reach 56% **faster** and ideally to a **higher ceiling**.

## Changes vs. prior (v7)

- **Encoder (D17):** opt-in `model.adaptive_pool_size=4` → drops the intermediate `MaxPool` and
  inserts `AdaptiveMaxPool2d(4)` before `Flatten`, so the CNN flatten dim is **512 for every grid
  size** (`channels[-1]·4·4`). Default path (flag absent) byte-for-byte unchanged → v6/v7 reproducible.
  Fail-visibly guard raises if pre-pool spatial < N (degenerate upsample).
- **Warm-start (D18):** `train.py --init-from <run>` seeds a *fresh* run with **agent weights only**
  (strict), fresh optimizer, `start_update=0`. Strict load means a cross-size load without the
  adaptive head fails loudly instead of silently reloading stale state.
- Configs: [coverage_6x6_v8.yaml](../../config/training/coverage_6x6_v8.yaml) (seed),
  [coverage_8x8_v8.yaml](../../config/training/coverage_8x8_v8.yaml) (transfer). Reward/PPO held from v6/v7.

## Outcome & metrics

**Seed (6×6 adaptive, 2h):** avg_coverage **0.975**, **win 70.7%**, deaths 17.3w/12.0b. *No
regression vs v6 (67% win) — marginally better*, confirming the adaptive head doesn't hurt 6×6 (the
4×4 bottleneck retains more spatial info than v6's post-MaxPool 3×3 flatten).

**Transfer (8×8 ← seed, 4h) vs v7 from-scratch (8×8, 4h):**

| Metric | v7 from-scratch | **v8 transfer** |
|--------|-----------------|-----------------|
| Final win rate | 56.4% | **70.0%** |
| Avg coverage | 0.884 | 0.865 |
| Death split | 23.8% wall / 19.8% body / 56.4 win | **4.4% wall** / 25.5% body / 70.0 win |
| Updates in 4h | 4414 | 4414 |
| win≥10% @ update | 645 | **247** (2.6×) |
| win≥25% @ update | 1188 | **392** (3.0×) |
| win≥40% @ update | 2102 | **659** (3.2×) |
| win≥50% @ update | 3208 | **993** (3.2×) |

Early ramp (stdout): transfer hits avg snake ~7/64 by **update 6** vs from-scratch ~update 37.

## Learnings

- **The encoder transfers (open-Q#2: answered).** Warm-started 8×8 converges **~3× faster** and to a
  **higher final win rate (70% vs 56%)** — it inherits the seed's 70.7% rather than rediscovering
  coverage from random. The size-agnostic adaptive-pool head is the enabler.
- **Transfer is *not* zero-shot.** The 6×6 policy applied cold to 8×8 starts near random (update 1
  avg snake ~1.4) — absolute spatial habits don't map — but the learned *features* are a far better
  init than random, so it re-adapts in a handful of updates.
- **Wall-avoidance transfers cleanly; self-trapping does not.** Wall deaths collapse 23.8% → **4.4%**
  (walls look identical at any grid size, so that skill ports directly), while body deaths *rise*
  19.8% → 25.5%. Self-trapping is the genuinely scale-dependent, harder part of coverage and the
  main thing re-learned at each new size. **This is the signal to watch as the curriculum scales.**
- **Compute caveat (fail-visibly):** v8's total cost is higher — 2h seed + 4h transfer = **6h** vs
  v7's 4h. Transfer wins per-transfer-budget and the seed amortizes across *all* larger stages
  (10×10, 12×12 reuse the same seed/encoder), but it is not a free lunch at a single size.
- `best.pt` gotcha unchanged from v6/v7 (avg_return-based; saturates on wins).

## Kept → v9

Coverage reward (D12/D14), even curriculum (D13), win metrics (D15), `num_layers=2`,
**adaptive-pool encoder (D17)**, **`--init-from` warm-start curriculum (D18)**, 4-channel hybrid obs.

## Discarded

In-place `--resume <6x6run> rows=8 cols=8` for the curriculum (overwrites the seed dir, continues the
6×6 optimizer/update-count); and the cp-`best.pt`+`--resume` warm-start (silent: best.pt is a full
dict, would reload stale state with no shape error since shapes are grid-invariant). D18 replaces both.

## Next

- **v9:** warm-start **10×10** from the 6×6 seed (or chain 8×8→10×10) and find the pure-RL ceiling —
  where win rate falls off enough to motivate the Phase 2 structural prior (open-Q#1). Watch body
  (self-trap) deaths, the dominant failure mode.
- **win_bonus** scaling (open-Q#4): its relative weight keeps shrinking as the grid grows; revisit if
  10×10 stalls.
- Fix best-checkpoint criterion to track coverage/win-rate for coverage runs (carried from v6/v7).

## Sources

- `runs/0602_02_coverage-6x6-v8/` (seed), `runs/0602_04_coverage-8x8-v8/` (transfer) — configs, logs, metrics, videos
- commits `5bbc07e` (encoder + `--init-from`); design D17/D18
- [full-coverage-design.md](../full-coverage-design.md)
