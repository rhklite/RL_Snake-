# v9 — Pure-RL ceiling + curriculum breakthrough at 10×10

- **Date:** 2026-06-02
- **Run(s):** scratch `runs/0602_09_coverage-10x10-scratch/` · transfer `runs/0602_12_coverage-10x10-transfer/` (chained 8×8→10×10)
- **Grid:** 10×10 (100 cells) · **Obs:** 4-channel hybrid + food MLP, **num_layers=2**, **adaptive_pool_size=4** (both runs)
- **Outcome:** **success (boundary found)** — 10×10 is where pure-RL-from-scratch breaks down; the curriculum pushes past it but does not solve it
- **Decisions exercised:** D13/D14 (coverage reward, even grids), D17 (adaptive encoder), D18 (`--init-from` warm-start). No new decisions.

## Hypothesis

Find the **pure-RL coverage ceiling** (open-Q#1, the Phase 1→Phase 2 boundary): does pure RL still
fill 10×10, and does the curriculum (chained 8×8→10×10 warm-start) push past wherever from-scratch
stalls? Both runs use the **identical adaptive architecture**, differing only in init (random vs
warm-started from the v8 8×8 policy) — a cleaner scratch-vs-transfer comparison than v7/v8.

## Method note

"Run until plateau" was driven by an **external coverage-plateau watcher** (`runs/_plateau_watch.py`,
gitignored): the built-in detector keys off `avg_return` (noisy) and `death/win_pct` is *cumulative*
(always flattens as the denominator grows — not a convergence signal). The watcher stops a run when
the smoothed recent `avg_coverage` shows no ≥0.005 gain for 800 updates. A 10h `max_hours` backstop
bounded both. (A first noise-sensitive watcher variant stopped the transfer prematurely at ~76%; a
patience-based rewrite let it settle at ~78%.)

## Outcome & metrics

| 10×10 | from-scratch | **chained transfer (8×8→10×10)** |
|-------|--------------|----------------------------------|
| Plateau coverage (300-avg) | ~0.634 | **~0.782** |
| Peak coverage | 0.887 | **0.974** |
| Cumulative win rate | 16.4% | **47.7%** |
| Death split (final) | 37.5% wall / 46.1% body / 16.4 win | **5.4% wall** / 46.9% body / 47.7 win |
| Updates to plateau | ~2072 | ~3634 |
| cov(50-avg) ≥ 0.40 @ update | 316 | **52** (6×) |
| cov(50-avg) ≥ 0.55 @ update | 610 | **82** (7×) |
| cov(50-avg) ≥ 0.65 @ update | 804 | **146** (5.5×) |
| cov(50-avg) ≥ 0.75 @ update | **never** | 989 |

(`death/win_pct` is cumulative over the whole run, so it understates the recent rate; `avg_coverage`
is the clean windowed metric and the headline.)

## Learnings

- **10×10 is the from-scratch wall (open-Q#1, answered).** Pure RL from scratch plateaus at **~63%
  coverage / 16% win and never reaches 75%** — a sharp fall from 8×8 (88%/56%). This is the
  exploration wall §2.3 predicted: the agent essentially never randomly discovers near-perfect fills
  at this scale.
- **The curriculum breaks through but does not solve it.** Chained transfer reaches **~78%
  coverage / 48% win** (peak 0.974), converges **~5–7× faster**, and reaches a coverage level
  from-scratch *never* attains — but it plateaus far short of 100%. **Transfer mitigates the wall; it
  doesn't remove it.**
- **The residual wall is self-trapping, and it doesn't transfer.** Wall deaths collapse 37.5% →
  **5.4%** (wall-avoidance is scale-invariant and ports perfectly, consistent with v8), but body
  (self-trap) deaths are **~46–47% in *both* runs** — the dominant, scale-dependent failure. The
  curriculum converts saved wall-deaths into wins, but neither pure RL nor warm-start teaches the
  long-horizon planning needed to avoid boxing yourself in on a 100-cell grid.
- **Implication: ~10×10 is the Phase 1→Phase 2 boundary.** A reactive PPO policy (even warm-started)
  tops out around 78% coverage / 48% win here. Closing the gap to 100% requires the **Phase 2
  structural prior** (Hamiltonian/spiral guidance) — exactly the self-trap-avoidance the network
  can't learn reactively.

## Kept → next

Coverage reward (D12/D14), even curriculum (D13), adaptive encoder (D17), `--init-from` warm-start
(D18), `num_layers=2`, hybrid obs. The coverage-plateau watcher methodology.

## Discarded

Nothing. (From-scratch at 10×10 is retained as the baseline that quantifies the wall.)

## Next

- **Phase 2 (v10+):** add an optional Hamiltonian/spiral guidance reward (or action-masking to legal
  cycle moves, or residual RL on a cycle follower) targeting the self-trap failure mode. Start at
  10×10 where pure RL + curriculum demonstrably plateaus (78%), so the prior's contribution is
  measurable against this v9 transfer baseline. Design TBD — revisit open-Q#3 (shaping vs masking vs
  residual) with this data in hand.
- **win_bonus** (open-Q#4): now ~9% of a winning return at 10×10; a larger terminal bonus is a cheap
  thing to try before/with Phase 2, though it won't address self-trapping directly.
- Fix best-checkpoint criterion to track coverage/win-rate (carried from v6–v8).

## Sources

- `runs/0602_09_coverage-10x10-scratch/`, `runs/0602_12_coverage-10x10-transfer/` — configs, logs, metrics, videos
- [coverage_10x10_v8.yaml](../../config/training/coverage_10x10_v8.yaml); seed lineage v8 (`runs/0602_04_coverage-8x8-v8`)
- [full-coverage-design.md](../full-coverage-design.md) §2.3 (exploration wall), Phase 2
