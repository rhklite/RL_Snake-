# v7 — Pure-RL coverage on 8×8 (Phase 1, clean scale-up)

- **Date:** 2026-06-01
- **Run(s):** `runs/0601_22_coverage-8x8/`
- **Grid:** **8×8** · **Obs:** 4-channel hybrid + food MLP, **num_layers=2** (all carried from v6)
- **Outcome:** **success** — pure RL scales one even-grid step up; no architecture change
- **Decisions exercised:** D12–D16 (unchanged from v6). No new decisions.

## Hypothesis

Does the **exact v6 coverage recipe** (step penalty 0, shaping off, win_bonus 10, +1 food, −1
death, `num_layers=2`) scale from 6×6 to **8×8** with *only the grid size changed*? This is a
clean single-variable ablation (addresses open-Q#7's "no clean ablations" gripe) and the first
empirical probe of open-Q#1 — how large a grid pure RL can fill before a Phase 2 structural prior
is required.

## Changes vs. prior (v6)

- **Grid:** 6×6 → **8×8** (36 → 64 cells). *Everything else held identical.*
- New config [config/training/coverage_8x8.yaml](../../config/training/coverage_8x8.yaml);
  `max_hours` 2.0 → 4.0 (8×8 is harder, longer episodes).

## Outcome & metrics

| Metric | Value |
|--------|-------|
| **Avg coverage** | **0.884** (final 20-ep window); **peak 1.0** (full sweeps occurred) |
| **Win rate** (`death/win_pct`) | **56.4%** (max == final; mean last-50 ≈ 56.3%) |
| Max snake length | 64 / 64 (full grid) |
| Avg snake length (final) | 56.6 / 64 |
| Best avg return | ~73 peak (≈ 63 food + 10 win bonus); final ~64 |
| Death split (final) | **23.8% wall / 19.8% body / 56.4% win** |
| SPS | ~16k early → ~10k late (longer episodes) |
| Stop reason | **time limit (4.00h)** at update 4414 |

## Learnings

- **Pure RL scales 6×6 → 8×8 with zero recipe changes.** Win rate 67% → 56.4%, coverage ~0.98 →
  0.88. A **gentle degradation, not a wall** — 8×8 is still comfortably in pure-RL territory. The
  Phase1→Phase2 boundary is **beyond 8×8**; next probe is 10×10 (open-Q#1).
- **Stopped on the 4h clock, not on plateau.** The win-rate curve had flattened near ~56% (the
  plateau detector tracks *return*, which is noisy here), so 56% is likely close to this recipe's
  8×8 ceiling — but more time (or tuning) might lift it. Body deaths nearly doubled vs 6×6 (11% →
  19.8%): self-trapping is the dominant new failure mode at this scale, as expected for longer fills.
- **win_bonus=10 held constant and did not destabilize**, but its relative weight shrank: ~14% of a
  winning return at 8×8 vs ~22% at 6×6 (open-Q#4). It's the first knob to try if 10×10 stalls.
- **`best.pt` gotcha persists** (same as v6): best-checkpoint tracking uses `avg_return`, which
  saturates once wins land. Judge consistency from the **latest milestone video**
  (`videos/update_04000.mp4`), not `best.mp4`.

## Kept → v8

Coverage objective + reward (D12/D14), even-sided curriculum (D13), win detection & metrics (D15),
`num_layers=2`, 4-channel hybrid obs + HybridActorCritic. This 8×8 from-scratch run is the
**baseline** that v8's curriculum-transfer 8×8 must beat (faster convergence and/or higher win rate).

## Built during v7 (ready for v8, not yet run)

- **Size-agnostic encoder** (commit `5bbc07e`): opt-in `model.adaptive_pool_size` →
  `AdaptiveMaxPool2d(N)` before Flatten, dropping intermediate MaxPool so the CNN flatten dim is
  grid-invariant (512 for 6/8/10/12 at `num_layers=2`). Default path byte-for-byte unchanged. This
  is the open-Q#2 fix that unblocks the curriculum.
- **`--init-from`**: warm-starts a *fresh* run from a checkpoint, loading agent weights only
  (strict), fresh optimizer, `start_update=0`. Replaces the silent cp-`best.pt`+`--resume` footgun
  (best.pt is a full dict → would reload stale 6×6 optimizer/update-count with no shape error, since
  adaptive shapes are grid-invariant).

## Discarded

Nothing new. (The from-scratch-per-size approach is superseded *as the default* by v8's transfer
path, but remains a valid baseline — this run is exactly that baseline.)

## Next

- **v8 (Phase 1 transfer):** (1) train 6×6 under the adaptive encoder
  (`coverage_6x6_v8`) — confirm no regression vs v6's 67%. (2) Warm-start 8×8 from that seed
  (`coverage_8x8_v8 --init-from`) and compare against this from-scratch 8×8 baseline (does transfer
  reach ≥56% win faster or higher?). (3) Push to 10×10 to find the pure-RL ceiling.
- Fix best-checkpoint criterion to track coverage/win-rate for coverage runs (carried from v6).

## Sources

- `runs/0601_22_coverage-8x8/` — `config.yaml`, `train.log`, `metrics.jsonl`, videos
- [coverage_8x8.yaml](../../config/training/coverage_8x8.yaml); commit `1dd987a` (run code state)
- [full-coverage-design.md](../full-coverage-design.md) (Phase 1 plan)
