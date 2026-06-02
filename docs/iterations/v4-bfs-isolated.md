# v4 — BFS isolated + tail/scaling fix (D10/D11)

- **Date:** 2026-03-17
- **Run(s):** `runs/archive/0317_22_bfs-reachability/` (mid-run fix at update ~9000)
- **Grid:** 32×32 · **Obs:** 4-channel (body gradient, head, food, BFS) + food MLP, **MaxPool restored**
- **Outcome:** mid-run fix (plateau broken)
- **Decisions introduced:** [D10](../design-packet.md) (linear body-gradient normalization), [D11](../design-packet.md) (softened BFS length ramp)

## Hypothesis

Isolate the BFS variable: re-add the proven 4-layer MaxPool CNN and keep BFS as the *only* change
vs v2 (undoing the v3 architecture confounds). Fix BFS normalization to per-step max distance (not
the grid-area divisor) and scale BFS by snake length so it's silent at length 1 and ramps up as
topology matters. If a plateau appears, swap in linear body-gradient normalization (D10) and the
softened BFS ramp min(1.0,(n−1)/50) (D11) mid-run to break it.

## Changes vs. prior

- **Reverted** to 4-layer MaxPool CNN (~385K params); BFS kept as sole obs change vs v2 (commit `fbd589b`).
- BFS normalized by **per-step max distance** (commit `d027574`).
- BFS scaled by (len−1)/grid_area to gate early learning (commit `745e367`).
- Resume + unlimited-updates + video-HUD infrastructure (commit `0a5d2c1`).
- **Mid-run @ update 9000:** body gradient (n−i)/n → **0.1 + 0.9·(n−1−i)/max(n−1,1)** (D10); BFS
  scale (n−1)/(rows·cols−1) → **min(1.0,(n−1)/50)** (D11).

## Outcome & metrics

| Phase | Avg snake size | Note |
|-------|----------------|------|
| Pre-fix plateau (upd 8000–9000) | ~95–97 | 44% wall / 56% body deaths |
| At fix (upd 9000) | 89.5 | return 55.4 |
| Post-fix dip (upd 9500) | 73.8 | transient |
| Post-fix recovery (upd 10000) | **105** | new local max |
| Final logged (upd 10124) | 85.7 | |

- Tail value at len 95: **0.0105 → 0.1** after D10. BFS signal at len 95: **9.2% → 100%** after D11.

## Learnings

- Isolating BFS from the v3 confounds **restored healthy training** — the run grew to a ~95–97
  plateau with the familiar 44/56 wall/body split, proving v3's collapse was architectural, not the
  BFS channel itself.
- At plateau the original 1/n tail (~0.0105) is invisible post ReLU+MaxPool, and D5 area-scaling
  delivers only 9.2% BFS strength — both starve the policy exactly where topology matters.
- The mid-run D10/D11 swap caused a transient dip then recovered to a new max (105). A naive
  `max(value,0.1)` clamp was tried first but collapsed the last ~9 tail segments to one value;
  full linear normalization fixed it.
- **Caveat:** D10 and D11 were applied *together* mid-run — neither was cleanly isolated. (Open
  question carried forward.)

## Kept → v5

BFS + MaxPool architecture (v3 confounds stay reverted), linear body gradient (D10), softened BFS
ramp (D11), per-step BFS normalization, resume infrastructure.

## Discarded

Original raw gradient (n−i)/n (→ D10), original area scaling (→ D11), naive clamp.

## Next

Validate the D10/D11 design **from scratch** on 32×32 to confirm it isn't an artifact of the
0317_22 trajectory. → v5.

## Sources

- [design-packet.md D10/D11](../design-packet.md) + plateau analysis, `runs/archive/0317_22_bfs-reachability/train.log`
- commits `fbd589b`, `d027574`, `745e367`, `0a5d2c1`
