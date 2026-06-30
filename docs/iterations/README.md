# Iteration Log

Chronological record of every hypothesis tested on the Snake-RL project, numbered `v1..vN`.
Each version = one coherent hypothesis test (a run, or a focused decision applied mid-run).
This is the canonical history; the per-version files hold full detail.

> **Start here:** [HISTORY.md](HISTORY.md) is the consolidated **challenge → test → solution → gap**
> chain across *all* iterations (v0→v19), where each gap motivates the next. This README is the
> table-of-contents + open-questions ledger; HISTORY.md is the narrative spine. v1–v10 also have
> standalone per-version files; v11–v19 detail lives in HISTORY.md.
>
> **How to use this:** before starting a new experiment, read the latest version file and the
> [open questions](#open-questions-carried-forward). When you finish a run, add a new `vN+1`
> file using the same template and update the table below. Decision-log IDs (`D1`..`D16`) live in
> [design-packet.md](../design-packet.md) (D1–D11) and [full-coverage-design.md](../full-coverage-design.md) (D12–D16).

## Versions at a glance

| Ver | Date | Grid | Headline change | Outcome | Best result |
|-----|------|------|-----------------|---------|-------------|
| [v1](v1-baseline.md) | 2026-03-15 | 32×32 | Baseline PPO, single int8 grid, raise step penalty to −0.025 | partial | return 27.8, len 45.8 |
| [v2](v2-body-gradient.md) | 2026-03-16 | 32×32 | Body-gradient + per-element channels (3-ch), 4-layer CNN | partial | return 49.9, len 83 (+80%) |
| [v3](v3-bfs-maxpool-removed.md) | 2026-03-16 | 32×32 | +BFS channel, **remove MaxPool**, 1×1 bottleneck | **failure** | len ~1 (never ate) |
| [v4](v4-bfs-isolated.md) | 2026-03-17 | 32×32 | BFS isolated (MaxPool restored) + mid-run tail/scaling fix (D10/D11) | mid-run fix | plateau ~95→105 |
| [v5](v5-fromscratch-validation.md) | 2026-03-20 | 32×32 | From-scratch validation of v4 design, 1024 envs | partial | return 64, len 107 |
| [v6](v6-coverage-6x6.md) | 2026-05-26 | 6×6 | **Pivot to 100% coverage**: win bonus, step penalty→0, shaping off | **success** | **100% coverage, 67% win rate** |
| [v7](v7-coverage-8x8.md) | 2026-06-01 | 8×8 | Clean scale-up: v6 recipe, **only grid size 6×6→8×8** | **success** | 88% coverage, 56% win rate |
| [v8](v8-adaptive-transfer-8x8.md) | 2026-06-02 | 6×6→8×8 | **Size-agnostic encoder** (adaptive pool) + **`--init-from` warm-start curriculum** | **success** | 8×8 transfer **70% win** (vs 56% scratch), ~3× faster |
| [v9](v9-ceiling-10x10.md) | 2026-06-02 | 10×10 | Pure-RL **ceiling probe**: from-scratch vs chained 8×8→10×10 transfer | **success (boundary)** | scratch walled ~63% cov/16% win; transfer ~78% cov/48% win (self-trap wall) |
| [v10](v10-cycle-shaping-10x10.md) | 2026-06-03 | 10×10 | Hamiltonian cycle-shaping prior, β=0.03 bonus-only (D19); β=0 control | **negative** | control 0.817 ≥ shaped 0.811 → β=0.03 inert; real gain = warm-restart (**v9.5 = 0.817**) |
| v11 *(detail in [HISTORY](HISTORY.md))* | 2026-06-05 | 12×12 | Asymmetric actor-critic (privileged cycle critic) | **negative** | 0.70 < baseline; critic can't teach the actor → β bonus is the only lever |
| v12 *([HISTORY](HISTORY.md))* | 2026-06-04→05 | 10×10 | Escalate β 0.03→0.08, 3-seed de-risk | **success** | **0.890 ± 0.009** (3 seeds); but β=0.08 ceilings ~0.80 @12×12 |
| v13 *([HISTORY](HISTORY.md))* | 2026-06-10 | 10×10 | Co-scale **β=0.15 / win_bonus=20**; probe β=0.18 | **success** | **0.939** (best 10×10); β=0.18 unstable → lock β=0.15/win=20 |
| v14 *([HISTORY](HISTORY.md))* | 2026-06-10→12 | 12→20² | Scale locked recipe up the ladder (A/C/D) | **wall** | 12²=0.903, 16²=0.82, **20²=0.645 (82% body)** — self-avoidance collapses |
| v15 *([HISTORY](HISTORY.md))* | 2026-06-28 | 20×20 | **Action-masking** the action space (safety + cycle), open-Q#3 opt 2 | **success (peak)** | safety **0.716 / 18% win** beats v14's 0.645; unstable late (→timeout 34%); cycle timeout-defeated 0.21 |
| v16 *([HISTORY](HISTORY.md))* | 2026-06-28 | 20×20 | Safety-mask + **LR anneal** base→0 (stability lever) | **success (stable)** | **0.765 peak, HOLDS 0.728, win 27%** — fixes v15's decay; best stable result |
| v17–v19 *([HISTORY](HISTORY.md))* | 2026-06-28→29 | 24²/22² | Climb the ladder: masked recipe past 20×20 (mask-first / mask-later / 22×22) | **negative** | masking ceilings at 20×20 — 22²/24² all collapse to ~0.10, 66–83% timeout-wander |

**Trajectory:** food-seeking optimization (v1–v5) plateaued at ~10% grid fill on 32×32 no matter
the observation richness — the bottleneck was the *objective*, not the network. v6 reframed the
goal to explicit coverage and pure RL solved a small even grid completely; v7 showed the same recipe
scales one step up (8×8, 56% win) with no architecture change — a gentle degradation, not a wall.
v8 made the encoder grid-invariant (adaptive pooling) and showed a 6×6 seed **transfers** to 8×8 via
warm-start: ~3× faster convergence and a higher ceiling (70% vs 56%). v9 found the pure-RL boundary at
~10×10 (residual wall = self-trapping); v10 added a cycle-shaping prior that was **inert at β=0.03**
(the real gain was a fresh-optimizer warm-restart, v9.5=0.817); v11's privileged-critic detour failed;
v12–v13 then **escalated the β prior** — β=0.08 → 0.890, and **β=0.15/win_bonus=20 → 0.939** at 10×10,
the locked recipe. v14 scaled it up the ladder (12²=0.903, 16²=0.82) until it **hit a wall at 20×20
(0.645, 82% self-trap deaths)**; v15 then **broke the wall with action-masking** — masking off wall/body
suicides (open-Q#3 opt 2) reached **0.716/0.756 peak (2-seed) / 13–18% win**, beating v14, though it
decays from the peak late; v16 **fixed the decay with LR annealing** — a *stable* **0.765 peak holding
0.728, win 27%**, the best stable result. v17–v19 then tried to **climb the ladder past 20×20** (24×24
mask-first, mask-later, and a gentler 22×22 step) and all **failed** — masking's benefit ceilings sharply
at 20×20; beyond it the masked snake **wanders to timeout** (no death + `step_penalty=0` makes circling
free). So **v16's 0.728 @20×20 is the banked headline**; the 24×24 wall is the current frontier (next:
re-add a small step penalty so wandering is costly). Full chain: [HISTORY.md](HISTORY.md); forward
plan: [full-coverage-design.md](../full-coverage-design.md).

## v0 — pre-history (not numbered)

Before v1 there were exploratory throwaway runs preserved under [runs/archive/](../../runs/archive/):
`cnn_grid_*_grid12x12_*` and `*_grid64x64_*` (pure-CNN on int8 grid, 12×12 and 64×64) and early
`hybrid_*_grid32x32/64x64` runs. These predate the postmortem discipline, have no recorded
hypotheses, and established only the pipeline + that a plain CNN on a raw grid learns weakly. They
are not part of the numbered lineage.

## Open questions carried forward

Unresolved across iterations (from [design-packet.md §10](../design-packet.md) and
[full-coverage-design.md §10](../full-coverage-design.md)):

1. ~~**Pure-RL coverage ceiling**~~ — **answered (v9): ~10×10 is the boundary.** From-scratch walls
   at ~63% cov / 16% win (never reaches 75%); chained transfer reaches ~78% cov / 48% win but also
   plateaus. The residual wall is **self-trapping** (body deaths ~47% in both), which neither pure RL
   nor warm-start removes → motivates the Phase 2 structural prior at ~10×10+.
2. ~~**Encoder transfer across grid sizes**~~ — **answered (v8):** an `AdaptiveMaxPool2d` head makes
   the encoder grid-invariant and a 6×6 seed warm-starts 8×8 (~3× faster, 70% vs 56% win). [D17/D18]
3. **Phase 2 prior form** — shaping-toward-Hamiltonian vs. action-masking to legal cycle moves vs.
   residual RL on a cycle follower?
4. **`win_bonus` magnitude** — 10.0 worked on 6×6; not validated at larger scales.
5. **Does BFS (D3) help coverage** or is it noise on small grids? Re-evaluate per scale.
6. **Potential-based shaping** (γ·φ(s′)−φ(s)) stability — proposed, never tested.
7. **Clean ablations never run** — D10 (tail floor) and D11 (BFS ramp) were applied *together*
   mid-run in v4; neither was isolated. Same for hybrid-MLP vs. food-as-5th-channel.
8. **1×1 bottleneck vs. MaxPool removal** — v3 confounded both; v4 reverted both together, so which
   one caused the v3 collapse is still unknown.
9. **Undocumented rationale** — why hybrid CNN+MLP over a 5th grid channel (D6), and why 4 CNN
   layers in v2.
