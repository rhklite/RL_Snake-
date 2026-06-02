# Iteration Log

Chronological record of every hypothesis tested on the Snake-RL project, numbered `v1..vN`.
Each version = one coherent hypothesis test (a run, or a focused decision applied mid-run).
This is the canonical history; the per-version files hold full detail.

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

**Trajectory:** food-seeking optimization (v1–v5) plateaued at ~10% grid fill on 32×32 no matter
the observation richness — the bottleneck was the *objective*, not the network. v6 reframed the
goal to explicit coverage and pure RL solved a small even grid completely. See
[full-coverage-design.md](../full-coverage-design.md) for the forward plan (Phase 1 curriculum,
Phase 2 Hamiltonian prior).

## v0 — pre-history (not numbered)

Before v1 there were exploratory throwaway runs preserved under [runs/archive/](../../runs/archive/):
`cnn_grid_*_grid12x12_*` and `*_grid64x64_*` (pure-CNN on int8 grid, 12×12 and 64×64) and early
`hybrid_*_grid32x32/64x64` runs. These predate the postmortem discipline, have no recorded
hypotheses, and established only the pipeline + that a plain CNN on a raw grid learns weakly. They
are not part of the numbered lineage.

## Open questions carried forward

Unresolved across iterations (from [design-packet.md §10](../design-packet.md) and
[full-coverage-design.md §10](../full-coverage-design.md)):

1. **Pure-RL coverage ceiling** — how large a grid can RL fill before a Phase 2 structural prior is
   required? Empirical; find via the curriculum.
2. **Encoder transfer across grid sizes** — does one encoder transfer, or is a global-pooling head /
   per-size fine-tune needed? (Blocks the curriculum.)
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
