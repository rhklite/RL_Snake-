# v10 — Hamiltonian cycle-shaping prior at 10×10 (Phase 2 opens)

- **Date:** 2026-06-03
- **Run:** `runs/0603_01_coverage-10x10-cycle/` (warm-started from the v9 transfer endpoint `runs/0602_12_coverage-10x10-transfer/checkpoints/agent_3500.pt`)
- **Grid:** 10×10 (100 cells) · **Obs:** 4-channel hybrid + food MLP (unchanged from v9 → strict warm-start), **num_layers=2**, **adaptive_pool_size=4**
- **New knob:** `cycle_beta=0.03` (Hamiltonian cycle-alignment shaping); `best_metric=coverage`
- **Outcome:** **success (prior works on its target)** — self-trap deaths and win rate both move ~10 pts in the right direction with no cycle-walking pathology; coverage gain modest. Plateau, not solved.
- **Decisions exercised:** D19 (cycle-alignment shaping reward), D14 (coverage reward), D17 (adaptive encoder), D18 (`--init-from`). New decision: **D19**.

## Hypothesis

v9 showed ~10×10 is the Phase 1→2 boundary: a reactive PPO policy (even warm-started) plateaus at
~78% coverage / 48% win, bottlenecked by **self-trapping** (body deaths ~47%, scale-dependent,
doesn't transfer, can't be learned reactively). Phase 2 adds a **structural prior**: a reward-only,
bonus-only Hamiltonian cycle-alignment term (`+cycle_beta` for the on-cycle successor move, 0
otherwise). Following the cycle is a guaranteed full-fill, never-self-trapping path, so the prior
should target exactly the self-trap failure. Start at 10×10 warm-started from the v9 transfer policy
so the prior's contribution is measured against v9's 78%/48% plateau.

## Method note

Reward-only (obs unchanged at 4 channels) so the v9 strict warm-start loads tensor-for-tensor.
Bonus-only (never penalizes off-cycle) so it never taxes a legitimate food shortcut; `β=0.03` keeps
the discounted farm ceiling `β/(1−γ)=3.0 ≪ win_bonus=10`, so no cycle-walking trajectory can
out-rank a win. Gated before launch by `scripts/verify_cycle_shaping.py` (8/8: cycle validity incl.
zero-U-turn, pre-move-head firing, clean death/win, disabled-no-op, farm ceiling, strict v9 load).
"Run until plateau" driven by the v9 coverage-plateau watcher (`runs/_plateau_watch.py`, gitignored;
smoothed `avg_coverage`, no ≥0.005 gain over 800 updates past a 1500-update floor). best.pt is
coverage-selected (shaped `avg_return` is **not** comparable to v9).

## Outcome & metrics

| 10×10 | v9 transfer (baseline) | **v10 cycle-shaping (β=0.03)** |
|-------|------------------------|-------------------------------|
| Plateau coverage (300-avg) | 0.782 | **0.807** |
| Plateau coverage (200-avg) | — | 0.811 |
| Peak coverage (windowed) | 0.974 | **0.984** |
| Cumulative win rate | 47.7% | **58.1%** |
| Body / self-trap deaths | 46.9% | **37.4%** |
| Wall deaths | 5.4% | 4.5% |
| Timeout deaths | ~0 | **0.0%** |
| Plateau @ update | ~3634 | ~2625 (watcher) |

(Death splits are **cumulative** over the whole run, so they understate the recent rate;
`avg_coverage` is the windowed headline. Speed-to-threshold is omitted as a comparison: v10 starts
*from* v9's endpoint, so it inherits v9's level immediately — only the plateau and death-split shifts
are meaningful here.)

## Learnings

- **The cycle prior reduces self-trapping, as designed.** Body deaths **46.9% → 37.4%** and win rate
  **47.7% → 58.1%** — both ~10 pts in the intended direction. The structural prior attacks the exact
  failure mode reactive PPO couldn't, confirming the v9 diagnosis.
- **No cycle-walking pathology.** Timeout deaths stayed **0.0%** and coverage peaked at 0.984 — the
  main design risk (over-committing to the safe lap and timing out instead of winning) did not
  materialize. `β=0.03` (bonus-only) guided without dominating, as the farm-ceiling argument predicted.
- **The wall is dented, not removed.** Coverage plateau rose only **+2.5 pts (0.78 → 0.81)** and body
  deaths are **still the #1 failure (37%)**. A fixed soft reward bonus nudges the policy toward the
  cycle but doesn't enforce it; the residual self-traps persist where the agent deviates for food and
  boxes itself in.
- **Confound — gain is not cleanly attributed to shaping.** v10 warm-restarts from v9's checkpoint
  with a **fresh optimizer**; an optimizer reset + re-exploration (ent_coef 0.03) can itself unstick a
  plateau. The `cycle_beta=0.0` warm-restart control (below) is required to separate shaping from
  continued-training gain. **This is the load-bearing open item for v11.**

## Kept → next

D19 (cycle-alignment shaping), `best_metric=coverage`, β=0.03 default, reward-only/warm-start-safe
design, the verification gate, the coverage-plateau watcher methodology.

## Discarded

Nothing. (β=0.03 fixed is retained; annealing was not needed since no cycle-walking appeared.)

## Next

- **Run the `cycle_beta=0.0` control** (highest priority — isolates shaping from continued-training):
  `python train.py --training coverage_10x10_v10 --init-from runs/0602_12_coverage-10x10-transfer
  training.cycle_beta=0.0 training.hypothesis_slug=coverage-10x10-cycle-ctrl`. If the control reaches
  ~0.81 too, the gain is the optimizer reset, not the prior, and the shaping needs to be stronger.
- **If shaping is confirmed but plateaus:** escalate per open-Q#3 — stronger prior (cycle obs-channel,
  accepting a warm-start break), action-masking to legal cycle moves, or residual RL on a cycle
  follower. A β sweep (0.05, 0.08) is the cheap first probe.
- **Phase B refine** (optional): `cycle_beta=0.0` warm-restart *from this v10 run* to let late RL
  recover shortcuts off the cycle with zero guidance pressure.
- Carried: fix any remaining best-checkpoint edge cases; re-evaluate the BFS channel (open-Q#5).

## Sources

- `runs/0603_01_coverage-10x10-cycle/` — config, metrics, checkpoints (best.pt coverage-selected), videos
- [coverage_10x10_v10.yaml](../../config/training/coverage_10x10_v10.yaml); warm-start lineage v9 (`runs/0602_12_coverage-10x10-transfer`)
- [full-coverage-design.md](../full-coverage-design.md) §9 D19, §10 open-Q#3, Phase 2
- Gate: [scripts/verify_cycle_shaping.py](../../scripts/verify_cycle_shaping.py); commit `1e8c5db`
