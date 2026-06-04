# v10 — Hamiltonian cycle-shaping prior at 10×10 (Phase 2 opens)

- **Date:** 2026-06-03
- **Run:** `runs/0603_01_coverage-10x10-cycle/` (warm-started from the v9 transfer endpoint `runs/0602_12_coverage-10x10-transfer/checkpoints/agent_3500.pt`)
- **Grid:** 10×10 (100 cells) · **Obs:** 4-channel hybrid + food MLP (unchanged from v9 → strict warm-start), **num_layers=2**, **adaptive_pool_size=4**
- **New knob:** `cycle_beta=0.03` (Hamiltonian cycle-alignment shaping); `best_metric=coverage`
- **Control:** `runs/0603_04_coverage-10x10-cycle-ctrl/` — same v9 warm-start, `cycle_beta=0.0` (shaping OFF)
- **Outcome:** **NEGATIVE for the prior.** The β=0 control matches/exceeds v10 on every metric, so the
  v9→v10 gain is the **fresh-optimizer warm-restart + continued training, NOT the cycle prior**. β=0.03
  is inert at this strength. (An earlier version of this doc, written before the control finished,
  claimed "success" — that was a premature attribution; corrected here.)
- **Decisions exercised:** D19 (cycle-alignment shaping reward), D14 (coverage reward), D17 (adaptive encoder), D18 (`--init-from`). New decision: **D19** (kept as a tested-but-inert-at-β=0.03 mechanism, not as a confirmed win).

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

| 10×10 | v9 transfer (baseline) | v10 cycle-shaping (β=0.03) | **control (β=0, shaping OFF)** |
|-------|------------------------|---------------------------|--------------------------------|
| Plateau coverage (200-avg) | 0.782 | 0.811 | **0.817** |
| Peak coverage (windowed) | 0.974 | 0.984 | **0.995** |
| Recent win rate (windowed) | ~48% | 60% | **62%** |
| Recent body / self-trap deaths | ~47% | 35% | **34%** |
| Cumulative win rate | 47.7% | 58.1% | **60%** |
| Wall deaths | 5.4% | 4.5% | ~4% |
| Timeout deaths | ~0 | 0.0% | 0.0% |
| Plateau @ update | ~3634 | ~2636 (watcher) | ~2695 (watcher) |

**The control matches/exceeds v10 on every axis.** Both warm-start from the *same* v9 checkpoint;
the only difference is `cycle_beta` (0.03 vs 0). So the entire v9→v10 movement is the fresh-optimizer
warm-restart, not the shaping. Caveat: single seed each (both seed=1), so the v10↔control 0.006-cov
gap is within noise — but the direction (control ≥ v10) is unambiguous; there is no hidden v10 win.
Death splits are cumulative (understate recent); recent rates are reconstructed by differencing
cumulative `death/*_pct` against estimated episode counts (same method for both runs, so comparable).

## Learnings

- **The β=0.03 cycle prior is inert — the control settles the attribution.** The `cycle_beta=0.0`
  control (same v9 warm-start, shaping off) reached **0.817 cov / 62% win / 34% body**, matching or
  beating v10's **0.811 / 60% / 35%**. The shaped and unshaped policies converge to the same behavior:
  a 0.03/step bonus (discounted ceiling 3.0) is swamped by the coverage/win signal and doesn't shift
  the policy. The v9 self-trap diagnosis may still be right, but **this prior at this strength does not
  address it.**
- **The real lever was the fresh-optimizer warm-restart, not shaping.** Simply re-training v9's
  checkpoint with a reset optimizer + re-exploration (ent_coef 0.03) moved **cov 0.78→0.82, win
  48→62%, body 47→34%**. v9 was **not actually converged**; the optimizer reset unstuck it. This is a
  genuine result (call it v9.5) that is independent of any prior and was previously hidden because v9
  reported its own warm-restart as "the curriculum."
- **No cycle-walking pathology (but moot).** Timeouts stayed 0.0% in both runs — the farm-ceiling
  argument held — but since the bonus didn't change behavior, this tells us little about the design.
- **Process lesson.** The interim "shaping is working" read (and the pre-control draft of this doc)
  was a premature attribution. The single-flag control was cheap and decisive; it should gate the
  *headline*, not just appear in "Next." Never claim a warm-restart improvement for the intervention
  until the no-intervention warm-restart is ruled out.

## Kept → next

The reward-only / warm-start-safe shaping *infrastructure* (`build_cycle_succ`, `cycle_beta` plumbing,
the verification gate), `best_metric=coverage`, and the coverage-plateau watcher methodology — all
sound and reusable. The **v9.5 finding** (a fresh-optimizer warm-restart of v9 reaches ~0.82/62%/34%)
is the new baseline to beat.

## Discarded

**β=0.03 as an effective setting** — proven inert by the control. Not the mechanism, just too weak to
register. (D19's bonus-only/fixed-β design is retained as scaffolding; its *efficacy* is unproven.)

## Next (control is DONE — it refuted the prior; decision pending)

- **(A) Escalate the prior.** β sweep (0.05, 0.08 — ceiling rises 5.0/8.0, still < win_bonus=10) to
  test whether a *stronger* push registers; if still inert, climb the open-Q#3 ladder: cycle
  obs-channel (accepts a warm-start break), action-masking to legal cycle moves, or residual RL on a
  cycle follower. Each must be measured against the **v9.5 control (0.817)**, not v9 (0.782).
- **(B) Bank v9.5, drop shaping.** Log the warm-restart gain as the result, chase the residual ~34%
  self-trap another way (e.g. a second warm-restart, longer ent-anneal, or a stronger structural prior
  directly).
- Multi-seed any future claim — single-seed deltas here (~0.006 cov) are within run noise.
- Carried: re-evaluate the BFS channel (open-Q#5).

## Sources

- `runs/0603_01_coverage-10x10-cycle/` — v10 β=0.03 run: config, metrics, checkpoints (best.pt coverage-selected), videos
- `runs/0603_04_coverage-10x10-cycle-ctrl/` — β=0 control (the decisive A/B); `runs/_overnight_v10.log` — orchestrator chain log
- [coverage_10x10_v10.yaml](../../config/training/coverage_10x10_v10.yaml); warm-start lineage v9 (`runs/0602_12_coverage-10x10-transfer`)
- [full-coverage-design.md](../full-coverage-design.md) §9 D19, §10 open-Q#3, Phase 2
- Gate: [scripts/verify_cycle_shaping.py](../../scripts/verify_cycle_shaping.py); commit `1e8c5db`
