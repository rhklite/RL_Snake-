# Snake-RL — Iteration History (challenge → test → solution → gap chain)

> **Purpose.** A single, LLM-consumable narrative of *every* iteration from project conception to the
> current frontier. Each entry is framed as **Challenge → What was tested → Solution → Gap**, and the
> **Gap of `vN` is the Challenge of `vN+1`** — the chain is closed end to end. This is the spine;
> per-version files in this folder ([v1](v1-baseline.md)…[v10](v10-cycle-shaping-10x10.md)) hold full
> detail for v1–v10, and v11–v14 detail lives **here** (no standalone files yet). Decision IDs
> `D1`–`D19` are defined in [design-packet.md](../design-packet.md) (D1–D11) and
> [full-coverage-design.md](../full-coverage-design.md) (D12–D19).
>
> **The one throughline:** the project's hard problem is **self-trapping (body-collision deaths)**.
> It first appears at v8, becomes *the* wall at v9, and every iteration after is an attempt to move it.
> Wall-avoidance was solved early and transfers for free; self-avoidance at scale is the entire game.

---

## Spine — the chain at a glance

| Ver | Date | Grid | Challenge inherited | Intervention tested | Outcome | Residual gap → next |
|----|------|------|---------------------|---------------------|---------|----------------------|
| v0 | –2026-03 | 12–64² | none (bootstrap) | plain CNN on raw int8 grid | weak learning, pipeline works | objective + encoding both unproven |
| [v1](v1-baseline.md) | 03-15 | 32² | weak baseline | PPO, single int8 grid, step pen −0.025 | return 27.8 / len 45.8; **zigzag hacking** | ordinal grid → CNN can't read head/body/age |
| [v2](v2-body-gradient.md) | 03-16 | 32² | zigzag, blind encoding | body-gradient + per-element channels, 4-layer CNN (D1,D2) | **+80%** → len 83; zigzag gone | **topology blindness** — can't see connectivity through gaps |
| [v3](v3-bfs-maxpool-removed.md) | 03-16 | 32² | topology blindness | +BFS channel **and** remove MaxPool **and** 1×1 bottleneck (3 at once) | **FAILURE** — len ~1, never ate | 3 confounded changes; can't attribute the collapse |
| [v4](v4-bfs-isolated.md) | 03-17 | 32² | confounded failure | BFS isolated (MaxPool restored) + mid-run tail/BFS-ramp fix (D10,D11) | plateau len ~95→105 | still ~10% fill; D10+D11 applied together (unclean) |
| [v5](v5-fromscratch-validation.md) | 03-20 | 32² | is the design sound from scratch? | full from-scratch rerun, 1024 envs | return 64 / len 107 (~10% fill) | **the *objective* is the ceiling, not the network** |
| [v6](v6-coverage-6x6.md) | 05-26 | 6² | objective is wrong | **PIVOT → 100% coverage**: win+10, step pen 0, shaping off (D12–D16) | **SUCCESS** — ~100% cov, 67% win | only 6×6; CNN-depth hack; encoder not size-agnostic |
| [v7](v7-coverage-8x8.md) | 06-01 | 8² | does coverage scale? | same recipe, only grid 6→8 | **SUCCESS** — 88% cov, 56% win (gentle, not a wall) | every scale needs a from-scratch retrain; body deaths appear |
| [v8](v8-adaptive-transfer-8x8.md) | 06-02 | 6→8 | no transfer; per-scale retrain | adaptive-pool encoder (D17) + `--init-from` warm-start (D18) | **SUCCESS** — 70% win vs 56%, ~3× faster | **self-trap (body) deaths are the scaling bottleneck** |
| [v9](v9-ceiling-10x10.md) | 06-02 | 10² | where does pure RL break? | scratch vs chained-transfer ceiling probe | scratch walls 63%/16%; transfer **78%/48%** then plateaus | **the self-trap wall — body ~47%; neither RL nor warm-start removes it** |
| [v10](v10-cycle-shaping-10x10.md) | 06-03 | 10² | self-trap wall needs a structural prior | Hamiltonian cycle-shaping reward, β=0.03, bonus-only (D19); β=0 control | **NEGATIVE** — control 0.817 ≥ shaped 0.811; β=0.03 **inert** | prior too weak; *real* gain was the **warm-restart (v9.5 = 0.817)** |
| v11 | 06-05 | 12² | crack the wall from the value side | asymmetric actor-critic (privileged cycle critic) | **NEGATIVE** — 0.70 vs baseline; critic can't teach the actor | **β reward bonus is the only lever that has ever moved the wall** |
| v12 | 06-04→05 | 10²/12² | β=0.03 inert — is the *prior* dead or just too weak? | escalate β 0.03→0.08; 3-seed de-risk | **β registers** — 10×10 **0.890±0.009** (3 seeds); β=0.08 @12×12 ceilings ~0.80 | dose-response real but β=0.08 won't scale; how high can the dose go? |
| v13 | 06-10 | 10² | how far does dose-response go, stably? | co-scale **β=0.15 / win_bonus=20**; probe β=0.18 | **0.939 @10×10** (best ever); β=0.18 peaks 0.955 but **decays** | β=0.15/win=20 = the **locked stable recipe**; does it climb the grid ladder? |
| v14 | 06-10→12 | 12→16→20² | scale the locked recipe up | curriculum ladder A/C/D at β=0.15/win=20 | 12²=**0.903**, 16²=**0.82**, 20²=**0.645** (82% body) | **WALL at 20×20 (400 cells)** — self-avoidance collapses |
| v15 | 06-28 | 20² | wall = 82% body deaths; β reward lever tapped out | **action-masking** the action space (open-Q#3 opt 2): safety (mask wall/body suicide) + cycle | safety **0.716 peak / 18% win** (beats v14's 0.645); cycle timeout-defeated 0.21 | masking lifts the ceiling but is **unstable late** (failure shifts body→timeout); **current frontier** |

**Verified-number note.** v12–v15 figures are recomputed from `runs/*/metrics.jsonl` (roll-100 `avg_coverage`,
final death mix), not just memory. v1–v11 figures are from the per-version docs / design packet.

---

## v0 — Pre-history (unnumbered)

- **Challenge.** Nothing exists; need a working PPO + vectorized-env pipeline.
- **Tested.** Pure CNN on a raw int8 grid at 12×12 and 64×64 (throwaway runs, now in [runs/archive/](../../runs/archive/)).
- **Solution.** Pipeline works; a plain CNN on a raw grid learns *weakly*.
- **Gap.** Both the **observation encoding** and the **reward objective** are unproven — opens v1.

## v1 — Baseline PPO (32×32)

- **Challenge.** Establish a real baseline with a defined reward.
- **Tested.** PPO, single int8 grid (0/1/2/3), step penalty raised to −0.025 (D8); δ-Manhattan shaping (D7).
- **Solution.** return 27.8, len 45.8. The agent eats but exhibits **zigzag / lawnmower reward-hacking**
  (postmortems `0315_14_GE`, `0315_2100_OP`).
- **Gap.** The ordinal int8 grid forces the CNN to learn that "2=head" vs "1=body" numerically, and carries
  **no segment-age** signal → brittle, hackable. → v2.

## v2 — Body-gradient + per-element channels (32×32)

- **Challenge.** Make the grid legible to the CNN and kill zigzag.
- **Tested.** Body-gradient channel (head=1.0→tail, encodes age, **D1**) + separate head/body/food channels
  (**D2**); CNN deepened to 4 layers.
- **Solution.** **+80%** (len 45.8→83, return ~49.9); zigzag eliminated (confirmed `0316_0848_SYNTH`).
- **Gap.** **Topology blindness** — a 4-layer 3×3 CNN (RF ~15–20) cannot tell whether two regions are connected
  through a narrow gap on a 32×32 board (postmortem `0316_0833_OP`). → v3.

## v3 — BFS channel, MaxPool removed, 1×1 bottleneck (32×32)

- **Challenge.** Give the network the connectivity information it can't compute.
- **Tested.** Added a **BFS-reachability** 4th channel (D3,D4,D5) **and** removed MaxPool **and** added a 1×1
  bottleneck — three architectural changes in one run.
- **Solution.** **Catastrophic failure** — snake length ~1, the policy stopped eating entirely
  (postmortem `0317_0015_OP`).
- **Gap.** Three confounded changes → the collapse can't be attributed (open question: was it MaxPool removal
  or the bottleneck?). → v4.

## v4 — BFS isolated + mid-run encoding fixes (32×32)

- **Challenge.** Recover from v3; isolate the BFS channel on a known-good architecture.
- **Tested.** BFS kept, **MaxPool restored**, 1×1 bottleneck dropped. Mid-run at update ~9000, two encoding
  fixes applied *together*: linear body-gradient remap to [0.1,1.0] (**D10**, tail was vanishing under ReLU)
  and BFS ramp to full by length 51 (**D11**).
- **Solution.** Broke a size-~95 plateau → ~105; training stable again.
- **Gap.** Still only ~10% of a 32×32 grid filled. D10 and D11 were never isolated (clean-ablation debt). → v5.

## v5 — From-scratch validation (32×32)

- **Challenge.** Is the v4 design genuinely sound, or an artifact of mid-run patching?
- **Tested.** Full from-scratch rerun of the v4 design, 1024 parallel envs.
- **Solution.** return 64, len 107 — reproduces v4. The stack is sound.
- **Gap — the pivot insight.** Food-seeking **tops out at ~10% grid fill on 32×32 regardless of observation
  richness**. The bottleneck is the **objective**, not the network. → motivates the v6 reframe.

> **Design break (Mar 20 → May 26).** Wrote [full-coverage-design.md](../full-coverage-design.md): the end goal
> is reframed from food-seeking to **100% coverage** (D12), which is a Hamiltonian-path problem requiring
> **even-celled grids** (D13) and a **win condition + coverage reward** (D14,D15). Plan: pure RL on small grids
> (Phase 1), structural prior where it breaks (Phase 2).

## v6 — Coverage pivot, Phase 0 gate (6×6)

- **Challenge.** Can pure RL fill a *small even* grid if the objective is fixed?
- **Tested.** Coverage reward: **win_bonus +10**, **step_penalty 0**, shaping off, +1 food / −1 death (D14);
  win-detection + coverage metrics (D15); `num_layers=2` to fit 6×6 (D16).
- **Solution.** **PASS** — ~98–100% coverage, **67% win**, stable entropy. Same v4/v5 obs+net stack, only the
  *reward* changed → confirms v5's "objective was the ceiling."
- **Gap.** Validated only at 6×6; CNN depth is a per-grid hack; the encoder can't transfer across sizes
  (flatten dim is grid-dependent). → v7.

## v7 — Clean scale-up (8×8)

- **Challenge.** Does the coverage recipe survive one step up in scale?
- **Tested.** Identical recipe, **only** grid 6×6→8×8.
- **Solution.** **88% coverage, 56% win** in ~4h — a **gentle degradation, not a wall**. Pure RL scales one rung.
- **Gap.** Each new size needs a **from-scratch retrain** (slow), and **body-collision deaths begin to grow** as
  episodes lengthen. → v8.

## v8 — Size-agnostic encoder + warm-start curriculum (6×8)

- **Challenge.** Make one policy transferable across grid sizes (remove the retrain tax).
- **Tested.** `AdaptiveMaxPool2d(N=4)` head → grid-invariant flatten dim (**D17**); `--init-from` warm-start that
  seeds a fresh run with weights only, fresh optimizer (**D18**). 8×8 warm-started from a 6×6 seed.
- **Solution.** **70% win vs 56% from-scratch, ~3× faster**; 6×6 seed unregressed (70.7%). Encoder transfer
  **confirmed** (open-Q#2 closed).
- **Gap.** The death profile shifts **wall→body**: wall-avoidance transfers perfectly, but **self-trapping
  (body deaths) is now the scaling bottleneck** and does *not* transfer. → v9.

## v9 — Pure-RL ceiling probe (10×10)

- **Challenge.** Where exactly does pure RL break, and does the curriculum push past it?
- **Tested.** Same adaptive arch, two inits: from-scratch vs chained 8×8→10×10 transfer.
- **Solution.** From-scratch **walls at 63% cov / 16% win** (never hits 75%). Transfer reaches **78% / 48%**
  (peak 0.974), ~5–7× faster — but **also plateaus**. **~10×10 is the Phase 1→2 boundary** (open-Q#1 answered).
- **Gap — the wall, stated.** Wall deaths collapse 37.5%→5.4% (solved, transfers), but **body/self-trap deaths
  are ~47% in *both* runs** — the dominant, scale-dependent failure. A **reactive PPO policy cannot learn the
  long-horizon planning to avoid boxing itself in.** Needs a structural prior. → v10.

## v10 — Hamiltonian cycle-shaping prior (10×10) · Phase 2 opens

- **Challenge.** Add a structural prior that targets self-trapping specifically.
- **Tested.** Reward-only, **bonus-only** cycle-alignment term: `+β` for the on-cycle successor move, 0 off-cycle,
  **β=0.03**, warm-start-safe (obs stays 4-ch, v9 checkpoint loads tensor-for-tensor) (**D19**). **Critically, a
  `β=0` control** warm-started from the same v9 checkpoint.
- **Solution.** **NEGATIVE for the prior.** Control (β=0) reached **0.817 cov / 62% win / 34% body** ≥ shaped
  (β=0.03) **0.811 / 60% / 35%**. β=0.03 is **inert** — its discounted ceiling (3.0) is swamped by the
  coverage/win signal.
- **Sub-result v9.5 (the real find).** The *entire* v9→v10 gain was the **fresh-optimizer warm-restart**: just
  re-training v9 with a reset optimizer + re-exploration moved **cov 0.78→0.82, win 48→62%, body 47→34%**. v9 was
  never converged. **New baseline to beat = 0.817, not 0.782.**
- **Gap.** Is the cycle prior *dead*, or just *too weak at β=0.03*? Process lesson: gate the headline on the
  single-flag control. → forks into v11 (value-side) and v12 (stronger β).

## v11 — Asymmetric actor-critic (12×12) · negative side-branch

- **Challenge.** If the *reward* prior is inert, can a **privileged critic** (one that sees the cycle) crack the
  wall from the value side instead?
- **Tested.** Asymmetric actor-critic: critic gets a privileged cycle feature, actor keeps the 4-ch obs
  (commit `ecf8050`, run `runs/0605_04_coverage-12x12-asym`).
- **Solution.** **NEGATIVE** — underperformed the shared-critic baseline (roll-cov peak **0.767**, final 0.696
  vs baseline ~0.79). Adversarial critique confirmed it's structural: the critic is **only a GAE baseline**, so
  it can't reach the wall the *actor* hits; and the cycle is deployment-computable, so hiding it from the actor
  buys nothing.
- **Gap.** Value-side tricks **cannot teach the policy a new action**. The **β reward bonus is the only lever
  that has ever moved the self-trap wall** → double down on it. Line **retired**. (Caveat logged: 12×12 under the
  plateau-watcher is *undertrained* — long episodes make windowed coverage noisy and trip patience early; 12×12
  needs a fixed long run for a true ceiling.) → v12.

## v12 — Escalate the prior + de-risk (10×10, 12×12)

- **Challenge.** β=0.03 was inert (v10) and the value side is dead (v11) — does a **stronger** cycle bonus
  register, and is the result real (everything so far is seed=1)?
- **Tested.** Raise **β 0.03→0.08** at 10×10 (`runs/0604_15…b08`); confirm with **3 seeds**
  (`runs/0605_10…s2`, `runs/0605_15…s3`); push β=0.08 up to 12×12 (`runs/0604_23…12x12…b08`).
- **Solution.** **The prior registers at β=0.08.** 10×10 reaches **0.890 ± 0.009** across 3 seeds
  (final cov 0.891 / 0.896 / 0.885; win ~70%; body ~26%) — a real lift over the v9.5 0.817 baseline, **n=1 risk
  retired**. Dose-response is real (β=0.03 inert → β=0.08 works).
- **Gap.** β=0.08 **does not scale**: at 12×12 it peaks ~0.82 then **decays to ~0.69** — ~0.80 is its 12×12
  ceiling. The dose that works at 10×10 is too weak one rung up → **raise the dose**. → v13.

## v13 — Dose-response & stability: lock β=0.15 / win_bonus=20 (10×10)

- **Challenge.** How high can the cycle dose go before it pays off / destabilizes?
- **Tested.** Co-scale the terminal and shaping signals: **β=0.15 / win_bonus=20** at 10×10
  (`runs/0610_01…b15-w20`); probe one notch hotter, **β=0.18** (`runs/0610_18…b18-w20`).
- **Solution.** **0.939 cov @10×10 — the best 10×10 result of the project** (win 77%, body 19%, timeout 0%).
  Dose-response continues. But **β=0.18 is unstable**: it *peaks higher* (0.955) then **decays to 0.897** late
  (same decay signature as over-hot runs; not cycle-walking — timeouts 0%).
- **Solution, locked.** **β=0.15 / win_bonus=20 is the stable curriculum recipe** — the rule correctly prefers
  stability over β=0.18's higher-but-collapsing peak.
- **Gap.** Validated only at 10×10. Does the locked recipe climb the **grid ladder**? (Parked: LR-decay/β-anneal
  late phase might lock in the ~0.95 peak.) → v14.

## v14 — Curriculum scaling ladder (12×12 → 16×16 → 20×20)

- **Challenge.** Scale the locked β=0.15/win=20 recipe up the even-grid ladder via warm-start.
- **Tested.** A: 12×12 (`runs/0610_13…`); C2: 16×16 (`runs/0610_22…`, resumed); D2: 20×20
  (`runs/0611_15…-v2`, warm-started from converged C2). All max_hours=0, coverage-plateau-watcher stops, run
  via `runs/_chain_cd2.sh`.
- **Solution — clean scaling, then a wall.**
  - **12×12 → 0.903** (win 61%, body 29%) — closes the gap β=0.08 left at ~0.80; curriculum + strong prior scales.
  - **16×16 → 0.82** (roll-peak 0.821 @upd~3097, decays to 0.773; body 40%).
  - **20×20 → 0.645** (roll-peak @upd~2183, final 0.558; **body 82%**, win 5%). Process SIGTERM'd at upd ~2535 on
    2026-06-12 02:41 (clean watcher plateau-stop — entropy/value_loss/SPS all healthy, **not** a crash). Nothing
    running since.
  - Trajectory: 10²=0.939 → 12²=0.903 → 16²=0.82 → **20²=0.645**.
- **Gap — the current open problem.** The recipe that scaled cleanly 10→12→16 **breaks at 20×20 (400 cells)**:
  coverage collapses to ~0.64 and **self-avoidance dominates failure (82% body deaths)**. The snake gets long
  enough to trap itself and the reactive policy + cycle bonus can't plan around 400 cells. This is a **regime
  change, not slow training**. The self-trap wall — dented at every prior step — **reasserts itself at scale**.

---

## v15 — Action-masking the self-trap wall (20×20) · **current frontier**

- **Challenge.** v14's wall: at 20×20 the locked β=0.15/win=20 recipe plateaus at 0.645 with **82% body deaths**,
  and the β reward lever is tapped out (v13 showed β>0.15 destabilizes). Open-Q#3's remaining option: constrain
  the **action space** directly (a mask), not the reward.
- **Tested.** Two masks behind a `mask_mode` flag, warm-started from C2 (16×16, identical init to v14's D2), on the
  new batched env; first NaN-crashed at upд229 (`-inf` mask → `0·log0` gradient) — fixed with a `-1e8` fill + a
  non-finite-grad guard, re-verified (`scripts/verify_action_masking.py` 9/9), then re-run clean past upд229.
  - **safety** (`runs/0628_07_coverage-20x20-mask-safety`): mask any move into wall/body — the env's exact death
    rule, opposite-guard-resolved, with an all-fatal→all-legal fallback. The policy can never 1-step-suicide.
  - **cycle** (`runs/0628_09_coverage-20x20-mask-cycle`): force the Hamiltonian successor — by-construction fill.
- **Solution — masking beats the wall.**
  - **safety → 0.716 roll-cov peak (cov 0.66), win 18%** (~upд600) — clears v14's 0.645 and ~4× the win rate.
    `best.pt` (coverage-selected) preserves the peak policy.
  - **cycle → 0.21**, body 0% / wall 0% (zero collision deaths) but **92% timeout** — under the env's 200·len
    budget the cycle-follower can't reach random food fast enough to grow, so it's timeout-defeated. The ~1.0
    by-construction fill is real (proven by the cycle check in `verify_action_masking.py`) but needs a relaxed
    `max_steps_factor` to show.
  - **Bracket:** v14 shaping 0.645 → **safety 0.716** → cycle (given time) ~1.0.
- **Gap — masking lifts the ceiling but destabilizes late.** safety peaks at 0.716 then **collapses to ~0.18** by
  upд~2200: with suicide forbidden, a degraded late policy **wanders to the timeout (34%)** instead of dying — the
  failure mode shifts **body → timeout**. The peak is preserved in `best.pt`, but the *plateau* is unstable. →v16.

---

## Current status & next step (as of 2026-06-28)

- **Frontier:** v15 / 20×20 action-masking. **safety-mask BEAT v14** — peak 0.716 / 18% win
  (`runs/0628_07_coverage-20x20-mask-safety/checkpoints/best.pt`) vs v14's 0.645 / ~5%. **Open-Q#3 resolves in
  masking's favor: constraining the action space beats the β reward lever at 400 cells.** Nothing training now.
- **Caveat:** safety-mask is **unstable late** (collapses 0.716→0.18 via timeout). The deployable policy is the
  peak (`best.pt`); the plateau is not. cycle-mask was timeout-defeated (0.21) under the 200·len budget.
- **Next steps, in order:**
  1. **Confirm 0.716 with a 2nd seed** — single-seed so far; the project rule is multi-seed any claim. Also tells
     whether the late collapse is a consistent masking property or seed noise.
  2. **Stabilize the peak** — LR-decay / entropy-anneal late phase, or early-stop on the coverage peak, to turn
     the 0.716 peak into a held plateau (the v16 challenge).
  3. **True cycle ceiling** — rerun cycle-mask with a large `max_steps_factor` (200·len defeated it at 0.21).
  4. **Then** revisit the board-size ladder (24×24→…) now that masking moves the wall.

## How the numbering maps to artifacts

- **v1–v10**: formal per-version docs in this folder; configs `config/training/coverage_*_v{6,7,8,9,10}.yaml`.
- **v11**: commit `ecf8050`, run `runs/0605_04_coverage-12x12-asym` (no standalone doc — detail above).
- **v12–v14**: **introduced by this document** to extend the chain. Earlier sessions tracked them by ad-hoc
  labels (**H1/H2/A/B/C/D**) and run dirs, never as v-numbers. They reuse the `coverage_10x10_v10.yaml` config
  with `cycle_beta` / `win_bonus` / grid overrides. Run-dir map:
  - v12 → `0604_15…b08`, `0605_10…s2`, `0605_15…s3`, `0604_23…12x12…b08`
  - v13 → `0610_01…b15-w20` (H1), `0610_18…b18-w20` (B)
  - v14 → `0610_13…b15-w20` (A, 12²), `0610_22…b15-w20` (C2, 16²), `0611_15…-v2` (D2, 20²)
- **v15**: first version since v10 with its **own config + code** (not a v10 override). Commits `20cce3b`
  (action masking) / `25c4c22` (`-1e8` NaN fix + non-finite-grad guard) / `4e667be` (batched `vec_snake_env.py`,
  `training.vectorized_env`). Config `config/training/coverage_20x20_v15.yaml` (`mask_mode`). Runs: safety
  `runs/0628_07…-mask-safety`, cycle `runs/0628_09…-mask-cycle`. Gates: `verify_action_masking.py` (9/9),
  `verify_vec_env.py` (5/5, batched-env equivalence). Local speed thermal-capped; batched env ready for CUDA.
- **Negative/abandoned**: `0611_08_coverage-20x20-b15-w20` (first D attempt, inherited a half-baked C, showed
  first-ever timeouts at 400 cells — abandoned, replaced by the `-v2` D2 run). v15 safety first crash
  `runs/0628_06…-mask-safety` (the `-inf` NaN @upд229, pre-fix — superseded by `0628_07`).

## Decision-log index (for cross-reference)

D1 body-gradient · D2 per-element channels · D3 BFS channel · D4 inverted-distance BFS · D5 BFS length-scaling ·
D6 hybrid CNN+MLP · D7 δ-Manhattan shaping · D8 step penalty −0.025 · D9 Numba BFS · D10 [0.1,1.0] gradient remap ·
D11 BFS ramp to len 51 — *([design-packet.md](../design-packet.md))*; D12 coverage goal · D13 even grids ·
D14 coverage reward · D15 win detection+metrics · D16 per-grid CNN depth · D17 adaptive-pool encoder ·
D18 `--init-from` warm-start · D19 cycle-shaping reward — *([full-coverage-design.md](../full-coverage-design.md))*.
</content>
</invoke>
