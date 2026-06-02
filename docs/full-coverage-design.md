# Full-Grid Coverage Design

Snake RL — design doc for the **100% grid-coverage** end goal.

> **Status**: living document. Future iterations should read this first, run against it,
> and append results / new decisions here. It is the authority for the coverage goal and
> **supersedes the "Hamiltonian / perfect-play is a non-goal" stance in
> [design-packet.md](design-packet.md) §3** (see D12).

---

## 1. End goal

A learned agent (PPO) that drives the snake to **fill every cell of the grid** — a perfect
game. Length `rows*cols` with no empty cells left. The previously-discussed mental model is
an **inward spiral / space-filling traversal**; that behavior is what coverage looks like in
practice, but the goal is for it to *emerge from learning*, scaffolded only where pure RL
provably cannot reach it.

## 2. Feasibility reality (read before planning runs)

These are hard constraints, not tuning knobs:

1. **Perfect fill is a Hamiltonian-path problem.** Filling the grid traces a Hamiltonian path;
   a *repeatable* perfect strategy traces a Hamiltonian *cycle*.
2. **Parity constraint.** A Hamiltonian cycle exists only on grids with an **even number of
   cells**. On odd-cell grids (e.g. 5×5 = 25, 7×7 = 49) a full cycle does not exist and 100%
   from an arbitrary start can be impossible by parity. **All curriculum grids must be
   even-sided** (4×4, 6×6, 8×8, 10×10 …). The current 32×32 = 1024 is even (fine).
3. **The exploration wall.** Pure from-scratch RL reliably reaches 100% only on *small* grids.
   Past roughly 8×8–12×12 the agent will essentially never *randomly* discover a near-perfect
   fill, regardless of training time. This is why field-standard "perfect snake" agents use a
   Hamiltonian cycle (often with learned/heuristic shortcuts), not tabula-rasa RL.
4. **The current reward actively fights coverage** (see §3): the step penalty and distance
   shaping reward short, direct paths — the opposite of the long, indirect routes filling
   requires. Note score already equals coverage (each food = one new cell, `length = score+1`),
   so the *objective sign* is correct; the problem is the late-game disincentive, not direction.

## 3. Why the existing setup cannot reach the goal as-is

From the current env ([snake_env.py](../snake_env.py)) and the run history:

- Reward today: `+1` food, `−1` death, `−0.025`/step, `+0.01·Δmanhattan` shaping.
- **No win condition exists.** When the grid fills, `_place_food` drops food on the head and
  the episode never terminates as a win — coverage is neither detected nor rewarded.
- **Late-game food is net-negative.** When the grid is crowded, reaching the next food costs
  more than `+1` in accumulated step penalty, so the policy learns to *stop growing*. Best
  result to date plateaus at length ~95 / 1024 (~9%).
- Distance shaping pushes greedy beelines that create self-traps.

## 4. Plan (phased)

The architecture (hybrid 4-channel obs + BFS channel + PPO) is kept. We change the
**objective** and add a **curriculum**, then add a **structural prior** only where RL gives out.

### Phase 0 — Fix the objective + instrumentation  *(implemented, un-executed)*

Make coverage learnable and measurable before scaling anything:

1. **Win detection + termination** in the env: `len(snake) == rows*cols` → terminate, cause
   `"win"`, add a configurable `win_bonus`.
2. **Coverage-oriented reward variant**, gated behind config so existing runs are untouched:
   configurable `step_penalty` (set ~0; γ=0.99 already rewards efficiency) and `win_bonus`;
   distance shaping off (`dist_shaping_alpha=0`).
3. **Coverage metrics**: log `avg_coverage`, `max_snake_length`, and win rate (`death/win_pct`).
4. **6×6 config** (`config/training/coverage_6x6.yaml`) sized for a small even grid.

**Phase 0 acceptance gate (the first experiment to run):** train on 6×6 and confirm the agent
can reach 100% (length 36) with non-trivial frequency. If it cannot fill a 6×6, the reward is
still wrong and must be fixed before any scaling.

### Phase 1 — Pure RL + curriculum on small even grids

Start at 4×4/6×6 where 100% is genuinely learnable; transfer/grow the grid size as each scale
is mastered (curriculum). Goal: show coverage (spiral-like behavior) *emerges* from learning.
Curriculum advances when the win rate at the current size clears a threshold.

### Phase 2 — Structural prior where pure RL plateaus

At the grid size where pure RL stalls below 100% (likely ≥10×10), add an **optional
Hamiltonian/spiral guidance reward**: shape toward following a space-filling cycle and let RL
learn safe shortcuts off it. This is the only thing that *guarantees* 100% on large grids while
remaining a learning agent with a safety scaffold. Design TBD; revisit after Phase 1 data.

## 5. Reward design (coverage mode)

| Term | Value | Rationale |
|---|---|---|
| Food eaten | `+1` | One new cell covered; return == coverage. |
| Death (wall/body) | `−1` | Unchanged. |
| Win (grid full) | `+win_bonus` | Sharp terminal signal for the actual goal. |
| Step penalty | `~0` (configurable) | Removing it stops punishing the long fill paths. γ supplies efficiency pressure. |
| Distance shaping | off (`alpha=0`) | Causes greedy self-traps; counterproductive for filling. |

## 6. Curriculum mechanism (Phase 1, not yet built)

Sketch for the next iteration: train at grid size `k×k`; when rolling win rate ≥ threshold over
a window, advance to the next even size and resume from the current policy. Open question
whether the CNN transfers across sizes (the flatten dim changes — likely needs a size-agnostic
head, e.g. global pooling, or per-size fine-tune from a shared encoder).

## 7. Architecture note for small grids

`HybridActorCritic` applies `MaxPool(2,2)` after every conv but the last
([model.py:199-200](../model.py#L199-L200)). On 6×6 with the default `num_layers=4`, spatial
dims collapse below 1 and the model breaks. **Small grids must reduce `num_layers`** (6×6 uses
`num_layers: 2` → 6→3 after one pool, last conv keeps 3×3, flatten = 32·9 = 288). This will
need a principled fix (adaptive pooling) in Phase 1 to support a single transferable encoder
across the curriculum.

## 8. Success criteria

- **Phase 0 gate**: ≥1 in N episodes reaches length 36 on 6×6 (any non-trivial win rate).
- **Phase 1**: ≥90% win rate on 6×6 and 8×8; some wins on 10×10.
- **End goal**: reliable 100% fill at the target grid size (32×32 via Phase 2 prior).

## 9. Decision log

### D12: Reframe end goal to 100% coverage (supersedes Hamiltonian non-goal)
- **Decision**: The project's end goal is now perfect grid coverage. The design-packet stance
  that Hamiltonian/perfect-play is a non-goal is superseded; perfect play *is* the target, and a
  structural prior is explicitly on the table (Phase 2).
- **Why**: User goal. Pure food-seeking optimization tops out at ~9% coverage; the objective
  itself must change.

### D13: Even-sided curriculum grids (parity)
- **Decision**: Only even-cell grids in the curriculum (4×4, 6×6, 8×8, …); 32×32 as the eventual
  target.
- **Why**: Hamiltonian cycle / full fill from arbitrary start requires an even cell count (§2.2).
- **Alternatives**: Odd grids — rejected, 100% can be impossible by parity.

### D14: Coverage-oriented reward (configurable step penalty + win bonus, shaping off)
- **Decision**: Add configurable `step_penalty` (default keeps `−0.025`; coverage mode ~0) and
  `win_bonus` (default `0.0`); coverage runs disable distance shaping.
- **Why**: The `−0.025` step penalty makes late-game food net-negative and caps growth; distance
  shaping causes self-traps. Discounting (γ=0.99) already rewards efficiency without a per-step
  penalty.
- **Alternatives**: Keep step penalty but add survival bonus — rejected as redundant given the
  win bonus and γ. Per-new-cell reward — unnecessary; food reward already equals new-cell reward.

### D15: Win condition + termination + coverage metric
- **Decision**: Terminate with cause `"win"` when `len(snake) == rows*cols`; log `avg_coverage`,
  `max_snake_length`, and win rate.
- **Why**: The goal was previously neither detectable nor rewarded; can't optimize or measure
  what isn't logged.

### D16: Small grids require reduced CNN depth (interim)
- **Decision**: 6×6 uses `num_layers: 2`. A size-agnostic encoder (e.g. adaptive/global pooling)
  is deferred to Phase 1 when curriculum transfer is built.
- **Why**: Default `num_layers: 4` + `MaxPool(2,2)` collapses 6×6 below 1×1 (§7).
- **Status**: superseded for the curriculum by **D17** (the deferred adaptive head is now built).

### D17: Size-agnostic encoder via adaptive pooling (v8)
- **Decision**: Opt-in `model.adaptive_pool_size: int|null` (default null). When set to `N`, the
  hybrid CNN drops the intermediate `MaxPool(2,2)` layers and inserts `AdaptiveMaxPool2d(N)` before
  `Flatten`, so the CNN flatten dim = `channels[-1]·N·N` is **identical across all grid sizes**.
  Phase 1 uses `N=4`. A guard raises if the pre-pool spatial dim < `N` (degenerate upsample).
- **Why**: The flatten dim was computed from a fixed `(rows,cols)` dummy → a smaller-grid checkpoint
  could not load into a larger-grid model, blocking the curriculum (open-Q#2). Adaptive pooling makes
  every parameter shape grid-invariant, enabling weight transfer.
- **Why null default**: keep v6/v7 (and the live runs' saved configs) byte-for-byte reproducible.
- **Why N=4, MaxPool removed**: with MaxPool kept, 6×6 collapses to 3×3 pre-pool, so `N=4` would
  upsample degenerately; removing MaxPool keeps pre-pool spatial == grid size (6≥4, safe). The 4×4
  bottleneck also retains more spatial detail than v6's post-MaxPool 3×3 flatten (no 6×6 regression).
- **Verified (v8)**: strict 6×6→8×8/10×10 load; default path unchanged (flat 288 @6×6); no 6×6 win
  regression (70.7% vs v6 67%).
- **Alternatives**: global pool (1×1) — discards spatial detail, rejected; fixed max-canvas + masking
  — more env surgery, deferred.

### D18: Warm-start curriculum via `--init-from` (not in-place resume) (v8)
- **Decision**: `train.py --init-from <ckpt|run>` seeds a **fresh** run with agent weights only
  (strict load), fresh optimizer, `start_update=0`, `best_avg_return` reset. The curriculum advances
  by warm-starting the next grid size from the previous stage's checkpoint.
- **Why**: `--resume <run> rows=k cols=k` resumes **in place** (overwrites the seed dir, continues the
  prior optimizer/update-count) — wrong for a new stage. And cp-`best.pt`+`--resume` loads silently
  wrong: `best.pt` is a full dict, so the resume path reloads stale optimizer/update with **no shape
  error** to catch it (shapes are grid-invariant under D17). `--init-from` makes the warm-start
  explicit and **fails visibly** on a genuine shape mismatch (e.g. transfer without D17).
- **Result (v8)**: 8×8 warm-start from a 6×6 seed → 70% win vs 56% from-scratch, ~3× faster.

## 10. Open questions

1. How large a grid can pure RL fill (Phase 1 → Phase 2 boundary)? Empirical, find via curriculum.
2. Does the encoder transfer across grid sizes, or do we need a global-pooling head / per-size
   fine-tune?
3. Phase 2: shaping-toward-Hamiltonian vs. action-masking to legal cycle moves vs. residual RL on
   top of a cycle follower — which gives the best efficiency/guarantee trade-off?
4. Right `win_bonus` magnitude relative to `+1` food and `−1` death so it's a clear but not
   destabilizing terminal signal.
5. Does the BFS channel (D3) actually help coverage, or is it noise on small grids? Re-evaluate
   per scale.

## 11. Implementation status

- **Phase 0**: implemented and **executed — acceptance gate PASSED**. Run
  `runs/0526_23_coverage-6x6` (2026-05-26). By update ~2393 (~20 min, ~13.5k SPS): **avg
  coverage 1.0**, **win rate ~67%** (`death/win_pct`), avg return 45.0, entropy ~0.28 (settled,
  no collapse), clip_frac ~0.025. The coverage reward (step_penalty 0 + win_bonus 10, shaping
  off) was sufficient for pure RL to learn full space-filling on 6×6. Manually stopped (well past
  the gate); `best.pt` is the policy of record. No `agent_final.pt` (killed before final save).
- **Phase 1** (curriculum on larger even grids, size-agnostic encoder): **in progress.**
  - **v7** (`runs/0601_22_coverage-8x8`, 2026-06-01): 8×8 pure RL from scratch, same recipe as v6,
    only grid size changed → **56% win, 88% coverage** in 4h. Pure RL scales one step up; gentle
    degradation, not a wall.
  - **v8** (seed `runs/0602_02_coverage-6x6-v8` + transfer `runs/0602_04_coverage-8x8-v8`,
    2026-06-02): built the **size-agnostic adaptive-pool encoder (D17)** and **`--init-from`
    warm-start (D18)**. 6×6 seed = 70.7% win (no regression vs v6). 8×8 warm-started from it →
    **70% win, ~3× faster** than v7 from-scratch. Encoder transfer (open-Q#2) **confirmed.** Death
    profile shifted wall→body: wall-avoidance transfers, self-trapping is the scaling bottleneck.
  - **Next (v9):** 10×10 warm-start to find the pure-RL ceiling (the Phase 1→2 boundary).
- **Phase 2** (Hamiltonian/spiral prior): not started.
</content>
</invoke>
