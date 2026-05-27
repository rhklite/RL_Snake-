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

- **Phase 0 code**: implemented, **not yet executed** — env win/coverage/reward knobs, training
  coverage logging, and `config/training/coverage_6x6.yaml`. First run to perform is the Phase 0
  6×6 acceptance gate (§4, §8).
- Everything else (curriculum, Phase 2 prior): not started.
</content>
</invoke>
