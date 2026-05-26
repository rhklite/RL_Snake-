# Design Packet

Snake RL -- Neural Network Architecture & Reward Design

---

## 1. Problem

- A PPO agent learning to play Snake on a grid needs an observation representation and reward structure that let it progress from basic food-seeking to long-term spatial planning (avoiding self-traps at high body length).
- Early runs hit degenerate behaviors (zigzag reward hacking, topology blindness) that could not be solved by hyperparameter tuning alone; the root causes were in observation encoding and reward formulation.

## 2. Goals

- Encode the grid state so the CNN can distinguish head, body segments, food, and spatial reachability without relying on ordinal integer tricks.
- Provide the network with precomputed topological information (BFS reachability) that a finite-receptive-field CNN cannot learn from raw grids.
- Shape rewards to encourage food-seeking without introducing degenerate movement patterns (zigzag, staircase).
- Keep per-step compute overhead acceptable (Numba JIT for BFS).

## 3. Non-goals

- Hamiltonian-cycle or perfect-play strategies; the goal is learning, not scripted solutions.
- Multi-agent or competitive Snake variants.
- Real-time inference latency optimization (training throughput is the priority).
- Final reward function tuning -- several reward alternatives are proposed but not yet validated.

## 4. Context

- Grid sizes: 12x12 (default) and 32x32 (primary training grid).
- Framework: Gymnasium env + PPO (custom implementation).
- Three observation modes exist: raw integer grid, 11-d feature vector, and hybrid (4-channel grid + 2-d food vector).
- The hybrid mode with the HybridActorCritic model is the current active design.
- Training tracked via Weights & Biases; postmortems written after each significant run.

### Source inventory

| Source type | Identifier | Content |
|---|---|---|
| Transcript | 85f0a9de | BFS reachability discussion, topology blindness, CNN receptive field limits |
| Transcript | 354b9be1 | Body gradient hypothesis formulation |
| Transcript | 00040cf8 | Postmortem confirming gradient + per-element channels reduced zigzag |
| Transcript | 0d676732 | Hybrid config, 3-channel setup |
| Postmortem | 0315_14_GE.md | Baseline run analysis (Gemini), zigzag/lawnmower diagnosis |
| Postmortem | 0315_2100_OP.md | Baseline run analysis (Opus), reward ratio proposals |
| Postmortem | 0316_0830_OP.md | Body-gradient run analysis (Gemini), missing tail channel |
| Postmortem | 0316_0833_OP.md | Body-gradient run analysis (Opus), topology blindness, reward proposals |
| Postmortem | 0316_0848_SYNTH.md | Synthesis of 0316 analyses, merged recommendations |
| Postmortem | 0317_0015_OP.md | BFS reachability run failure analysis |
| Doc | shelved-ideas.md | Multi-channel encoding, step penalty history |
| Run analysis | 0317_22 plateau | Snake size ~95 plateau at update 9000, 44% wall / 56% body deaths, motivating D10 and D11 |
| Code | snake_env.py (root) | Current env with hybrid obs, BFS, reward logic |
| Code | env/snake_env.py | Older env with binary +0.1/-0.1 shaping |
| Code | model.py | HybridActorCritic, GridActorCritic, FeatureActorCritic |

## 5. Requirements

### Functional

- Observation must separate head, body, food, and reachability into distinct CNN channels so the network does not need to disentangle them from a shared value space.
- Body channel must encode segment age (temporal ordering) so the CNN can infer which cells free up soonest.
- BFS channel must use forward-looking body clearance: a cell is traversable if the occupying segment will have moved away by the time the snake arrives.
- Food offset vector (dx, dy) must be available as a direct scalar input to the MLP branch.
- Reward must include food (+1), death (-1), step penalty, and optional distance shaping.

### Non-functional

- BFS computation must not reduce training SPS below ~1500 on a 32x32 grid (Numba JIT target).
- Observation computation must be deterministic given the same game state.

## 6. Proposed Design

### Observation space (hybrid mode)

**Grid tensor**: (B, 4, rows, cols) float32

- Channel 0 -- Body gradient: value = 0.1 + 0.9 * (n - 1 - i) / max(n - 1, 1) for each segment i. Head = 1.0, tail = 0.1. Linear normalization guarantees every segment a unique, evenly-spaced value in [0.1, 1.0] regardless of body length (see D10).
- Channel 1 -- Head location: binary mask, 1.0 at head cell.
- Channel 2 -- Food location: binary mask, 1.0 at food cell.
- Channel 3 -- BFS reachability: inverted distance from head, normalized by max observed distance. Head = 1.0, farther reachable cells decrease toward 0, unreachable = 0. Scaled by min(1.0, (n-1)/50) so it ramps from zero at length 1 to full strength by length 51 (see D11).

**Food vector**: (B, 2) float32 -- normalized signed offset (dx/(cols-1), dy/(rows-1)).

### BFS algorithm

- Numba-JIT compiled BFS from head position.
- Body clearance grid: body_clearance[y,x] = n - i (steps until segment i leaves that cell).
- A neighbor cell is traversable if body_clearance[y,x] <= BFS distance d (forward-looking passability).
- Output: inverted distance map, 1.0 - distance/max_distance for reachable cells, 0 for unreachable.

### Network architecture (HybridActorCritic)

- **CNN encoder**: 4 conv layers, channels [4, 16, 32, 64, 128], kernel 3x3, padding 1, ReLU. MaxPool(2,2) after all but the last layer. Flatten, then linear projection to hidden_size (128).
- **MLP encoder**: Linear(2, 64) + ReLU for the food offset vector.
- **Fusion**: Concatenate CNN features (128) and MLP features (64) = 192. Linear(192, 128) + ReLU.
- **Heads**: Actor Linear(128, 4) with std=0.01 init. Critic Linear(128, 1) with std=1.0 init.
- **Weight init**: Orthogonal initialization (std=sqrt(2)) on all layers except heads.

### Reward structure (current)

- +1.0 for eating food.
- -1.0 for death (wall or body collision).
- -0.025 per step (step penalty).
- +alpha * (prev_manhattan - curr_manhattan) distance shaping, alpha=0.01 default. Only on non-eating steps.

## 7. Decision Log

### D1: Body gradient encoding (head=1.0, tail=1/n)

- **Decision**: Encode body segments as a continuous gradient from head (1.0) to tail (1/n) rather than a binary mask or integer cell values.
- **Why chosen**: Encodes segment age, letting the CNN infer direction of travel, body length, and which cells free up soonest. Empirically eliminated zigzag reward hacking and improved best return by 80%.
- **Alternatives considered**:
  - Single int8 grid with values 0/1/2/3 (baseline). Rejected: CNN must learn that "2 means head" vs "1 means body" -- a brittle numerical distinction.
  - Binary body occupancy channel. Rejected: loses temporal ordering; all body cells look identical.
- **Sources**: transcript 354b9be1, transcript 00040cf8, postmortem 0316_0833_OP.md, postmortem 0316_0848_SYNTH.md

### D2: Separate per-element channels (head, body, food)

- **Decision**: Give each semantic element its own channel instead of encoding them in a shared value space.
- **Why chosen**: Removes the need for the CNN to disentangle overlapping encodings. Head and food are single-cell binary masks; body is a continuous gradient. The signal per element is cleaner.
- **Alternatives considered**:
  - Single-channel int8 grid. Rejected after shelved-ideas revisit trigger was met (body deaths > 30% at 14% training progress).
  - Multi-channel with frame stacking (6 channels for 2 frames). Deferred: added complexity without clear benefit given the gradient encoding.
- **Sources**: shelved-ideas.md item 1, postmortem 0316_0848_SYNTH.md

### D3: BFS reachability as 4th channel

- **Decision**: Precompute BFS reachability from the head and provide it as a 4th observation channel.
- **Why chosen**: The body gradient encodes segment age but not spatial connectivity. A 4-layer CNN with 3x3 kernels has an effective receptive field of ~15-20 cells; it cannot determine if distant regions are connected through a narrow gap on a 32x32 grid. BFS is an iterative graph algorithm that standard CNNs empirically struggle to approximate.
- **Alternatives considered**:
  - Rely on deeper/wider CNN to learn reachability from raw grid. Rejected: function approximation problem that CNNs struggle with; receptive field is fundamentally limited.
  - Flood-fill count as a scalar observation. Considered but a full spatial map was preferred for richer signal.
  - Strict passability (body cells are always obstacles). Rejected in favor of forward-looking passability (cells become traversable when the occupying segment will have cleared by arrival time).
- **Sources**: transcript 85f0a9de, postmortem 0316_0833_OP.md, postmortem 0316_0848_SYNTH.md

### D4: Inverted distance encoding for BFS channel

- **Decision**: Encode BFS as 1.0 at head, decreasing toward 0 for distant reachable cells, 0 for unreachable.
- **Why chosen**: Avoids conflating head (distance 0) with unreachable cells (both would be 0 in raw distance encoding). The inverted form gives the CNN a natural gradient from "here" to "far away."
- **Alternatives considered**:
  - Raw BFS distance. Rejected: head and unreachable both map to 0.
  - Binary reachable/unreachable mask. Rejected: loses distance information.
- **Sources**: transcript 85f0a9de

### D5: BFS channel scaling by snake length ratio

- **Decision**: Multiply BFS channel by (n-1)/(rows*cols-1) so it is near-zero when the snake is short.
- **Why chosen**: At length 1 all cells are reachable with near-identical values; the BFS channel is uninformative and adds noise. Scaling suppresses it early and lets it grow as topology becomes relevant.
- **Alternatives considered**: None recorded. (Assumption: introduced to address the chicken-and-egg problem identified in postmortem 0317_0015_OP.md where BFS noise at length 1 contributed to premature policy collapse.)
- **Sources**: snake_env.py (root), postmortem 0317_0015_OP.md (problem identification)

### D6: Hybrid CNN + MLP architecture

- **Decision**: Use a CNN for the 4-channel grid and a separate MLP for the 2-d food offset vector, fused before actor/critic heads.
- **Why chosen**: (Assumption) The food offset is a scalar directional signal that the CNN would need extra layers to extract from a single hot pixel on a large grid. The MLP provides a direct shortcut path. No explicit rationale was recorded in transcripts.
- **Alternatives considered**: No explicit alternatives recorded.
- **Sources**: model.py (HybridActorCritic), transcript 0d676732

### D7: Distance shaping reward (current: delta-Manhattan, alpha=0.01)

- **Decision**: Add alpha * (prev_manhattan_dist - curr_manhattan_dist) per non-eating step.
- **Why chosen**: Provides dense food-seeking signal without destabilizing training. Replaced the older binary +0.1/-0.1 shaping.
- **Alternatives considered**:
  - Binary +0.1/-0.1 for closer/farther (older env). Replaced: does not scale with distance change magnitude.
  - Potential-based shaping: gamma * phi(s') - phi(s), phi = -manhattan_distance. Recommended but not yet implemented. Policy-invariant; eliminates staircase bias.
  - Remove shaping entirely (rely on food reward + step penalty). Considered as fallback if potential-based is too disruptive.
  - Higher alpha (0.05) with lower step penalty (-0.01). Proposed to flip the ratio so food-approach dominates. Risk: destabilize early training.
- **Sources**: postmortem 0315_2100_OP.md, postmortem 0316_0833_OP.md, postmortem 0316_0848_SYNTH.md, env/snake_env.py (older design), snake_env.py (current)

### D8: Step penalty increased from -0.01 to -0.025

- **Decision**: Raise step penalty to 2.5x the distance shaping signal.
- **Why chosen**: Discourage zigzag/wasted steps. The -0.01 penalty was insufficient; the food reward (+1.0) dominated, so the agent prioritized food-reaching reliability over step efficiency.
- **Alternatives considered**:
  - -0.03 step penalty. Shelved; focus shifted to observation changes instead of further reward tuning.
- **Sources**: shelved-ideas.md item 3, postmortem 0315_2100_OP.md

### D9: Numba JIT for BFS

- **Decision**: Compile the BFS function with Numba njit.
- **Why chosen**: Pure-Python BFS was projected at ~800 SPS. Numba recovered throughput to ~1750 SPS (2x improvement), keeping training viable on 32x32 grids.
- **Alternatives considered**: None recorded. (Assumption: Cython or C extension was not considered due to development overhead.)
- **Sources**: postmortem 0317_0015_OP.md, snake_env.py (root)

### D10: Normalized body gradient (linear remap to [0.1, 1.0])

- **Decision**: Replace the raw body gradient (n - i) / n with a linear normalization: 0.1 + 0.9 * (n - 1 - i) / max(n - 1, 1). This maps head to 1.0 and tail to 0.1 with uniform spacing between all segments.
- **Why chosen**: Run 0317_22 plateaued at snake size ~95 on a 32x32 grid. The original gradient D1 gave the tail a value of 1/95 = 0.0105, effectively invisible after ReLU + MaxPool(2,2). Body collisions accounted for 56% of deaths at the plateau. A naive clamp (max(value, 0.1)) was implemented first but collapsed the last ~9 segments to a shared value of 0.1, destroying temporal ordering in the tail region. The linear normalization fixes both problems: every segment gets a unique value, the tail stays visible at 0.1, and the spacing is uniform (0.9/(n-1) per segment) regardless of body length.
- **Alternatives considered**:
  - No floor (status quo D1, min=1/n). Rejected: empirically correlated with plateau at size ~95 where tail became invisible.
  - Naive clamp max((n - i) / n, 0.1). Implemented first, then replaced: preserves visibility but collapses multiple tail segments to the same value, losing temporal ordering where the snake is most vulnerable to self-collision.
  - Higher floor (0.2). Not tested; 0.1 chosen as the minimum that stays above zero after pooling without compressing the usable range too aggressively.
- **Sources**: run 0317_22 plateau analysis, design-packet.md open question 1, edge case "Tail vanishing under ReLU"

### D11: Softened BFS scaling (ramp to full by length 51)

- **Decision**: Replace the BFS scaling factor (n-1)/(rows*cols-1) with min(1.0, (n-1)/50), reaching full BFS signal strength at snake length 51.
- **Why chosen**: The original scaling factor was proportional to total grid area, yielding only 9.2% signal strength at snake length 95 on a 32x32 grid (94/1023). Topology awareness is most critical at lengths 50-100 where the snake is long enough to partition the grid into disconnected regions but short enough that survival depends on navigating those partitions. The new formula ramps linearly from 0 at length 1 to 1.0 at length 51, giving the policy full BFS information throughout the plateau region. The zero-at-length-1 property from D5 is preserved (both formulas yield 0 when n=1).
- **Alternatives considered**:
  - Original scaling (n-1)/(rows*cols-1) (D5). Rejected: too conservative; BFS signal was still heavily suppressed at the lengths where it matters most.
  - Threshold activation (enable BFS at length K, off before). Rejected: hard cutoff introduces a discontinuity in the observation space that could destabilize learning.
  - No scaling (always full strength). Rejected: BFS is near-uniform at short lengths and adds noise; some suppression at low lengths is desirable.
- **Sources**: run 0317_22 plateau analysis, design-packet.md open question 3

## 8. Tradeoffs

### Benefits

- Body gradient + separate channels empirically eliminated zigzag and improved performance 80%.
- BFS channel directly addresses topology blindness without requiring the CNN to learn graph algorithms.
- Hybrid CNN+MLP gives each modality a natural processing path.
- Numba JIT keeps BFS overhead manageable.

### Costs

- BFS computation reduces SPS from ~8000 to ~1750 (4.5x slowdown) on 32x32 grids.
- Body gradient tail value was vanishing toward zero as snake grows (1/80=0.0125 at length 80). Mitigated by linear normalization to [0.1, 1.0] (D10); usable gradient range is compressed to 0.9 instead of the original ~1.0, but every segment retains a unique value at any body length.
- 4-channel observation + MaxPool removal dramatically increases the flatten dimension, changing learning dynamics.
- Distance shaping introduces staircase movement bias.

### Risks

- BFS channel is uninformative at snake length 1 (all cells reachable); the scaling mitigation is untested in a full training run.
- The first BFS training run (0317_0015) failed catastrophically due to premature policy collapse; architecture changes (MaxPool removal, 1x1 bottleneck) were confounded with the observation change.
- Potential-based shaping (the recommended reward fix) has not been implemented or tested.

### Operational complexity

- Numba JIT adds a compilation warmup on first call and a cache dependency.
- The hybrid observation requires dict-based obs handling throughout the training pipeline (memory, PPO update, agent).

## 9. Edge Cases

- **Snake length 1**: BFS channel is nearly uniform (all cells reachable). Scaling factor min(1.0, (n-1)/50) = 0, so channel is zeroed out. The CNN must learn from channels 0-2 only during early game. Body gradient normalization gives the single head cell 0.1 + 0.9 * 0/max(0,1) = 0.1 + 0.9 * 0 = 0.1... but wait, at n=1 i=0: 0.1 + 0.9 * (1-1-0)/max(0,1) = 0.1 + 0 = 0.1. However the head channel (ch1) is 1.0, so the head is still clearly marked.
- **Snake fills entire grid**: Game terminates with a win. BFS channel is irrelevant. Body gradient tail = 0.1, clearly distinct from empty space (0.0). All segments uniformly spaced in [0.1, 1.0].
- **Tail vanishing under ReLU**: Resolved by D10. Linear normalization guarantees tail = 0.1 at any body length. At length 80+, tail is 0.1 instead of 0.0125, and every segment has a unique value with spacing 0.9/79.
- **MaxPool destroys 1-cell resolution**: MaxPool(2,2) merges adjacent cells, losing the distinction between "1 cell from body" and "2 cells from body." The postmortem flagged this but MaxPool was retained in the working architecture for its regularization and dimension-reduction benefits.
- **BFS + architecture regression**: The first BFS run changed MaxPool removal, 1x1 bottleneck, and BFS simultaneously. The failure could not be attributed to any single change.

## 10. Open Questions

1. ~~Should the body gradient have a floor value (e.g., 0.1 instead of 1/n) to prevent tail vanishing under ReLU?~~ Resolved: linear normalization to [0.1, 1.0] implemented (D10). A naive clamp was tried first but replaced with proper normalization that preserves unique values for all segments. Applied mid-run 0317_22 at update 9000 to break plateau at snake size ~95.
2. Is potential-based shaping (gamma * phi(s') - phi(s)) stable in practice for this environment, or does it require tuning?
3. ~~What is the right BFS scaling strategy: the current (n-1)/(rows*cols-1) multiplicative factor, or a threshold-based activation (enable BFS channel only after snake reaches length K)?~~ Resolved: linear ramp min(1.0, (n-1)/50), full strength by length 51 (D11). Applied mid-run 0317_22 at update 9000.
4. Should the auxiliary open-space reward (+0.001 * reachable_cells / total_cells) be combined with the BFS observation channel, or does it double-count the same information?
5. Is the 1x1 bottleneck (128 to 4 channels) too aggressive for early learning, or was the MaxPool removal the actual cause of the 0317_0015 regression?
6. Why was the hybrid CNN+MLP split chosen over encoding food position as a 5th grid channel? No explicit rationale was recorded.

## 11. Assumptions

- The food offset vector (dx/cols, dy/rows) is more useful as a direct MLP input than as an additional grid channel. No A/B test has been run.
- Orthogonal initialization with std=sqrt(2) is appropriate for this architecture. Standard practice from CleanRL/PPO literature, not explicitly validated.
- The BFS scaling factor was introduced to address the chicken-and-egg problem from 0317_0015, but no transcript explicitly records this reasoning.
- The 4-layer CNN depth (channels 4->16->32->64->128) was chosen to increase capacity for the richer 4-channel input. The 3-channel run used 2 layers; the jump to 4 layers was part of the body-gradient hypothesis but the layer count was not explicitly justified.

## 12. Recommended Next Steps

### Immediate

1. Run the BFS channel with the restored MaxPool architecture (4-channel + MaxPool) to isolate whether the observation or the architecture caused the 0317_0015 regression. (Recommended in postmortem 0317_0015_OP.md experiment 2.)
2. Implement potential-based distance shaping and test against current delta-Manhattan.

### Validation

3. ~~A/B test body gradient with floor (min=0.1) vs current (min=1/n) to quantify tail-vanishing impact.~~ Implemented: linear normalization to [0.1, 1.0] applied mid-run 0317_22 (D10). A/B isolation not performed; change was applied alongside D11.
4. ~~Ablate the BFS scaling factor: compare current scaling vs unscaled vs threshold activation.~~ Implemented: linear ramp min(1.0, (n-1)/50) applied mid-run 0317_22 (D11). Not a clean ablation; change was applied alongside D10.
5. Measure whether the MLP food-offset branch actually helps: compare hybrid vs CNN-only with food as a 5th grid channel.

### Documentation

6. Record explicit rationale for the hybrid CNN+MLP split (currently an assumption).
7. Record explicit rationale for the 4-layer CNN depth choice.

## 13. Handoff Summary

Addressed to a follow-up documentation model:

**Already decided (do not change)**:
- 4-channel grid layout: body gradient (ch0), head binary (ch1), food binary (ch2), BFS reachability (ch3).
- Body gradient formula: 0.1 + 0.9 * (n - 1 - i) / max(n - 1, 1), head=1.0, tail=0.1, uniform spacing (D10).
- BFS uses forward-looking body clearance (not strict passability).
- BFS encoding: inverted distance, head=1.0, unreachable=0. Scaled by min(1.0, (n-1)/50) (D11).
- Hybrid architecture: CNN for grid, MLP for food offset, fused before heads.
- Numba JIT for BFS.

**Still needs clarification** (see Open Questions):
- Hybrid vs pure-CNN rationale.
- Optimal reward formulation (potential-based vs delta-Manhattan vs no shaping).

**Safe to expand into final docs**:
- Sections 4 (Context), 6 (Proposed Design), 7 (Decision Log), 8 (Tradeoffs) have strong source backing.
- Decision Log entries D1-D5 and D7-D9 are well-sourced; D6 is an assumption and should be flagged.

**Requires caution**:
- Section 12 (Next Steps) reflects postmortem recommendations that may have been superseded by newer runs not captured here.

## Missing Information I Need

1. **Hybrid architecture rationale**: No transcript or doc records why CNN+MLP was chosen over a pure CNN with food as a grid channel. An explicit A/B test result or design discussion would fill this gap.
2. **BFS scaling factor origin**: The (n-1)/(rows*cols-1) multiplier appears in code but was never discussed in any transcript. Was it introduced to fix the 0317_0015 failure, or was it part of the original design?
3. **4-layer CNN depth justification**: The jump from 2 to 4 conv layers coincided with the body-gradient hypothesis. Was the depth increase motivated by the richer observation, or by a separate receptive-field analysis?
4. **Results from any run after 0317_0015**: The latest postmortem covers a failed BFS run. If subsequent runs exist (e.g., 0317_01_bfs-reachability visible in the IDE), their outcomes would fill the "does BFS actually help?" question.
5. **Reward ablation data**: Multiple reward alternatives were proposed (potential-based, survival bonus, auxiliary open-space) but none have recorded experimental results. Any empirical comparison would strengthen the Decision Log for D7/D8.
