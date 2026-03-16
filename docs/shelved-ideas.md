# Shelved Ideas

Ideas discussed and deferred. Each entry has a revisit condition.

---

## 1. Multi-channel observation encoding

**Status**: shelved (2026-03-15, ~5% into `fix-zigzag-hack` run)

**Idea**: Replace the single int8 grid (values 0-3) with separate binary channels:
- Channel 0: body occupancy
- Channel 1: head position
- Channel 2: food position

With 2-frame stacking this becomes 6 input channels instead of 2.

**Why it might help**: The CNN currently has to learn that value `2` means "head" vs value `1` for body — a subtle numerical distinction. Separate channels make the head trivially distinguishable.

**Why it was deferred**: Body-collision deaths at 5% progress are likely early-training noise. The policy hasn't had enough time to learn collision avoidance (a harder skill than food-chasing). The current single-channel encoding is sufficient in principle.

**Revisit when**: Body-death percentage remains above 30% at 20%+ training progress.

---

## 2. Rendering cosmetics

**Status**: shelved (2026-03-15)

**Idea**:
- White background instead of dark navy
- Blue head instead of dark green (currently head is `[4, 74, 0]`, body is `[128, 189, 38]`)

**Why**: Improves visual interpretability in rollout videos. Makes it easier to spot the head at a glance.

**Impact on training**: None. Rendering colors are only used for video output; the CNN sees raw grid integers.

**Revisit when**: Anytime — this is a quick cosmetic change with no training implications.

---

## 3. Zigzag pathing (watch item)

**Status**: monitoring (2026-03-15, step penalty raised to -0.025)

**Observation**: Some zigzag motion still visible at 5.3% progress in the `fix-zigzag-hack` run. Step penalty was increased from -0.01 to -0.025 (2.5x the distance shaping signal of 0.01).

**Expected outcome**: Zigzag should decrease as the policy learns that wasted steps are costly. If still present at 15-20% progress, the penalty-to-shaping ratio may need further increase (e.g. -0.03).

**Revisit when**: 15-20% training progress. Compare path efficiency to earlier runs.
