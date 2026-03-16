---
name: log-hypothesis
description: Captures the experiment hypothesis, distills a slug, launches training with W&B tracking, and opens the dashboard. Use when the user mentions logging a hypothesis, starting training, "train with hypothesis", "what am I testing", "start a run", or wants to launch a training experiment.
---

# Log Hypothesis and Train

End-to-end flow: capture hypothesis, launch training with W&B, open the dashboard.

## Steps

1. **Ask** the user to describe their experiment intent in free text (one or two sentences).
2. **Distill a slug**: extract 2-4 key concepts, lowercase, hyphen-separated.
   - Strip punctuation, collapse spaces to hyphens.
   - Examples: `test-entropy-collapse`, `higher-dist-alpha`, `deeper-cnn`
3. **Confirm** with the user: present the slug and any config overrides. Ask if the slug should be adjusted.
4. **Launch training** with W&B tracking via `scripts/train_wandb.sh`, passing the slug and any config overrides:
   ```
   bash scripts/train_wandb.sh --training hybrid_64x64 \
       "training.hypothesis_slug=<slug>" \
       <any other overrides>
   ```
   The script:
   - Runs training detached (`nohup`, `disown`) with `--track` for W&B.
   - Waits for the run directory + log file, then opens the W&B dashboard URL.
5. **Report** to the user:
   - PID and how to stop (`kill <PID>`)
   - Run directory path (e.g. `runs/0315_04_fix-zigzag-hack/`)
   - W&B dashboard URL
   - Remind: "Fill in `hypothesis.md` in the run directory with what you're testing and the expected outcome."

## hypothesis.md scaffold

`train.py` writes this file to `runs/<experiment_name>/hypothesis.md` with the slug pre-filled.
The user fills in all other sections manually.

```markdown
# Hypothesis

**Slug**: <slug>

## What I'm testing
<!-- fill in before the run -->

## Expected outcome
<!-- fill in before the run -->

## Key config changes
<!-- e.g. ppo.ent_coef: 0.01 → 0.03 -->

## Result (fill in after run)
<!-- outcome, actual metrics, next steps -->
```

## Slug rules

- Lowercase, hyphens only (no underscores, spaces, or special chars)
- 2-4 words
- No dates or version numbers (already in the dir name timestamp)
