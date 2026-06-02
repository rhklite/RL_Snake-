# v2 — Body gradient + per-element channels

- **Date:** 2026-03-16
- **Run(s):** `runs/archive/0316_01_v2/` (config `bodyGradient`)
- **Grid:** 32×32 · **Obs:** 3-channel (body gradient, head, food) + food-offset MLP
- **Outcome:** partial (best result of the food-seeking era)
- **Decisions introduced:** [D1](../design-packet.md) (body gradient), [D2](../design-packet.md) (per-element channels), [D6](../design-packet.md) (hybrid CNN+MLP)

## Hypothesis

Encoding body segments as a continuous gradient from head (1.0) down to tail (1/n), plus separate
binary head and food channels, gives the CNN enough signal to infer travel direction and segment
age — eliminating zigzag. Deepening to 4 conv layers (4→16→32→64→128) adds capacity for the richer
input. Same reward as v1.

## Changes vs. prior

- Replaced single int8 grid with **3-channel** obs: body gradient (head=1.0, tail=1/n), head binary,
  food binary (D1, D2).
- CNN depth **2 → 4 layers** (commit `2a6608a`).
- Kept hybrid CNN + food-offset MLP (D6) and the v1 reward.

## Outcome & metrics

| Metric | Value |
|--------|-------|
| Best return | **49.94** (update 5319) — **+80% vs v1** |
| Best snake length | **83.1** — +80% vs v1 |
| Final return | 40.91 |
| Death split (final) | 43% wall / 56% body / 0.1% timeout |
| Grid fill at death | ~7% (len ~72 / 1024) |
| Entropy | stable ~0.4 |
| SPS | 10,300 → 8,000 |

## Learnings

- **Zigzag eliminated**; best return and length both up ~80%. This is the strongest food-seeking
  result in the whole lineage.
- Snake still dies by body collision at only ~7% fill. The body gradient encodes *age*, not
  *connectivity* — the CNN can't tell which regions become enclosed pockets.
- A diagonal **staircase** pattern persists (Manhattan shaping rewards alternating steps). The
  0316 synthesis settled a Gemini/Opus disagreement: zigzag is gone; staircase is now dominant.
- Tail value 1/n drops to 0.0125 at length 80 — near the ReLU/MaxPool noise floor. Foreshadows the
  v4 tail-vanishing fix.

## Kept → v3

Body gradient (D1), per-element channels (D2), 4-layer CNN, hybrid head (D6), reward (D7/D8).

## Discarded

The assumption that an observation-only fix could close the **topology** gap.

## Next

Add a 4th channel encoding spatial **reachability** (BFS flood-fill from the head) so the CNN
doesn't have to learn connectivity from a finite receptive field.

## Sources

- [0316_0830_OP.md](../postmortem/0316_0830_OP.md), [0316_0833_OP.md](../postmortem/0316_0833_OP.md), [0316_0848_SYNTH.md](../postmortem/0316_0848_SYNTH.md)
- [design-packet.md D1/D2](../design-packet.md), commit `2a6608a`
