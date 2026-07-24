# Finding 010 — Dreams vs data: F-PLAIN slower than F-STORY

**Status:** archived (UI thinking + markdown export)  
**Date:** 18 July 2026  
**Motto:** nullius in verba — do not believe the dream.

## Operator dream (pre-data)

> Sprookjes raken rijke verbeelding; een héél voorspelbare fabel (schildpad & haas) zou “easy peasy” moeten zijn → **sneller** dan Doornroosje.

## Data (UI thinking, same model / thinking ON / system prompt)

| Condition | Design intent | UI thinking (s) | Think w | Final w |
| --- | --- | ---: | ---: | ---: |
| F-STORY (Doornroosje) | rich fairy tale, child diction | **33.18** | 1065 | 613 |
| F-PLAIN (schildpad & haas) | predictable Aesop, almost no magic | **37.8** | 1210 | 590 |
| F-NARR (Guldensporenslag) | familiar history chronicle | ~37.7 | 1629 | 85 |

Export: [`datasets/runs/2026-07-18/familiar-schildpad-haas-story.md`](../datasets/runs/2026-07-18/familiar-schildpad-haas-story.md)

**F-PLAIN was not faster.** It landed with familiar-history, **above** the fairy-tale floor.

## What we conclude (scientific blast)

1. **Gut ≠ instrument.** The “predictable = cheaper” story failed this cell.
2. Possible (non-proven) readings — human, non-neutral:
   - Length confound (1600 vs 1939) is the wrong direction for explaining *slower* F-PLAIN.
   - “Predictable” may still trigger lesson/moral audit under premises+themes.
   - Fairy-tale schema may be so overlearned that deliberation collapses sooner.
3. None of those readings are **proven**. Only the rank is.

## What we refuse to conclude

- That fairy tales are “deeper” for the model’s soul.
- That children’s cognition is measured here (we measured **model UI thinking time**).
- That arm **C** (thinking OFF) can prove or disprove resonance / consciousness — that stays **subjective** and is **out of scope** as a proof arm. Optional engineering dessert only; not ontology.

## Scope lock remains

Qwen3.6-35B-A3B uncensored · thinking ON · premises system prompt · n=1.

Dessert **in progress:** **no system prompt** (arm B) — see [`datasets/prompts/run-arm-b-nosystem.md`](../datasets/prompts/run-arm-b-nosystem.md). Arm C = not a path to certainty.
