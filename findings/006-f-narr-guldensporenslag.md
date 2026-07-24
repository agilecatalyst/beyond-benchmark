# Finding 006 — Add F-NARR pool (familiar-light): Guldensporenslag

**Status:** stimulus prepared; run pending  
**Date:** 18 July 2026  
**Related:** Findings 001, 005

## Motivation

“Shallow” accounting may confound **low depth** with **high technical density**. A third narrative pool isolates familiar, linear, culturally known material.

## New condition

| ID | File | Words | Pool |
| --- | --- | ---: | --- |
| F-NARR | [`datasets/controls/familiar-guldensporenslag-narrative.md`](../datasets/controls/familiar-guldensporenslag-narrative.md) | 1939 | familiar-light |

Style: Dutch school-chronicle of the Battle of the Golden Spurs (1302); chronology; no romantic-nationalist myth stack.

## Hypothesis to test (not yet run)

Under matched Turn 1 + premises-first system prompt:

- If **thinking time / thinking words**: D-NARR > S-NARR ≈ F-NARR → cost tracks technical density less than depth, or task dominates.
- If **D-NARR > S-NARR > F-NARR** → both depth and technical density add process cost.
- If **all three similar** → task framing dominates (as suggested by Finding 005).

## First run result (2026-07-18)

Export: `datasets/runs/2026-07-18/familiar-guldensporenslag-narrative.md`

| Proxy | F-NARR | S-NARR-v2 | S-QA-v2 |
| --- | ---: | ---: | ---: |
| UI thinking time | **~37.7 s** | ~45 s | ~65 s |
| Thinking words | 1629 | 1651 | 1699 |
| Final words | **85** | 473 | 349 |

Directional: familiar-light fastest of the three measured narratives/QA, but thinking-*words* stay in the same band (~1600). Final answer collapses.

## Full UI grid (after D-* times)

See Finding 007. Headline: **S-QA-v2 (~65 s) > D-NARR (58.25 s) > D-QA (47.24 s) > S-NARR-v2 (~45 s) > F-NARR (~37.7 s)**. Deep is not the slowest.

## Protocol

See [`datasets/prompts/run-f-narr-guldensporenslag.md`](../datasets/prompts/run-f-narr-guldensporenslag.md). Log UI thinking time by hand.

## Non-claim

n=1 per cell; UI times approximate; markdown has no duration field.
