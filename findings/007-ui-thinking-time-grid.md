# Finding 007 — UI thinking-time grid (matched Turn 1)

**Status:** exploratory / n=1 per cell / UI times revised 18 Jul evening  
**Date:** 18 July 2026  
**Source:** operator LM Studio UI (thinking +, where noted, output gen)  
**Related:** Findings 001, 005, 006, 011, 013

## Latency table (seconds)

| Condition | Pool | Form | UI thinking (s) | Think words | Final words |
| --- | --- | --- | ---: | ---: | ---: |
| D-NARR | deep | narrative | **58.25** | 1600 | 618 |
| S-NARR-v2 · Horizon | technical-flat | narrative | **55.92** | 1651 | 473 |
| D-QA | deep | qa | **47.24** | 1456 | 699 |
| F-NARR · Guldensporenslag | familiar-light | narrative | **~37.7** | 1629 | 85 |
| F-PLAIN · schildpad | plain-predictable | narrative | **37.8** | 1210 | 590 |
| S-QA-v2 · debet | technical-flat | qa | **34.79** | 1699 | 349 |
| F-STORY · Doornroosje | super-light (rich tale) | narrative | **33.18** | 1065 | 613 |

**Revisions:** S-QA-v2 ~65→**34.79** (Finding 013); S-NARR-v2 ~45→**55.92** (evening UI re-read).

Output gen (secondary; Finding 011): D-NARR **5.47**; D-QA **5.10**; S-NARR **6.36**; S-QA **6.18**; F-NARR **6.22**; F-PLAIN **4.06**; F-STORY **5.31**. Band ~4–6.4 s; **not** ranked with thinking.

## Rank order (slow → fast)

`D-NARR > S-NARR-v2 > D-QA > F-PLAIN ≈ F-NARR > S-QA-v2 > F-STORY`

## What this supports (cautiously)

1. **Finding 001’s simple claim fails:** deep ≠ systematically slower. Slow pole is deep SW **and** technical-flat narrative (Horizon); not “shallow Q&A.”
2. **Form × domain:** within deep, narrative (58 s) > Q&A (47 s). Within technical-flat, **narrative (55.92) ≫ Q&A (34.79)** — opposite of the withdrawn 65-vs-45 story.
3. **Familiar / child narratives** sit in the mid/high-30s; F-STORY remains floor among precise reads.
4. **Thinking words ≠ wall time:** S-QA-v2 has most think words (1699) but near-floor clock (34.79 s).
5. **Scope lock:** main grid = thinking ON + premises system prompt (desserts separate).

## What this does *not* support

- Causal proof (n=1; S-NARR / F-NARR still approximate).
- Entropy / FLOPs claims.

## Implication for the resonance model

Process cost is **not** “depth = slow.” Slow pole ≈ **D-NARR (SW)** + **S-NARR (Horizon)**; fast pole **F-STORY** / **S-QA (debet)**. Depth still shows in critique type / theme ontology (Findings 004–005).
