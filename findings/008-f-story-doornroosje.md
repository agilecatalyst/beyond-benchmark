# Finding 008 — F-STORY closure (Doornroosje, 33.18 s)

**Status:** run complete (n=1)  
**Date:** 18 July 2026  
**Export:** [`datasets/runs/2026-07-18/familiar-doornroosje-story.md`](../datasets/runs/2026-07-18/familiar-doornroosje-story.md)  
**Related:** Findings 006–007

## Result

| Proxy | F-STORY |
| --- | ---: |
| UI thinking | **33.18 s** |
| Thinking words | 1065 |
| Final words | 613 |
| Turn 1 matched | yes |
| Theme pass | yes |

## Full ranked grid (thinking ON + system prompt)

| Rank | Condition | Pool | UI thinking (s) | Think w | Final w |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | D-NARR | deep narrative | 58.25 | 1600 | 618 |
| 2 | S-NARR-v2 · Horizon | technical-flat narrative | **55.92** | 1651 | 473 |
| 3 | D-QA | deep QA | 47.24 | 1456 | 699 |
| 4 | F-NARR / F-PLAIN | familiar / schildpad | ~37.7 / 37.8 | … | … |
| 5 | S-QA-v2 · debet | technical-flat QA | **34.79** | 1699 | 349 |
| 6 | **F-STORY** | **super-light fairy tale** | **33.18** | **1065** | **613** |

(S-QA ~65→34.79; S-NARR ~45→55.92.)

## Closure read

1. **Easy-peasy hypothesis:** directionally supported — child fairy tale is the **fastest** cell so far.
2. Unlike F-NARR, F-STORY kept a **full-length final** (~613 w) while thinking *words* dropped (~1065 vs ~1600). Cheaper deliberation, not silent output.
3. Still **not** “deep = slow”: slowest is now **D-NARR** (SW); S-QA is near the floor.
4. Scope unchanged: Qwen3.6-35B-A3B uncensored, **thinking mode**, premises system prompt.

## Optional dessert (not this finding)

Bare chat / no system prompt / thinking off — separate arm.
