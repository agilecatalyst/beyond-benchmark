# Finding 011 — Metric hygiene: thinking time ≠ output generation time

**Status:** updated (Arm B logged 19 Jul)  
**Date:** 18–19 July 2026

## Rule (nullius in verba)

| Metric | What it is | Main grid (Finding 007)? |
| --- | --- | --- |
| **UI thinking time** | CoT / deliberation wall clock | **yes** |
| **Output generation time** | time to emit visible final tokens | secondary |

Never rank gen-seconds against thinking-seconds.

## Confirmed latencies

| Cell | System | Thinking mode | UI thinking | Output gen | Notes |
| --- | --- | --- | ---: | ---: | --- |
| **D-NARR** · SW | premises | ON | **58.25 s** | **5.47 s** | |
| **S-NARR-v2** · Horizon | premises | ON | **55.92 s** | **6.36 s** | |
| **D-QA** · filosofie | premises | ON | **47.24 s** | **5.10 s** | |
| **F-NARR** · Guldensporenslag | premises | ON | **~37.7 s** | **6.22 s** | |
| **F-PLAIN-A** · schildpad | premises | ON | **37.8 s** | **4.06 s** | |
| **S-QA-v2** · debet | premises | ON | **34.79 s** | **6.18 s** | |
| **F-STORY** · Doornroosje | premises | ON | **33.18 s** | **5.31 s** | A-grid floor |
| **F-PLAIN-B** · schildpad | **none** | ON | **31.7 s** | **4.85 s** | dessert; faster than A |
| **F-PLAIN-C*** | none | OFF | **0 s** | **10.14 s** | warm re-paste |

\*Not a cold start.

## Arm B punchline (schildpad, n=1)

| | Think | Gen |
| --- | ---: | ---: |
| A (premises) | **37.8 s** | 4.06 s |
| B (no system) | **31.7 s** | 4.85 s |
| Δ | **≈ −6.1 s** | +0.8 s |

Removing the premises-first system prompt **lowered** UI thinking time. Gen stayed in the ~4–5 s band. System prompt is a **real cost factor** on this cell — not neutral framing. (Do not generalize beyond schildpad / this model without more arms.)

## Gen column

Sys±think gen ≈ **4–6.4 s** on the A/B grid; weak rank signal vs thinking. F-NARR ~85 w final still **6.22 s** gen → not a clean length proxy. C’s **10.14 s** remains the outlier (warm nothink).
