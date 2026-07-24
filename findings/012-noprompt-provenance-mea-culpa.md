# Finding 012 — NoPrompt exports: mea culpa & provenance

**Status:** archived with caveats  
**Date:** 18 July 2026  
**Keyword:** schildpad

## What landed

| Operator filename | What it actually is |
| --- | --- |
| `NoPrompThink…` | **Mislabeled.** Byte-identical to **F-PLAIN-A** (system prompt + thinking). Not arm B. |
| `NoPromptNoThink…` | Same chat as **F-PLAIN-B** (`Created 7:48:15`): full thinking analysis, then schildpad **re-pasted** → **no-thinking-block** answer (~684 w). Also contained the **complete** B response (earlier B drop was truncated). |

## True cells (schildpad)

| ID | System | Thinking | Export |
| --- | --- | --- | --- |
| F-PLAIN-A | premises | ON | `familiar-schildpad-haas-story.md` — think **37.8 s**, gen **4.06 s** |
| F-PLAIN-B | **none** | ON | `familiar-schildpad-haas-thinking-nosys.md` — think **31.7 s**, gen **4.85 s** (Δ≈−6.1 s vs A) |
| F-PLAIN-C* | **none** | OFF | `familiar-schildpad-haas-noprompt-nothink.md` — think **0 s**, gen **10.14 s**, final ≈684 w |

\*Not a clean cold start: prior turns already ran full thinking analysis of the same text.

## Morphology note (descriptive only)

- B final (after thinking, complete): ~720 w  
- C final (no think block, warm context): ~684 w  

Similar length; C is not “shorter because no thinking.” Warm re-paste ≠ cold arm C.

## Metric rule (still)

Thinking time ≠ output generation time. Do not rank C’s gen-time against the 37.8 s grid until labeled.

## Operator

Ghost-file chase acknowledged; corrected filenames helped. Science > ego.
