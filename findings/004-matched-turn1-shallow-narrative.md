# Finding 004 — Matched Turn 1: shallow narrative re-run

**Status:** exploratory (narrative arm only; S-QA-v2 pending)  
**Date:** 18 July 2026  
**Fixes confound from Finding 003:** identical Turn 1 including “bedenkingen over de tekst en de thema's.”

## Comparison (main evaluation turn)

| | Turn 1 matched? | Thinking words | Final words | Premises stance | Theme pass? |
| --- | --- | ---: | ---: | --- | --- |
| D-NARR | yes | ~1600 | ~620 | Flags factual errors (physics) | Yes — literary / Force / dualism |
| S-NARR-v1 | **no** (truncated) | ~960–1160 | ~170 | All correct + asks “what next?” | **No** |
| S-NARR-v2 | **yes** | ~1650 | ~470 | All correct within scope | **Yes** — themes *about* procedural flatness |

Export: `datasets/runs/2026-07-18/shallow-accounting-narrative-v2.md`

## What changed when the prompt matched

Finding 003’s sharp “shallow has no meaning layer” largely **dissolved**. With the same theme request:

1. Thinking volume for S-NARR-v2 ≈ D-NARR (not ~60%).
2. A full **Bedenkingen over thema's** section appears (style-as-content, automation vs care, accuracy as moral value, limitation as strength).
3. The remaining contrast is **qualitative**, not presence/absence:
   - Deep: *error-seeking across domains* (Hz vs breath) → metaphor of resonance.
   - Shallow: *consistency audit* → thematizes the *absence of interpretation* itself.

## Implications for Finding 001

- Task framing was a large fraction of the earlier “overhead” gap.
- Under matched “evaluate + themes,” length-matched shallow text still incurs heavy deliberation — so **depth ≠ the only driver of compute**.
- A cleaner depth signal may be: **critique type** (falsify cross-domain claims vs confirm internal arithmetic) and **theme ontology** (imposed metaphor stack vs meta-commentary on flat procedure), not raw thinking length.

## Caveat

Section XIV of the accounting control *announces* that philosophy is omitted. That may invite thematizing shallowness. A stricter shallow control would omit that meta-disclaimer.

## Still needed

- S-QA-v2 with the same matched Turn 1  
- Wall-clock / token telemetry for cost claims  
