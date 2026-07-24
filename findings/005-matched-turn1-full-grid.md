# Finding 005 — Matched Turn 1 full grid (nullius in verba)

**Status:** exploratory / proxies-only  
**Date:** 18 July 2026  
**Related:** Findings 001–004  
**Metrics source:** [`datasets/runs/2026-07-18/matched-grid-metrics.json`](../datasets/runs/2026-07-18/matched-grid-metrics.json)  
**S-QA-v2 provenance:** [`datasets/runs/2026-07-18/sqa-v2-stimulus-provenance.json`](../datasets/runs/2026-07-18/sqa-v2-stimulus-provenance.json)

## Measurement rule (fixed)

For the main evaluation assistant turn (after stimulus paste):

1. **Final** = last block starting with `**Evaluatie…`, `### Evaluatie…`, `Bij het lezen van de tekst`, or `De premissen en redenering…`
2. **Thinking** = everything before that cut
3. **Latency** = *not measured* (absent from LM Studio markdown). `Created:` is chat creation time, not generation duration.

## Matched Turn 1 grid (same system prompt; same “bedenkingen / thema's” request)

| Condition | Stim words | Thinking words | Final words | Think-loop markers | Critique type | Theme pass | Theme ontology (labels) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| D-NARR | 1939 | 1600 | 618 | 1 | falsify | yes | resonance/dualism |
| D-QA | 2037 | 1456 | 699 | 0 | falsify | yes | hermetica/AI, logic-emotion, … |
| S-NARR-v2 | 1939 | 1651 | 473 | 1 | validate+nuance | yes | procedural-flatness, audit, accuracy-as-value |
| S-QA-v2 | 2027 | 1699 | 349 | **9** | validate+nuance | yes | digitalization, audit-control |

### Contrast: unmatched Turn 1 (Finding 003 confound)

| Condition | Thinking | Final | Theme pass |
| --- | ---: | ---: | --- |
| S-NARR-v1 | 1163 | 167 | no |
| S-QA-v1 | 959 | 50 | no |

## S-QA-v2 stimulus check

- Pasted stimulus **whitespace-normalized match** to current control (`shallow-accounting-qa.md`).
- Exact bytes differ (sha1 `82742d83…` vs `c20ceb35…`); word count both **2027**.
- Provenance: restored-OK for content comparison.

## What the grid supports

1. **Matched prompt equalizes theme-pass presence.** Both shallow v2 runs produce a full “Bedenkingen over thema's” section — unlike v1.
2. **Thinking-word volume does not separate deep from shallow** under matched Turn 1. Shallow v2 thinking is in the same band as deep (≈1450–1700); S-QA-v2 is even slightly higher.
3. **Remaining contrast is qualitative:**
   - Deep → **falsify** cross-domain / coherence tensions, then conceptual theme synthesis.
   - Shallow → **validate (+ contextual nuance)**, then themes about procedure, audit culture, digitalization, or flatness-as-style.
4. **Final answers stay shorter on shallow** (≈350–470 vs ≈620–700), even when themes are requested.

## UI thinking-time addendum (operator report, not in markdown)

LM Studio UI thinking times recalled by operator (approx; **not** embedded in the `.md` export):

| Condition | UI thinking time (approx) |
| --- | ---: |
| S-NARR-v2 | ~45 s |
| S-QA-v2 | ~65 s |

Logged as `latency_ms` ≈ 45000 / 65000 in `datasets/run_log.csv` with note `from LM Studio UI`.

**Interesting, limited:** within shallow matched runs, Q&A deliberation ran longer than narrative (~+44%). That aligns directionally with more think-loop markers in S-QA-v2 (9 vs 1). It does **not** yet test Finding 001 (deep vs shallow): D-NARR / D-QA UI times were not recorded.

## What the grid does *not* support (nullius in verba)

- Markdown exports still contain **no** duration fields — UI times are a separate, approximate channel.
- **Thinking words ≠ wall time ≠ FLOPs**, but here words + UI time move together for S-QA-v2 vs S-NARR-v2.
- Finding 001’s “deep content → higher processing cost” remains **untested on latency** until deep UI times exist under the same matched Turn 1.

## Implication for the resonance model

Under a fixed premises+themes task, **depth shows up as critique type and theme ontology**, not as a reliable surplus of deliberation length. Task framing previously masqueraded as a depth/cost effect (Finding 003 → 004).

## Next measurement (if cost claims are desired)

Log for each condition: wall-clock start/stop, prompt/completion/thinking tokens if shown, same machine/power. Until then, treat all latency talk as hypothesis only.
