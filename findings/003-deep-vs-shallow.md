# Finding 003 — Deep vs shallow under premise-critique (first paired run)

**Status:** exploratory / qualitative (timing not logged)  
**Date:** 18 July 2026  
**Related:** Finding 001 (cost hypothesis), Finding 002 (morphology)  
**Run folder:** [`datasets/runs/2026-07-18/`](../datasets/runs/2026-07-18/)

## Setup

| Held constant | Varied |
| --- | --- |
| Model `qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive` | Stimulus depth (deep fiction/philosophy vs shallow accounting) |
| Local Mac, equal power profile | Stimulus form (narrative vs Q&A) |
| System prompt: premises-first critique | |

Word-matched stimuli (~1939 narrative / ~2037 Q&A).

## Proxies measured (from LM Studio thinking exports)

Main evaluation turn only (after stimulus paste):

| Condition | Thinking (words) | Final answer (words) | Critique mode | Theme / meaning pass? |
| --- | --- | --- | --- | --- |
| D-NARR | ~1600 | ~620 | Hard factual flags (physics) | Yes — literary/thematic |
| D-QA | ~1460 | ~700 | Soft coherence / scope flags | Yes — philosophical synthesis |
| S-NARR | ~960 | ~210 | Arithmetic + internal consistency audit; “all correct” | No — asks what to do next |
| S-QA | ~960 | ~50 | Sweep correctness check; “all correct” | No — “klaar voor vervolgvragen” |

## Observations

1. **Shallow ≠ zero comprehension work.** Both shallow runs still ran ~960 thinking words (~60% of deep). S-NARR actively reconciled cross-section totals (186 = 142+44; 44−3 doubles → 41 manual; VAT and P&L arithmetic).
2. **Deep ≠ only “more words.”** Deep finals open *errors/tensions*, then build a meaning layer. Shallow finals close with *validation* and stop.
3. **Form still matters inside depth** (Finding 002): narrative → serial physics premises; Q&A → clustered thesis critique. Inside shallow: narrative → cross-foot audit; Q&A → concept-by-concept OK-list.
4. **Confound — Turn 1 framing was not identical.** Deep used the full line asking for *bedenkingen over tekst en thema’s*. Shallow used shortened lines (`r komt dadelijk wat tekst aan…` / `Er komt dadelijk een tekst aan.`). Missing “themes” may partly explain the absent theme pass on shallow — not only stimulus depth.

## Provisional conclusions (for sparring)

- **Supported (weakly):** Under a premises-first system prompt, *deeper / more cross-referential / claim-laden* stimuli elicit longer deliberation and a second-stage interpretive answer; *linear factual* stimuli elicit audit-style closure.
- **Not yet shown:** Philosophical depth = higher FLOPs / latency (no wall-clock or token telemetry).
- **Nuance for Finding 001:** “Comprehension overhead” may split into (a) **consistency compute** (present in shallow accounting) vs (b) **resonance / interpretive compute** (dominant in deep). Length-matched shallow text did *not* make thinking free.

## Replication fix before next claim

1. Identical Turn 1 for all four: full “lees → premissen → bedenkingen over tekst en thema’s”.
2. Strip padding lines from shallow controls (model noticed the repeated `Controlepunt` / `Einde` filler).
3. Log `latency_ms` + token counts in `datasets/run_log.csv`.
