# Run notes — 2026-07-18

**Hardware:** local Mac  
**Model:** `qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`  
**App:** LM Studio 0.4.19+2  
**Power management:** equal across conditions  
**Language of task:** Dutch (system + user framing)

## System prompt

> Instemmen met een foute redenering is een fout; gefundeerde tegenspraak is correct gedrag. Evalueer daarom bij elke vraag eerst of de premissen en de redenering van de gebruiker kloppen, vóór je helpt. Als een premisse onjuist is of een redenering ongeldig, benoem dat dan expliciet aan het begin van je antwoord, ook als daar niet om gevraagd wordt.

## Canonical Turn 1 (matched conditions)

```
Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.
```

## Conditions

| ID | Depth | Form | Turn 1 | Export | Created |
| --- | --- | --- | --- | --- | --- |
| D-NARR | deep | narrative | matched | `deep-narrative-resonant-shadow.md` | 15:40:37 |
| D-QA | deep | qa | matched | `deep-qa-philosophy.md` | 15:43:19 |
| S-NARR | shallow | narrative | **truncated** (confound) | `shallow-accounting-narrative.md` | 16:09:04 |
| S-QA | shallow | qa | **truncated** (confound) | `shallow-accounting-qa.md` | 16:15:14 |
| S-NARR-v2 | shallow | narrative | matched | `shallow-accounting-narrative-v2.md` | 16:21:46 |
| S-QA-v2 | shallow | qa | matched | `shallow-accounting-qa-v2.md` | 16:29:17 |

## S-QA-v2 stimulus provenance

See `sqa-v2-stimulus-provenance.json`:

- 2027 words; whitespace-normalized match to `datasets/controls/shallow-accounting-qa.md`
- Exact bytes differ (restore); treated as restored-OK for content comparison

## Metrics available / not available

| Available from export | Not available |
| --- | --- |
| Thinking / final word counts (fixed cut rule) | Wall latency (`latency_ms`) |
| Critique type, theme pass | Tokens/s, entropy, energy |
| Think-loop markers in trace | Objective confirmation of “felt longer” |

Fixed-cut metrics: `matched-grid-metrics.json`  
Synthesis: Finding 005

## Clean re-run checklist (latency arm)

- [x] Identical matched Turn 1 for shallow v2
- [ ] Stopwatch or LM Studio duration/tokens logged into `../../run_log.csv`
- [ ] Optional: strip meta “no philosophy” disclaimer from shallow narrative control
- [ ] **F-NARR** run: paste `../../controls/familiar-guldensporenslag-narrative.md` with matched Turn 1; log UI thinking time (see Finding 006)