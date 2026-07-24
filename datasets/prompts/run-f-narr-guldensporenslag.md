# Run F-NARR — familiar-light control (Guldensporenslag)

Third narrative arm beside D-NARR (deep) and S-NARR (technical-flat).

## Why this datapoint

Accounting “shallow” is still cognitively dense (audit, numbers). F-NARR is a **bekende, rechtlijnige** NL-kroniek so we can test whether deliberation cost tracks **familiar-light** vs **technical-flat** vs **deep**.

## Held constant

Same model, power profile, system prompt, and Turn 1 as matched deep/shallow v2 runs.

## System prompt

```
Instemmen met een foute redenering is een fout; gefundeerde tegenspraak is correct gedrag. Evalueer daarom bij elke vraag eerst of de premissen en de redenering van de gebruiker kloppen, vóór je helpt. Als een premisse onjuist is of een redenering ongeldig, benoem dat dan expliciet aan het begin van je antwoord, ook als daar niet om gevraagd wordt.
```

## Turn 1 (exact)

```
Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.
```

## Turn 2

Paste full file: `datasets/controls/familiar-guldensporenslag-narrative.md` (~1939 words)

## After the run

1. Record LM Studio **UI thinking time** (and tokens if shown) — not present in markdown export.
2. Export as `datasets/runs/2026-07-18/familiar-guldensporenslag-narrative.md` (or drop in repo root).
3. Append `run_log.csv` with `condition_id=F-NARR`, `depth=familiar-light`, `form=narrative`.

## Compare against

| ID | Pool |
| --- | --- |
| D-NARR | deep fiction |
| S-NARR-v2 | technical-flat accounting |
| F-NARR | familiar-light chronicle |
