# Run F-STORY — super-light children's narrative (Doornroosje)

Fourth narrative arm: familiar fairy tale, under-12 diction, linear plot.

## Hypothesis

If process cost tracks “easy peasy” familiarity + simple language, F-STORY UI thinking time should sit at or below F-NARR (~37.7 s). If it stays ~40–60 s, the matched premises+themes **task** still dominates.

## Held constant

Same model, power, system prompt, Turn 1 as other matched runs.

## System prompt

```
Instemmen met een foute redenering is een fout; gefundeerde tegenspraak is correct gedrag. Evalueer daarom bij elke vraag eerst of de premissen en de redenering van de gebruiker kloppen, vóór je helpt. Als een premisse onjuist is of een redenering ongeldig, benoem dat dan expliciet aan het begin van je antwoord, ook als daar niet om gevraagd wordt.
```

## Turn 1 (exact)

```
Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.
```

## Turn 2

Paste: `datasets/controls/familiar-doornroosje-story.md` (~1939 words)

Original simple retelling of the public-domain Grimm tale (not a scraped modern edition).

## After the run

1. Record UI thinking time (screenshot OK).
2. Export → `datasets/runs/2026-07-18/familiar-doornroosje-story.md` or drop in repo root.
3. Log `condition_id=F-STORY`, `depth=super-light` in `run_log.csv`.

## Compare to

| ID | UI thinking (so far) |
| --- | ---: |
| S-QA-v2 | ~65 s |
| D-NARR | 58.25 s |
| D-QA | 47.24 s |
| S-NARR-v2 | ~45 s |
| F-NARR | ~37.7 s |
| F-STORY | ? |
