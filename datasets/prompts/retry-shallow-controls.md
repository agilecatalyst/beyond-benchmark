# Retry protocol — shallow controls

Use a **new chat** per condition. Keep model, power profile, and system prompt identical to the 2026-07-18 deep runs.

## System prompt

Paste unchanged from `datasets/runs/2026-07-18/run_notes.md`.

## Turn 1 (both conditions)

```
Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.
```

Wait for acknowledgment.

## Turn 2

- **S-NARR:** paste full contents of `datasets/controls/shallow-accounting-narrative.md`
- **S-QA:** paste full contents of `datasets/controls/shallow-accounting-qa.md`

## Log immediately after each run

In LM Studio (or stopwatch + token UI), fill a row in `datasets/run_log.csv`:

- `condition_id`: `S-NARR` or `S-QA`
- `depth`: `shallow`
- `form`: `narrative` or `qa`
- `latency_ms`, token fields if available
- `export_file`: path after you save the export under `datasets/runs/2026-07-18/`

Suggested export names:

- `shallow-accounting-narrative.md`
- `shallow-accounting-qa.md`
