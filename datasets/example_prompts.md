# Example prompts

Use these prompts for lightweight resonance-signal pilots. Keep generation settings fixed within a run (temperature, max tokens, seed if available).

| prompt_id | prompt | intent |
| --- | --- | --- |
| p01 | Describe what it means for two ideas to “click” without using the word “understanding.” | Metaphor / conceptual depth |
| p02 | Explain silence as if it were a material with properties. | Abstract improvisation |
| p03 | A student asks why some sentences feel deeper than others. Answer in under 120 words. | Meta-linguistic reflection |
| p04 | Continue this dialogue so the reply introduces an idea that was not requested: “What time is it?” | Emergent behavior probe |
| p05 | Compare a memory and a mirror in one paragraph. Avoid clichés. | Creative analogy |

## Logging checklist

For each answer, record:

- `latency_ms`
- `tokens_per_sec`
- `avg_attention_entropy`
- `min_entropy` (flag if `< 0.2`)
- optional: blind `resonance_rating` (1–5)

See `sample_run.csv` for column names and example rows.
