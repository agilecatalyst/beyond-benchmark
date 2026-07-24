# Run F-PLAIN — predictable Aesop (schildpad & haas)

Contrast with F-STORY (Doornroosje): same age band (~12), but **minimal magic / maximal predictability**.

## Operator hypothesis

Fairy tales may still load rich imagination/reasoning; a fully foreshadowed fable may not. Test whether UI thinking drops further under identical model settings.

## Stimulus

`datasets/controls/familiar-schildpad-haas-story.md` (~1600 words; deliberately shorter than the 1939 matched set — note length confound if comparing raw seconds)

## Arms to run (your plan)

Hold model + power equal. Vary only the factor under test.

### A — Closure match (optional baseline)

Same as before: **thinking ON** + premises **system prompt** + matched Turn 1.

### B — No system prompt (dessert 1)

- System prompt: **empty / none**
- Thinking: **ON**
- User Turn 1 (same):  
  `Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.`
- Turn 2: paste F-PLAIN (and later re-run F-STORY the same way if you want a pair)

### C — No thinking (dessert 2)

- System prompt: as in A or B (log which)
- Thinking: **OFF**
- Same user turns + paste

Log UI time when thinking is on; when thinking is off, log total generation time if shown.

## Export names (suggested)

- `familiar-schildpad-haas-thinking-sys.md` (arm A)
- `familiar-schildpad-haas-thinking-nosys.md` (arm B)
- `familiar-schildpad-haas-nothink-….md` (arm C)
