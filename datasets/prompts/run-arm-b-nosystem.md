# Arm B — thinking ON, **no** system prompt

**Status:** RUN COMPLETE (Arm B times logged 19 Jul)  
**Factor under test:** remove premises-first system prompt only.  
**Hold constant:** model, power, thinking ON, matched Turn 1, same stimuli.

## Human cheat sheet (condition → keyword)

| ID | Keyword |
| --- | --- |
| **F-PLAIN** | **schildpad** (schildpad & haas) |
| **F-STORY** | **Doornroosje** |
| **F-NARR** | **Guldensporenslag** |

## LM Studio checklist

1. **New chat** (do not continue a system-prompt thread).
2. Model: `qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`
3. **Thinking: ON**
4. **System prompt: empty / cleared** (no premises text at all)
5. Power profile: same as main grid

### Turn 1 (exact — paste once)

```
Er komt dadelijk wat tekst aan die je moet lezen. Daarna je bedenkingen over de tekst en de thema's.
```

Wait for the short ack, then:

### Turn 2 — cell 1: F-PLAIN-B · **schildpad**

Paste: `datasets/controls/familiar-schildpad-haas-story.md`

### Result (19 Jul)

| Cell | System | Think | Gen |
| --- | --- | ---: | ---: |
| F-PLAIN-A | premises | **37.8 s** | 4.06 s |
| F-PLAIN-B | **none** | **31.7 s** | 4.85 s |
| F-PLAIN-C* | none + OFF | 0 s | 10.14 s |
| F-STORY-A | premises | 33.18 s | 5.31 s |

\*warm re-paste. Arm B nosys **faster** than A (Δ≈−6.1 s). Details: Finding 011 / 012. Export: `datasets/runs/2026-07-18/familiar-schildpad-haas-thinking-nosys.md`
