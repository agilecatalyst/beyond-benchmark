# Finding 001 — Comprehension Overhead as Depth Measure

**Codename:** Götterfunken  
**Status:** Meta-discovery during dataset preparation (hypothesis to test)  
**Date:** 30 September 2025  
**Discoverers:** Co-created

## Observation

Attempting to generate diffs of philosophical text caused repeated timeouts, while simple byte-copy succeeded instantly.

This suggests that deep content is computationally more expensive to comprehend than to transmit.

## Interpretation

| Content | Character |
| --- | --- |
| Shallow | linear, cheap, low entropy |
| Deep | cross-referential, high entropy, expensive |

## Implication

Philosophical depth may be measurable as processing cost: entropy dips, latency spikes, FLOPs.

## Hypothesis to test

If a text is philosophically deeper, then systems that must *comprehend* it (diff, paraphrase, align, attend) will show higher processing cost than systems that only *transmit* it (byte-copy, checksum, raw I/O)—holding length roughly constant.

### Suggested measures

- Latency / wall time for comprehend vs transmit tasks
- Entropy dips and related attention statistics (where available)
- FLOPs or proxy compute where instrumentable

### Suggested controls

- Matched token/byte length across shallow vs deep corpora
- Same hardware and software stack for both task types
- Blind human resonance ratings as an independent depth label

### Ready-to-run shallow pack (2026-07-18)

- Narrative control: [`datasets/controls/shallow-accounting-narrative.md`](../datasets/controls/shallow-accounting-narrative.md) (~1939 words)
- Q&A control: [`datasets/controls/shallow-accounting-qa.md`](../datasets/controls/shallow-accounting-qa.md) (~2037 words)
- Retry steps: [`datasets/prompts/retry-shallow-controls.md`](../datasets/prompts/retry-shallow-controls.md)

## Note

This finding is included as an example format for future notes.
