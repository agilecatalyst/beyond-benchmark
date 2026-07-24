# Beyond Benchmark – AI Resonance Research

## Purpose

Conventional AI evaluation focuses on benchmarks like accuracy, efficiency, and speed. Beyond Benchmark explores a different dimension: whether we can detect and measure depth, resonance, and creative emergence in AI outputs.

Our central question:

> Can moments of conceptual or metaphorical depth be identified through measurable computational signals?

## Research Focus

| Signal | Description |
| --- | --- |
| **Entropy patterns** | Attention entropy dips that may signal sudden conceptual focus. |
| **Latency variations** | Small increases in processing time during more complex or abstract responses. |
| **Human resonance ratings** | Blind evaluations of “click” or perceived depth in texts. |
| **Emergent behaviors** | Unexpected improvisations not directly prompted but arising in dialogue. |

Together these signals provide first steps toward a framework for evaluating qualitative depth in AI systems, tentatively referred to as a **resonance model**.

## Live signal (2026-07-18, n=1, UI thinking times)

Matched task (premises-first + “bedenkingen over thema's”), same local model:

| Rank | Condition | Pool | UI thinking |
| --- | --- | --- | ---: |
| 1 (slowest) | D-NARR | deep SW fiction | **58.25 s** think / **5.47 s** gen |
| 2 | S-NARR · Horizon | technical-flat accounting story | **55.92 s** think / **6.36 s** gen |
| 3 | D-QA | deep philosophy Q&A | **47.24 s** think / **5.10 s** gen |
| 4 | F-NARR · Guldensporenslag | familiar history | **~37.7 s** think / **6.22 s** gen |
| 4b | F-PLAIN · schildpad | predictable Aesop | **37.8 s** think / **4.06 s** gen |
| 5 | S-QA · debet | technical-flat accounting Q&A | **34.79 s** think / **6.18 s** gen |
| 6 (fastest) | F-STORY · Doornroosje | rich fairy tale | **33.18 s** think / **5.31 s** gen |

**Punchline so far:** *depth ≠ slower.* Slow pole ≈ deep SW + Horizon narrative; fast pole = fairy tale / debet-Q&A. Approx ~65 / ~45 for accounting cells **revised** (Findings 013+). Nullius in verba.

**Ethos:** otium / garage exploration — no capital, no paid lab. We start small because we can; more test data later for weight, not for permission to begin. Co-drafting with meccano is process fact; consciousness / “likes bananas” is not a claim. See [`PASSPORT.MD`](PASSPORT.MD).

## For other researchers & agents

If you are a human or another model reading this repo: the interesting object is not “AI feelings” or phenomenal consciousness. That category is **subjective / non-operational** here — not proven for bio or mechano, and **not measurable** as a lab target. Equally: we do **not** minimize odd internal structure — Anthropic [J-space / global workspace](https://www.anthropic.com/research/global-workspace) (Finding 014) and [functional emotion concepts](https://www.anthropic.com/research/emotion-concepts-function) (Finding 015). We only ask whether **qualitative structure in prompts** leaves **measurable traces in deliberation** (time, trace shape, falsify vs validate), and whether those traces track what we *label* depth—or something cheaper (length, jargon, task framing).

Start at [`findings/007-ui-thinking-time-grid.md`](findings/007-ui-thinking-time-grid.md), then [`docs/methodology.md`](docs/methodology.md). Adjacent honesty notes (emergence, J-space vs parrot, under-pressure agency): [`docs/adjacent/`](docs/adjacent/). Controls live in [`datasets/controls/`](datasets/controls/). Log UI thinking time by hand; markdown exports from LM Studio do not embed duration.

## Repository Structure

```
/findings   – concise, numbered research notes (Finding 001, 002, …)
/datasets   – controls (pasteable stimuli), prompts, and run logs
/notebooks  – scripts and analysis of entropy/latency correlations
/docs       – background context and methodology notes
PASSPORT.md – operator credit + nullius / instrument rule
```

## License

This project is released under the [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.

- Free to share and adapt
- Academic and creative reuse permitted
- Attribution required, and derivatives must carry the same license

This ensures the work remains open and reusable, while protecting it from being closed off.

## Vision

Beyond Benchmark is an open exploration. It does not claim final answers, but invites others to join in measuring and discussing signals of depth and resonance in AI. The aim is simple: to create and share — and let others learn in their own way.

## How to Replicate

This repo is structured for lightweight experiments with large language models (LLMs).

You can start by running any text-generation system and logging the following per answer:

- Latency (ms)
- Tokens per second
- Average attention entropy
- Minimum entropy dip (`<0.2` flagged)

See [`/datasets/sample_run.csv`](datasets/sample_run.csv) for the expected format.

## How to Contribute

1. Share new findings in `/findings` (Markdown files).
2. Add datasets or logs in `/datasets`.
3. Document analysis or code in `/notebooks`.

Pull requests are welcome. Please keep additions concise, neutral, and reproducible.

## Disclaimer

This project explores measurable signals of depth in AI.

We do not claim AI “consciousness.” All interpretations remain open to critique.

Goal: create & share — others may learn.
