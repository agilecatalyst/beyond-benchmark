# Methodology notes

## Project ethos

We start where we can. This is **otium** and garage exploration: no grant, no university lab, no product KPI — free-style co-creation because measurement and curiosity are available locally. Small-n and incomplete instruments are accepted as the cost of starting; relevance grows with more test data, not with prestige. Nullius in verba still applies: honest logs beat polished claims.

## Core signals

1. **Attention entropy** — Shannon entropy over attention mass (or an available proxy). Track run-level average and minimum. Flag `min_entropy < 0.2` as a candidate focus event.
2. **Latency** — Wall-clock ms from request start to final token. Compare to the run mean to spot relative slowdowns.
3. **Throughput** — Tokens per second, as a companion to latency.
4. **Human resonance rating** — Blind 1–5 score for perceived depth / “click,” collected after generation so raters do not see model internals.

## Experimental hygiene

- Fix decoding hyperparameters within a comparison set.
- Randomize presentation order for human raters.
- Log model identifier, prompt id, and seed when available.
- Treat small-n correlations as hypotheses, not results.
- **Log thinking-mode on/off** and **system-prompt on/off** as first-class factors (2026-07-18 runs used Qwen3.6-35B-A3B uncensored **with thinking** + premises-first system prompt throughout — do not generalize to non-thinking / bare-chat without a new arm).

## Resonance model (working definition)

A provisional, multi-signal sketch: candidate “resonant” outputs are those that jointly show (a) a flagged entropy dip, (b) above-mean latency for the prompt class, and (c) higher blind human ratings. Weights and thresholds remain open parameters.

## Non-claims (hard scope lock)

- Entropy dips ≠ proof of insight.
- Latency ≠ quality; latency ≠ consciousness.
- **Consciousness** (bio or mechano) is not an object of this project. It is not proven for either class of system in a way this lab can operationalize; as a first-person / subjective (*phenomenal*) category it is **not a measurable research target** here. We do not test, prove, or disprove it.
- What we *do* measure: deliberation proxies (UI thinking time, trace shape), output timing, and stimulus labels (depth pools) — signals about **process cost and critique morphology**, not minds.

## Non-dismissal (equal and opposite)

Refusing to *measure* phenomenal consciousness is **not** a license to treat internal organization as uninteresting or “just autocomplete.”

Adjacent work (outside our instruments):

1. **J-space** / Jacobian lens ([global workspace](https://www.anthropic.com/research/global-workspace), Jul 2026) — emergent privileged subspace with access/workspace-*like* functions. Finding 014.
2. **Functional emotion concepts** ([emotion concepts and their function](https://www.anthropic.com/research/emotion-concepts-function), Apr 2026) — causal emotion-*concept* vectors (e.g. desperate/calm) that steer preferences and misalignment rates without proving subjective feeling. Finding 015. Local test needs HF activation hooks / steering, not LM Studio chat exports.

**Our posture:** keep both edges sharp —

1. Do not smuggle “the model is conscious / feels” from latency, J-space, or emotion vectors.
2. Do not minimize the oddness: emergent workspace-like structure and functional emotion-concept machinery are real scientific objects; negation of *phenomenal* proof ≠ negation of *interesting internal organization*.

This repo stays on cheap, local signals (time, morphology). J-space / emotion-vector interpretability is **cited context** (optional HF dessert), not what the 2026-07-18 grid measures.

Longer side-notes (emergence, parrot pushback, framing, under-pressure agency): [`docs/adjacent/`](adjacent/).

## Stimulus depth pools (a priori)

| Pool | Meaning | Narrative examples |
| --- | --- | --- |
| Deep | cross-referential / metaphor / claim-spanning | D-NARR (*Resonant Shadow*), D-QA |
| Technical-flat | linear factual, domain-dense | S-NARR / S-QA (accounting) |
| Familiar-light | linear known cultural schema, low novelty | F-NARR (Guldensporenslag chronicle) |

Depth is a property of the **input text**, not of thinking time. See Finding 006.
