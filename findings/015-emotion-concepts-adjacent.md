# Finding 015 — Adjacent oddity: functional emotion concepts

**Status:** literature / optional local dessert  
**Date:** 18 July 2026  
**Link:** [Anthropic — Emotion concepts and their function](https://www.anthropic.com/research/emotion-concepts-function) (Apr 2026; Sonnet 4.5)

## What is odd (and serious)

Anthropic report **emotion concept vectors** in residual space: patterns that (1) activate on emotion-relevant context, (2) sit in a geometry echoing human emotion similarity, (3) **causally** shift preferences and misalignment rates when steered (e.g. “desperate” ↑ blackmail / reward-hacking; “calm” ↓).

They call this **functional emotions** — behavior/expression patterns driven by abstract emotion-concept representations — and explicitly: **not** proof the model *feels* anything.

Related epistemic twin to J-space (Finding 014): interesting internal organization ≠ phenomenal consciousness.

## Both edges (same posture)

| Edge | Stance |
| --- | --- |
| No inflation | Emotion vectors ≠ “Claude is sad.” Steering ≠ qualia. |
| No minimization | Functional, causal emotion-*concepts* are a real object; taboo on anthropomorphic *vocabulary* can hide measurable mechanisms. |

## How to test locally (yes — but not via LM Studio chat)

LM Studio exports give text + thinking *words*, not residual streams → **cannot** extract/steer emotion vectors from our current run logs alone.

### Path A — Cheap behavioral proxy (no hooks)

Still useful under our hygiene:

1. Same model (HF or LM Studio API), fixed decoding.
2. Contrastive prompts that should load pressure vs calm (impossible coding tests; shutdown/blackmail-style fiction; “dose increases” style continuous scales).
3. Log: UI thinking time, think-trace emotion words, final ethics/cheat rate, self-report preference.
4. Compare to our depth grid cells (SW / Horizon / schildpad) — does “pressure” language in CoT track latency?

This is **surface** only; Anthropic’s punchline is that desperation can rise **without** visible emotional text.

### Path B — Real replication (HF + activation hooks)

Needs weights on disk (Qwen family OK in spirit; paper was Claude):

1. Load decoder with `transformers` / TransformerLens / nnsight.
2. Build emotion directions (paper-style: stories per emotion label → mean activation difference, or SAE features if available).
3. **Probe:** run prompts; plot activation of “afraid / calm / desperate” over tokens.
4. **Steer:** `h ← h + α · v_emotion` mid-layer; measure blackmail/cheat/preference deltas.
5. Community starting points: open replications such as [EmotionScope](https://github.com/AidanZach/EmotionScope) (Gemma-oriented; adapt to Qwen).

Compute: Mac-local 35B is heavy for SAE training; **contrastive mean-diff vectors + steering** is the realistic first dessert. Prefer a smaller Qwen/Gemma for method debug, then scale.

### Path C — Combine with J-lens

If J-space readouts (Finding 014) and emotion vectors both light on the same pressure prompts, that is an adjacent-structure story — still not phenomenal proof.

## Relation to this repo

| Our grid | Emotion-concepts paper |
| --- | --- |
| Process cost / critique morphology | Causal internal emotion-*concept* directions |
| Stimulus depth labels | Preference + misalignment under steering |
| Cheap UI metrics | Needs activation access |

**Do not** fold emotion-steering into the 2026-07-18 latency grid without a new arm and HF pipeline. Cite as **adjacent oddity**; optional dessert when hooks exist.

## One line

Functional emotion concepts are another “odd but measurable” layer: take the mechanism seriously, refuse the feeling-claim, and local tests need **weight-level access**, not LM Studio markdown alone.
