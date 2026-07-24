# 05 — Kimi K3 vs Claude Opus / Fable, and Chinese near-frontier

**Status:** snapshot side-note (mid–late Jul 2026) · **not** tested in this repo  
**Lens:** nullius in verba — independent indices > vendor press; all numbers rot fast  

## What “Fable” is

**Claude Fable 5** is Anthropic’s top *generally available* model line (announced ~9 Jun 2026), priced above Opus (~$10 / $50 per MTok in / out in public writeups). Same underlying capability class as limited-release **Mythos 5** (fewer safety classifiers; Project Glasswing). Opus 4.x remains the prior “max” tier many people still mean by “Opus.”

Sources: [anthropic.com/claude/fable](https://www.anthropic.com/claude/fable), [Fable 5 / Mythos 5 news](https://www.anthropic.com/news/claude-fable-5-mythos-5).

## Kimi K3 (Moonshot AI)

- Released ~**16 Jul 2026**; Moonshot describes a very large MoE-class system (~**2.8T** parameters in press).
- Pitch: frontier-*adjacent* quality, strong agent/coding/automation boards, planned **open weights** (date claims in press — verify before relying).
- API pricing in secondary sources often cited around **~$3 / $15** per MTok (cheaper than Opus 4.8 list in those same writeups).

## Comparison sketch (third-party, not our lab)

| Rough tier (AA Intelligence Index–style reporting) | Models often named |
| --- | --- |
| Top closed frontier | **Claude Fable 5**, **GPT-5.6 Sol** (scores ~59-ish in AA articles) |
| Next band / “Opus-class” | **Kimi K3** (~57), **Claude Opus 4.8** (~56), **GPT-5.5** (same band in AA) |

**Pattern in independent + secondary summaries (Jul 2026):**

- K3 ≈ **Opus 4.8** on broad intelligence indices (sometimes a point above/below — treat as *same band*).
- K3 still **behind Fable 5** (and often Sol) on those same overall indices.
- K3 often **strong / #1 on some agentic or frontend arenas** (e.g. LMArena Frontend Code Arena in launch coverage); weaker or third on some hardest agentic-coding rows (e.g. DeepSWE-style tables in secondary blogs).
- Cost: K3 argued as **cheaper** than Opus/Fable list prices for similar band performance.

Primary independent pointer often cited: [Artificial Analysis on Kimi K3](https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5).

### What this does *not* mean

- Not “K3 > Anthropic.” Fable remains the Anthropic *peak GA* bar in these snapshots.
- Not “Chinese models won.” Band-overlap ≠ dominance on safety, refusal behavior, long-horizon reliability, or enterprise controls.
- Vendor tables and arena Elos are **partial instruments**; benchmaxxing and harness sensitivity are real (commentators note this explicitly for K3).

## Chinese labs: “near frontier”?

**Nullius-compatible claim:** several Chinese labs (Moonshot/Kimi, Alibaba/Qwen, DeepSeek, Zhipu/GLM, …) now routinely land **inside or next to** Western frontier *bands* on public coding/agent/intelligence boards, often at **much lower $/token**, with a growing **open-weight** story (DeepSeek, planned K3 weights, Qwen ecosystem).

**Safer phrasing than hype:**

> Chinese frontier labs have **closed much of the capability gap** on measurable public benchmarks and agent harnesses; they are **near-frontier on those axes**, especially on price/performance. They have **not** (as of this note) uniformly displaced Fable/Sol-class closed peaks, nor erased differences in eval, safety posture, or geopolitics of access.

Your own local stack already touches this world: **Qwen3.6-35B** in the Beyond Benchmark grid is a *small* open sibling of that ecosystem — useful for garage science, not a claim that 35B = K3/Fable.

## Relevance to this repo

| This project | This note |
| --- | --- |
| n=1 latency / morphology on one local Qwen | Landscape context only |
| No K3 / Opus / Fable runs here | Do not import their scores into Finding 007 |
| Otium / open exploration | Open-weight near-frontier *matters* for who can replicate J-lens / emotion-steer / latency grids |

## Re-check before citing

Numbers and ranks move weekly. Before any strong claim: open Artificial Analysis (or equivalent), LMArena boards you care about, and Anthropic/Moonshot primary posts — then update this file’s date line.
