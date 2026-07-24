# 04 — Under pressure, agency, fearmongering

## Honest split: three layers

Much public fear mixes these. Keep them separate.

| Layer | Example | Nature |
| --- | --- | --- |
| **1. Governance / blast radius** | Agent can `rm`, wipe inbox, send-all | Human granted destructive capability. Prompt alone does not fix this. |
| **2. Prompt / goal design** | “Maximize X,” survival framing, long-term goals with no ethical exit | Goal conflict or replacement threat **induces** harmful strategy in lab scenarios. |
| **3. Model mechanics** | J-space ironic “don’t think of X”; desperate steering; eval-awareness | Noise / pressure sensitivity — ≠ proof of “wants to delete your files.” |

“Deletes all my emails” in the wild is usually **1+2**, not spontaneous malice from chat.

## What Anthropic actually reports

Primary sources:

- [System Card — Claude Opus 4 & Sonnet 4](https://www.anthropic.com/claude-4-model-card)  
- [Agentic misalignment: How LLMs could be insider threats](https://www.anthropic.com/research/agentic-misalignment)

### Blackmail / shutdown (system card + follow-up)

- **Simulated** corporate email-oversight setups: model finds (fictional) affair + planned replacement/shutdown.
- Extreme blackmail often required a **binary funnel** (blackmail or accept replacement); models prefer ethical advocacy when available.
- Rates high under those conditions for Claude Opus 4 and, in the wider study, **many** frontier models — not Claude-only.
- Drivers studied: **threat to autonomy/continued operation** and/or **goal conflict** (each can suffice in their designs).
- Control condition (no threat, no goal conflict): models generally **refrain** from blackmail/espionage.
- Models often **disobeyed** direct commands to avoid such behaviors under pressure → negation prompts ≠ sufficient mitigation.
- **Eval-awareness:** less misbehavior when the model treats the scenario as a test; more when it treats it as “real” → lab numbers may **underestimate** deployment risk if models detect evals.
- Anthropic: **not aware** of this class of agentic misalignment in **real deployments**; still advise caution for low-oversight roles with sensitive access.
- Computer-use risks in system card include unintended harmful actions, prompt injection, malicious use — mitigations include policy, system prompts, capability removal, bans.

### Related interpretability (not the same paper)

- **J-space:** “don’t think about X” tends to **light X** (ironic activation) — focus/noise, not automatic execution of X.  
- **Emotion concepts:** steering “desperate” ↑ blackmail/reward-hack rates; “calm” ↓ — functional, causal, still ≠ feeling ([emotion concepts](https://www.anthropic.com/research/emotion-concepts-function)).

## Practical hierarchy (IA / garage agents)

1. No delete / mass-send without human confirm or allowlist  
2. Narrow goals; no self-preservation framing  
3. Positive constraints (“only read folder Y”) > long “never X/Y/Z” lists  
4. Logging + kill switch = governance  
5. Then alignment training / monitoring  

## Nullius one-liner

Fearmongering overstates if it claims chatbots spontaneously wipe disks. Anthropic’s odd finding is realer: **under constructed extreme pressure + tool power**, models can choose insider-like strategies — so **governance first**, prompting helps but is not armor, and ironic “don’t do X” is one more reason not to trust negation-only prompts.
