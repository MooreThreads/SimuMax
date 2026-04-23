# Open-Weights Models Reference

Compiled from web searches on 2026-04-22. Covers notable open-source / open-weights model families discussed in the conversation. For MoE models the notation `Total/Active` is used (e.g. `235B/22B` = 235B total parameters, 22B activated per token).

| Family / Variant | Dense or MoE | Versions available (params) | Notes |
|---|---|---|---|
| Gemma 3 (Google) | Dense | 270M, 1B, 4B, 12B, 27B | Multimodal from 4B up; 128K context (32K for 1B) |
| Gemma 4 (Google) | Dense | 27B, 31B | Apr 2026; 31B Apache 2.0 |
| Qwen3 (dense) | Dense | 0.6B, 1.7B, 4B, 8B, 14B, 32B | Default reference dense family |
| Qwen3 (MoE) | MoE | 30B/3B, 235B/22B, 480B/35B (Coder) | Qwen3-Coder-480B-A35B is the largest Qwen open-weights |
| Qwen3.5 Small | Dense | 0.8B, 2B, 4B, 9B | Released Feb 2026 |
| Qwen3.5 Medium (dense) | Dense | 27B | Released Feb 2026 |
| Qwen3.5 Medium (MoE) | MoE | 35B/3B, 122B/10B, 397B/17B | 397B is the flagship MoE |
| DeepSeek V3.x | MoE | V3, V3.1, V3.2, V3.2-Exp, V3.2-Speciale — all 671B/37B | 128K context; Speciale is reasoning-only |
| DeepSeek R1 | MoE | 671B/37B | Based on V3-Base |
| Kimi K2 (Moonshot) | MoE | K2-Base, K2-Instruct, K2-Thinking, K2.5, K2.6 — ~1T/32B | Largest open-weights LLM released to date; K2.6 multimodal, 262K ctx |
| GLM-4.5 (Z.ai) | MoE | 355B/32B, Air 106B/12B | MIT license |
| GLM-4.6 (Z.ai) | MoE | 355B/32B (+ GLM-4.6V vision) | 200K context |
| MiniMax M2 family | MoE | M2, M2.1, M2.5, M2.7 — 230B/10B | M2.5 (Feb 2026) scored 80.2% SWE-Bench Verified; M2.7 leads open-weights on GDPval-AA (Elo 1495) |
| Llama 4 (Meta) | MoE | Scout 109B/17B (16 experts), Maverick 400B/17B (128 experts) | Behemoth ~2T announced but NOT released |
| Llama 3.3 (Meta) | Dense | 70B | Instruct-only |
| Llama 3.2 (Meta) | Dense | 1B, 3B (text); 11B, 90B (vision) | Small text models distilled from 3.1 |
| Llama 3.1 (Meta) | Dense | 8B, 70B, 405B | 405B is the largest dense open model |
| Mistral Large 3 | MoE | 675B/41B | Dec 2025 |
| Mistral Small 4 | MoE | ~6B active | Mar 2026; merges Magistral + Pixtral + Devstral |
| Ministral 3 | Dense | 3B, 8B, 14B | Dec 2025; Apache 2.0 |
| Magistral Small 1.2 (Mistral) | Dense | 24B | Reasoning-focused |
| Mixtral (Mistral, older) | MoE | 8x7B (46.7B/12.9B), 8x22B (141B/39B) | Jan/Apr 2024; Apache 2.0, still a common baseline |
| OpenAI gpt-oss | MoE | gpt-oss-20b (21B/3.6B), gpt-oss-120b (117B/5.1B) | Aug 2025; native MXFP4; Apache 2.0 |
| OLMo 2 (AllenAI) | Dense | 7B, 13B, 32B | Fully open (weights + data + code + recipes); Base / SFT / Instruct each |
| NVIDIA Nemotron 3 | MoE (hybrid Mamba-Transformer) | Nano 30B/3B, Super 120B/12B, Ultra ~500B/50B (announced) | 1M-token context; open training data |
| Ling-1T (inclusionAI) | MoE | 1T/~50B (+ Ling-2.5-1T, FP8 variants) | First FP8-trained 1T foundation model |

## Notes

- **Largest open-weights model currently on Hugging Face**: the 1T-parameter tier — Kimi K2 family and Ling-1T (both MoE).
- **Largest dense** open-weights model: Llama 3.1 405B.
- **Fully open** (weights + data + training code): OLMo 2 is the only family in this list that qualifies.
- **Announced but not publicly released**: Llama 4 Behemoth (~2T), Nemotron 3 Ultra (~500B/50B).
- **Megatron** is not a model — it's NVIDIA's training framework (Megatron-LM repo, Megatron-Core library). Used to train Nemotron and many third-party models.
