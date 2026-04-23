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

## Open Weights Big Models Reference

Filtered view of the table above: for each family, only variants with more than 50B total parameters are kept (MoE sized by total params). If no variant in a family clears 50B, the single largest available version is kept.

| Family / Variant | Dense or MoE | Versions kept (params) | Notes |
|---|---|---|---|
| Gemma 3 (Google) | Dense | 27B | Biggest available (no >50B variant) |
| Gemma 4 (Google) | Dense | 31B | Biggest available (no >50B variant) |
| Qwen3 (MoE) | MoE | 235B/22B, 480B/35B (Coder) | Qwen3-Coder-480B-A35B is the largest Qwen open-weights |
| Qwen3.5 Medium (MoE) | MoE | 122B/10B, 397B/17B | 397B is the flagship MoE |
| DeepSeek R1 | MoE | 671B/37B | Based on V3-Base |
| Kimi K2 (Moonshot) | MoE | K2-Base — ~1T/32B | Largest open-weights LLM released to date|
| GLM-4.5 (Z.ai) | MoE | 355B/32B, Air 106B/12B | MIT license |
| GLM-4.6 (Z.ai) | MoE | 355B/32B | 200K context |
| MiniMax M2 family | MoE | M2.5, M2.7 — 230B/10B | M2.5 scored 80.2% SWE-Bench Verified; M2.7 leads open-weights on GDPval-AA |
| Llama 3.1 (Meta) | Dense | 70B, 405B | 405B is the largest dense open model |
| Mistral Large 3 | MoE | 675B/41B | Dec 2025 |
| Mixtral (Mistral, older) | MoE | 8x22B (141B/39B) | Apr 2024; Apache 2.0 |
| OpenAI gpt-oss | MoE | gpt-oss-120b (117B/5.1B) | Aug 2025; native MXFP4; Apache 2.0 |
| OLMo 2 (AllenAI) | Dense | 32B | Biggest available (no >50B variant); fully open |
| Ling-1T (inclusionAI) | MoE | 1T/~50B (+ Ling-2.5-1T, FP8 variants) | First FP8-trained 1T foundation model |

---

## SimuMax config coverage and caveats

Every family in the filtered table above has at least one corresponding JSON
in `configs/models/`. See `docs/model_config_mapping.md` for the per-field
HF ↔ SimuMax mapping used to build them. This section records what was
skipped, where the simulator abstracts away architectural details, and where
the param-count sanity check diverges from the published model card.

### Coverage

Files currently in `configs/models/` (all validated by loading through
`ModelConfig.init_from_config_file` and comparing `param_numel` /
`activated_param_numel` against the model card):

| Big-models entry | SimuMax file(s) | Notes |
|---|---|---|
| Gemma 3 27B | `gemma3-27b.json` | See Gemma caveats below. |
| Gemma 4 31B | `gemma4-31b.json` | See Gemma caveats below. |
| Qwen3 MoE 235B/22B | `qwen3-235b-a22b.json` | — |
| Qwen3-Coder 480B/35B | `qwen3-coder-480b-a35b.json` | — |
| DeepSeek R1 671B/37B | `deepseek-r1.json` (and `deepseek-v3.json`, same architecture) | — |
| Kimi K2 (K2-Base/Instruct/Thinking) ~1T/32B | `kimi-k2.json` | All K2.x variants share the same architecture; one file covers the family. |
| GLM-4.5/GLM-4.5 355B/32B | `glm-4.5.json` | Architecturally identical to GLM-4.6; only context length differs. |
| GLM-4.5 Air 106B/12B | `glm-4.5-air.json` | — |
| MiniMax M2 / M2.5 230B/10B | `minimax-m2.json` | M2.5 shares M2's architecture. |
| Llama 3.1 70B | `llama3-70b.json` | Same architecture as Llama 3 70B (context length only differs, which `ModelConfig` ignores). |
| Llama 3.1 405B | `llama3-405b.json` | Pre-existing; has two known discrepancies vs HF (see §3.2 of the mapping doc). |
| Mixtral 8x22B 141B/39B | `mixtral-8x22b.json` | — |
| gpt-oss-120b 117B/5.1B | `gpt-oss-120b.json` | See gpt-oss caveats below. |
| OLMo 2 32B | `olmo2-32b.json` | — |
| Ling-1T ~1T/50B | `ling-1t.json` | — |

`configs/models/` also contains `llama4-scout.json` and `llama4-maverick.json`
from an earlier pass; the Llama 4 family has since been dropped from the
big-models table above but the files are kept for reference.

### Architectural features SimuMax does not (yet) model

These won't prevent a config from loading, but they mean the simulated
perf/memory numbers are approximations:

- **Sliding-window or hybrid attention patterns.** gpt-oss alternates
  sliding/full attention per layer; Gemma 3 and Gemma 4 use a sliding-window
  pattern with periodic full-attention layers. SimuMax models every layer
  as full attention ⇒ slight over-count of attention FLOPs / KV memory.
- **GeGLU vs SwiGLU.** Gemma 3/4 use gated GeLU (3-projection MLP). The
  config uses `use_swiglu: true` because the parameter count and FLOPs are
  structurally identical to SwiGLU; only the elementwise activation kernel
  differs, and SimuMax's activation cost model is kernel-agnostic at first
  order.
- **Tied word embeddings.** Gemma 3 and Gemma 4 set
  `tie_word_embeddings=true`. `ModelConfig` has no way to express this and
  always counts an independent output projection ⇒ overstates total params
  by `hidden_size × vocab_size` (a few percent for the 27–31B Gemmas).
- **Multi-Token Prediction (MTP) modules.** DeepSeek V3/R1 declare
  `num_nextn_predict_layers=1`; MiniMax M2 declares `num_mtp_modules=3`;
  GLM-4.5/4.6 also have MTP. These auxiliary layers are not modeled ⇒
  slight under-count of training FLOPs/memory when they're actually used.
- **Non-standard routing** (sigmoid scoring, grouped top-k,
  `noaux_tc`, routing bias, shared-expert sigmoid gate). SimuMax models
  the router as a single dense linear regardless of scoring function.
- **FP8 / MXFP4 native weights** (DeepSeek, Kimi, gpt-oss, MiniMax M2
  ship pre-quantized). Quantization is a `StrategyConfig.fp8` knob — not
  taken from `config.json`.

### Known param-count divergences from model cards

Validation harness loads each config and prints
`param_numel` / `activated_param_numel`. Observed gaps and their root
causes (all trace back to simplifications in
`simumax/core/config.py :: ModelConfig`, **not** to errors in the JSON
configs):

1. **`activated_param_numel` omits the shared expert.** `layer_act_elements`
   only counts `topk × mlp_elements`; it does not add the shared-expert
   block even when `moe_shared_expert_intermediate_size` is set. This is
   why GLM-4.5 reports 28.1B active vs 32B advertised, DeepSeek reports
   34.8B vs 37B, Kimi 30.0B vs 32B, Ling-1T 46.9B vs 50B.
2. **`param_numel` applies the MoE layer formula to every layer, ignoring
   `dense_layers`.** The leading dense layers are counted as if they had
   `expert_num × moe_ffn_hidden_size` weight. This inflates totals on
   every MoE model with a dense prefix: DeepSeek/R1 701B vs 671B,
   Kimi-K2 1040B vs 1000B, GLM-4.5 358B vs 355B, Ling-1T 1045B vs 1000B.
3. **Tied embeddings aren't modeled** (as noted above) — ~5% high on
   Gemma 27B / 31B.
4. **Llama 3.1 405B config disagrees with HF.** `layer_num=128` vs HF 126
   and `kv_head_num=16` vs HF 8. Pre-existing choices in
   `llama3-405b.json`, likely for divisibility; worth re-checking before
   treating it as ground-truth.

In every case where the number looks off, the JSON matches the upstream
HF `config.json`; the discrepancy is in how SimuMax rolls fields into a
parameter count, not in the config itself. If accuracy of
`param_numel` / `activated_param_numel` matters for a given study, fix the
formulae in `ModelConfig` rather than hand-editing the JSON.
