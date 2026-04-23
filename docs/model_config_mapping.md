# Model Config Mapping — SimuMax ↔ HuggingFace

This document explains the JSON schema SimuMax uses for `configs/models/*.json`,
how each field is consumed by the simulator, and how to populate it from the
`config.json` file that ships with a model on the HuggingFace Hub. It also
cross-references the five models already checked in (`llama3-70b`,
`llama3-405b`, `qwen3-32b`, `deepseek_v3`, `kimi`).

All source-code references below point at the current state of the repo
(`simumax/core/...`); line numbers may shift as the code evolves.

---

## 1. SimuMax `ModelConfig` schema

The authoritative definition lives in `simumax/core/config.py` as the
`ModelConfig` dataclass (~line 892). Only fields that are actually consumed by
the simulator are listed here — plus the post-init defaulting rules that
surprise most users.

### 1.1 Core transformer geometry (always required)

| Field | Type | Used by (code) | Meaning |
|---|---|---|---|
| `hidden_size` | int | Embedding, every linear, every LayerNorm | Model width / residual stream dimension. |
| `layer_num` | int | `LLMModel` layer loop, PP splitting (`perf_llm.py`) | Total number of transformer blocks. |
| `vocab_size` | int | Embedding, output linear, loss FLOPs (`flops_per_token`) | Effective vocab size. Can be padded at runtime via `maybe_pad_vocab_size(tp_size)` — SimuMax stores the original in `orig_vocab_size`. |
| `head_num` | int | Non-MLA QKV projection, MLA Q/KV up-projections | Number of attention query heads. |
| `kv_head_num` | int | Non-MLA QKV projection (`dense_module.py:2050`) | KV heads for GQA/MQA. For MLA models this is typically set equal to `head_num` — the simulator's MLA path does not actually consume it (MLA derives KV shape from `kv_lora_rank` + `qk_pos_emb_head_dim`). |
| `head_size` | int | Non-MLA QKV projection and `CoreAttention` (`dense_module.py:2077`) | Per-head hidden width. Required for non-MLA; for MLA attention it is not used in the compute path but is still kept for book-keeping. |
| `intermediate_size` | int | Dense MLP width, and the default for `moe_ffn_hidden_size` when unset | FFN hidden width for dense layers. |

### 1.2 Activation / normalization

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `use_swiglu` | — (must set) | `MLP`, `ExpertMLP`, `flops_per_token`, `mlp_elements` | `True` ⇒ gated MLP (SwiGLU, 3 projections: gate/up/down); `False` ⇒ plain MLP (2 projections). All modern open-weights LLMs (Llama, Qwen, DeepSeek, Kimi, Mistral, Gemma) use SwiGLU. |

SimuMax always uses RMSNorm for the block norms (`norm_type="rms_norm"` hard-coded in
`language_model.py`). HF configs using `LayerNorm` (e.g., classic GPT-2) are
therefore **not directly simulable**; they would need a code change.

### 1.3 Attention variant

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `attention_type` | `"mha"` | `LLMBlock` dispatch (`language_model.py:138`) — picks `MLAAttention` vs `Attention` | Either `"mha"` / `"gqa"` / `"mqa"` (all three funnel into the non-MLA `Attention`, distinguished only by `head_num` vs `kv_head_num`) or `"mla"`. |

For `attention_type == "mla"` the following five fields become **required**:

| Field | Used by | Meaning |
|---|---|---|
| `q_lora_rank` | `dense_module.py:2178-2221` | Low-rank factoring of the Q projection. If `None` the Q projection is built as a single full-rank matrix; if set, Q is split into down-proj → RMSNorm → up-proj. |
| `kv_lora_rank` | `dense_module.py:2223-2254` | Latent dim used for the compressed KV cache (the whole point of MLA). |
| `qk_head_dim` | `dense_module.py:2163, 2247` | Per-head Q/K dimension that goes through the score computation. (Called `qk_nope_head_dim` on HF.) |
| `qk_pos_emb_head_dim` | `dense_module.py:2163, 2226` | Per-head Q/K dimension carrying the rotary positional embedding. (Called `qk_rope_head_dim` on HF.) |
| `v_head_dim` | `dense_module.py:2162, 2274` | Per-head V dimension — decoupled from Q/K in MLA. |

The total per-head Q/K size used in scaled-dot-product is
`q_head_dim = qk_head_dim + qk_pos_emb_head_dim` (line 2163).

### 1.4 MoE

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `model_type` | `None` → auto-filled in `__post_init__` | `perf_llm.py` reporting only | `"dense"` or `"moe"`. If unset, defaults to `"moe"` when `expert_num > 1` else `"dense"`. It does **not** control dispatch — the block picks dense vs expert MLP from `expert_num == 1` or `use_dense=(i < dense_layers)` directly. |
| `expert_num` | `1` | `ExpertMLP` construction, `flops_per_token`, param counting | Total number of routed experts. |
| `topk` | `None` | `ExpertMLP`, `flops_per_token` (uses `topk-1` as the activation multiplier) | Experts activated per token. |
| `moe_ffn_hidden_size` | `None` → falls back to `intermediate_size` in `__post_init__` and in `init_from_config_file` | Expert MLP widths (`moe_module.py:1437`) | FFN hidden size per expert. |
| `moe_shared_expert_intermediate_size` | `None` | `ExpertMLP` — adds a `shared_expert` branch when set (`moe_module.py:1446`) | If set, a dense "always-on" shared expert of this width is added in parallel to the routed experts. |
| `dense_layers` | `0` | `perf_llm.py:693-746` distributes them across PP stages; `language_model.py:249` decides per-block whether to build `MLP` or `ExpertMLP` | Number of **leading** layers that are kept dense (no routing) in an otherwise-MoE model. Matches HF's `first_k_dense_replace`. |
| `moe_pad_expert_input_to_capacity`, `capacity`, `group_linear_mode` | `True`, `1`, `"parallel"` | `Permutation`, grouped-GEMM path | Dispatch/padding knobs. For most sims the defaults are fine. |

### 1.5 Misc / bookkeeping

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `None` | Free-form identifier — only printed, never used in computation. |
| `orig_vocab_size` | `None` | Populated by `maybe_pad_vocab_size(tp_size)`; stores the pre-padding vocab. Don't set by hand. |
| `make_vocab_size_divisible_by` | `128` | Padding granularity applied with `maybe_pad_vocab_size`. |
| `padded_vocab_size` | `True` | Enables the Megatron-style vocab-padding at sim start. |

---

## 2. HuggingFace `config.json` ↔ SimuMax `ModelConfig`

Naming follows the HuggingFace transformers `LlamaConfig` / `Qwen3Config` /
`DeepseekV3Config` conventions. Where HF uses different keys per model family
that map to the same concept, every alias is listed.

| SimuMax field | HF key(s) | Notes |
|---|---|---|
| `hidden_size` | `hidden_size` | 1-to-1. |
| `layer_num` | `num_hidden_layers` | 1-to-1. |
| `vocab_size` | `vocab_size` | 1-to-1. `orig_vocab_size` is derived by SimuMax. |
| `head_num` | `num_attention_heads` | 1-to-1. |
| `kv_head_num` | `num_key_value_heads` | HF defaults this to `num_attention_heads` when absent (classic MHA). For MQA/GQA it is explicit. |
| `head_size` | `head_dim` if present, else `hidden_size / num_attention_heads` | HF only added `head_dim` recently (Qwen3, Llama-3.1-405B). For older Llama 3 70B configs it's implicit. |
| `intermediate_size` | `intermediate_size` | For MoE models this is the **dense-layer** FFN width (first_k layers); it is *not* the expert width. |
| `use_swiglu` | infer from `hidden_act` + architecture | `hidden_act == "silu"` in a Llama/Qwen/DeepSeek/Kimi/Mistral/Gemma model ⇒ `True`. Classic GPT-2 style MLPs with `gelu`/`gelu_new` ⇒ `False`. |
| `attention_type` | infer from `model_type` / presence of `q_lora_rank` | `"mla"` when HF model_type is `deepseek_v2`, `deepseek_v3`, `kimi_k2`, or any config that has `q_lora_rank` / `kv_lora_rank` / `qk_nope_head_dim`. Otherwise `"mha"`. |
| `q_lora_rank` | `q_lora_rank` | MLA only. HF allows `null`; SimuMax treats `None` as "no Q LoRA". |
| `kv_lora_rank` | `kv_lora_rank` | MLA only. |
| `qk_head_dim` | `qk_nope_head_dim` | **Name change.** |
| `qk_pos_emb_head_dim` | `qk_rope_head_dim` | **Name change.** |
| `v_head_dim` | `v_head_dim` | 1-to-1 (MLA). |
| `expert_num` | `n_routed_experts` (DeepSeek/Kimi) / `num_experts` (Mixtral, Qwen-MoE) | HF's MoE naming varies by family. |
| `topk` | `num_experts_per_tok` | 1-to-1 on every MoE family we've seen. |
| `moe_ffn_hidden_size` | `moe_intermediate_size` (DeepSeek/Kimi) / `moe_intermediate_size` or `intermediate_size` (Qwen-MoE) | **Name change.** Per-expert FFN width. |
| `moe_shared_expert_intermediate_size` | `n_shared_experts * moe_intermediate_size` (DeepSeek/Kimi) or `shared_expert_intermediate_size` (Qwen-MoE) | DeepSeek/Kimi encode the shared expert as a *count* × the same intermediate; the repo's convention is to fold that into a single width value. |
| `dense_layers` | `first_k_dense_replace` | Leading dense layers in an MoE model. |
| `model_type` | derived (`"moe"` iff `expert_num > 1`) | HF's `model_type` is an architecture tag (`llama`, `qwen3`, `deepseek_v3`, `kimi_k2`) and is not directly comparable. |
| `model_name` | free | Convention in this repo: lowercase, underscores (e.g. `deepseek_v3`, `llama3_70b`). |

### 2.1 Fields that **only** live in SimuMax (no HF equivalent)

Set these to the defaults unless you know otherwise:

- `moe_pad_expert_input_to_capacity`, `capacity`, `group_linear_mode` — expert
  dispatch model; defaults match Megatron-Core behaviour.
- `make_vocab_size_divisible_by`, `padded_vocab_size` — Megatron-style vocab
  padding.

### 2.2 Fields that live in HF but **SimuMax ignores**

These are safe to drop when hand-writing a SimuMax config:

- `max_position_embeddings`, `rope_theta`, `rope_scaling`, `sliding_window`,
  `attention_dropout`, `attention_bias`, `tie_word_embeddings`, `torch_dtype`
  — SimuMax controls sequence length, dtype, and sparsity through
  `StrategyConfig` / `DisturbanceConfig`, not `ModelConfig`.
- `quantization_config` — quantization is a `StrategyConfig.fp8` knob.
- `bos_token_id`, `eos_token_id`, `initializer_range`, `rms_norm_eps` —
  irrelevant to perf simulation.
- DeepSeek-specific routing: `n_group`, `topk_group`, `topk_method`,
  `scoring_func`, `routed_scaling_factor`, `norm_topk_prob`,
  `aux_loss_alpha`, `seq_aux` — SimuMax models the router as a single dense
  linear regardless of the exact scoring function.
- `num_nextn_predict_layers` (multi-token prediction head in DeepSeek) —
  **not modeled**. If you care about MTP you need to extend the simulator.
- `moe_layer_freq` — SimuMax only supports the "contiguous dense prefix,
  then all-MoE" pattern (via `dense_layers`); interleaved MoE patterns are
  not expressible.

---

## 3. Per-model cross reference

Each column shows the value in the SimuMax config file alongside the value in
the upstream HuggingFace config.

### 3.1 `llama3-70b.json` ↔ `meta-llama/Meta-Llama-3-70B`

| SimuMax | value | HF key | HF value | match? |
|---|---|---|---|---|
| model_type | dense | — (derived: `expert_num==1`) | — | ✓ |
| hidden_size | 8192 | hidden_size | 8192 | ✓ |
| head_num | 64 | num_attention_heads | 64 | ✓ |
| kv_head_num | 8 | num_key_value_heads | 8 | ✓ |
| head_size | 128 | implicit (8192/64) | 128 | ✓ |
| intermediate_size | 28672 | intermediate_size | 28672 | ✓ |
| layer_num | 80 | num_hidden_layers | 80 | ✓ |
| vocab_size | 128256 | vocab_size | 128256 | ✓ |
| use_swiglu | true | hidden_act=silu (Llama MLP) | → true | ✓ |

### 3.2 `llama3-405b.json` ↔ `meta-llama/Meta-Llama-3.1-405B`

| SimuMax | value | HF key | HF value | match? |
|---|---|---|---|---|
| hidden_size | 16384 | hidden_size | 16384 | ✓ |
| head_num | 128 | num_attention_heads | 128 | ✓ |
| **kv_head_num** | **16** | **num_key_value_heads** | **8** | **✗ — SimuMax value disagrees with HF** |
| head_size | 128 | head_dim | 128 | ✓ |
| intermediate_size | 53248 | intermediate_size | 53248 | ✓ |
| **layer_num** | **128** | **num_hidden_layers** | **126** | **✗ — SimuMax value disagrees with HF** |
| vocab_size | 128256 | vocab_size | 128256 | ✓ |
| use_swiglu | true | hidden_act=silu | → true | ✓ |

> ⚠️ Two discrepancies on 405B. Both are plausibly intentional simplifications
> for divisibility (128 layers splits evenly across PP=8/16; 16 KV heads
> divides by larger TP sizes), but I can't confirm that from the codebase
> alone. Worth asking whoever authored `configs/models/llama3-405b.json`
> before using it as ground-truth for accuracy studies.

### 3.3 `qwen3-32b.json` ↔ `Qwen/Qwen3-32B`

| SimuMax | value | HF key | HF value | match? |
|---|---|---|---|---|
| hidden_size | 5120 | hidden_size | 5120 | ✓ |
| head_num | 64 | num_attention_heads | 64 | ✓ |
| kv_head_num | 8 | num_key_value_heads | 8 | ✓ |
| head_size | 128 | head_dim | 128 | ✓ |
| intermediate_size | 25600 | intermediate_size | 25600 | ✓ |
| layer_num | 64 | num_hidden_layers | 64 | ✓ |
| vocab_size | 151936 | vocab_size | 151936 | ✓ |

### 3.4 `deepseekv3.json` ↔ `deepseek-ai/DeepSeek-V3`

| SimuMax | value | HF key | HF value | match? |
|---|---|---|---|---|
| attention_type | mla | (derived: has q_lora_rank) | mla | ✓ |
| hidden_size | 7168 | hidden_size | 7168 | ✓ |
| head_num | 128 | num_attention_heads | 128 | ✓ |
| kv_head_num | 128 | num_key_value_heads | 128 | ✓ (ignored by MLA path) |
| head_size | 128 | — | — | (unused for MLA, kept for API) |
| intermediate_size | 18432 | intermediate_size | 18432 | ✓ (dense-prefix width) |
| moe_ffn_hidden_size | 2048 | moe_intermediate_size | 2048 | ✓ |
| moe_shared_expert_intermediate_size | 2048 | n_shared_experts×moe_intermediate_size (1×2048) | 2048 | ✓ |
| layer_num | 61 | num_hidden_layers | 61 | ✓ |
| dense_layers | 3 | first_k_dense_replace | 3 | ✓ |
| expert_num | 256 | n_routed_experts | 256 | ✓ |
| v_head_dim | 128 | v_head_dim | 128 | ✓ |
| qk_head_dim | 128 | qk_nope_head_dim | 128 | ✓ |
| qk_pos_emb_head_dim | 64 | qk_rope_head_dim | 64 | ✓ |
| q_lora_rank | 1536 | q_lora_rank | 1536 | ✓ |
| kv_lora_rank | 512 | kv_lora_rank | 512 | ✓ |
| topk | 8 | num_experts_per_tok | 8 | ✓ |
| vocab_size | 129280 | vocab_size | 129280 | ✓ |
| use_swiglu | true | hidden_act=silu | → true | ✓ |

### 3.5 `kimi-1T.json` ↔ `moonshotai/Kimi-K2-Instruct`

| SimuMax | value | HF key | HF value | match? |
|---|---|---|---|---|
| attention_type | mla | model_type=kimi_k2 (MLA) | mla | ✓ |
| hidden_size | 7168 | hidden_size | 7168 | ✓ |
| head_num | 64 | num_attention_heads | 64 | ✓ |
| kv_head_num | 64 | num_key_value_heads | 64 | ✓ |
| head_size | 128 | — | — | (unused for MLA) |
| intermediate_size | 18432 | intermediate_size | 18432 | ✓ |
| moe_ffn_hidden_size | 2048 | moe_intermediate_size | 2048 | ✓ |
| moe_shared_expert_intermediate_size | 2048 | n_shared_experts×moe_intermediate_size (1×2048) | 2048 | ✓ |
| layer_num | 61 | num_hidden_layers | 61 | ✓ |
| dense_layers | 1 | first_k_dense_replace | 1 | ✓ |
| expert_num | 384 | n_routed_experts | 384 | ✓ |
| v_head_dim | 128 | v_head_dim | 128 | ✓ |
| qk_head_dim | 128 | qk_nope_head_dim | 128 | ✓ |
| qk_pos_emb_head_dim | 64 | qk_rope_head_dim | 64 | ✓ |
| q_lora_rank | 1536 | q_lora_rank | 1536 | ✓ |
| kv_lora_rank | 512 | kv_lora_rank | 512 | ✓ |
| topk | 8 | num_experts_per_tok | 8 | ✓ |
| vocab_size | 163840 | vocab_size | 163840 | ✓ |
| use_swiglu | true | hidden_act=silu | → true | ✓ |

---

## 4. Recipe — adding a new model from a HuggingFace `config.json`

1. **Decide the family.** Is it dense Llama-style, MoE Llama-style (Mixtral,
   Qwen3-MoE, Llama-4), or MLA (DeepSeek-V3, Kimi-K2)? This determines which
   SimuMax fields are required.
2. **Copy the core geometry** (table 2): `hidden_size`, `layer_num`
   (= `num_hidden_layers`), `vocab_size`, `head_num` (= `num_attention_heads`),
   `kv_head_num` (= `num_key_value_heads`, or same as `head_num` if absent),
   `head_size` (= `head_dim`, or `hidden_size / head_num`),
   `intermediate_size`.
3. **Set activation.** For every modern open-weights LLM,
   `use_swiglu: true`. Only override if `hidden_act` is `gelu` / `relu` /
   absent-with-classic-MLP.
4. **If MLA** (any config with `q_lora_rank` / `qk_nope_head_dim` present):
   set `attention_type: "mla"`, copy `q_lora_rank`, `kv_lora_rank`,
   rename `qk_nope_head_dim` → `qk_head_dim`, `qk_rope_head_dim` →
   `qk_pos_emb_head_dim`, copy `v_head_dim`. Keep `head_size: 128` for
   compatibility even though MLA ignores it.
5. **If MoE** (any config with `n_routed_experts` / `num_experts` /
   `moe_intermediate_size`): set `model_type: "moe"`, `expert_num`,
   `topk` (= `num_experts_per_tok`), `moe_ffn_hidden_size` (=
   `moe_intermediate_size`), `dense_layers` (= `first_k_dense_replace`).
   If the HF config declares a shared expert
   (`n_shared_experts` or `shared_expert_intermediate_size`), compute
   `moe_shared_expert_intermediate_size = n_shared_experts ×
   moe_intermediate_size` or use the HF value directly.
6. **Pick a `model_name`** — lowercase, underscores, matches the file name.
7. **Sanity-check** by loading it:
   ```bash
   uv run python -c "from simumax.core.config import ModelConfig; print(ModelConfig.init_from_config_file('configs/models/NEWMODEL.json').param_numel)"
   ```
   Compare the reported `param_numel` (total) and
   `activated_param_numel` (active) to the model card's published numbers —
   if they don't match within a percent or two, something is off.

---

## 5. Known gaps / things to be careful about

Flagging these honestly rather than papering over them:

- **Llama-3.1-405B discrepancies.** The checked-in `llama3-405b.json` has
  `layer_num=128` and `kv_head_num=16`, but HuggingFace reports `126` and `8`.
  Likely intentional (divisibility), but confirm before using for accuracy
  work.
- **SwiGLU detection is heuristic.** There's no single HF field that says
  "this is SwiGLU" — you have to read the architecture class. All Llama /
  Qwen / DeepSeek / Kimi / Mistral / Gemma / Gemma-2 / Gemma-3 MLPs are
  SwiGLU. Classic GPT-NeoX / Pythia / GPT-2 are not.
- **RMSNorm is assumed everywhere.** `language_model.py` hard-codes
  `norm_type="rms_norm"`. Models with plain LayerNorm (classic GPT-2,
  OPT, BLOOM) cannot be simulated accurately without code changes.
- **Multi-token-prediction head is ignored.** DeepSeek-V3's
  `num_nextn_predict_layers=1` adds an MTP head that SimuMax does not model.
  For DeepSeek training simulation this understates FLOPs and memory by a
  small (single-layer) amount.
- **Router is modeled as a dense linear.** HF's `scoring_func` /
  `topk_method` / grouped-routing knobs don't affect the simulated cost.
- **`moe_layer_freq` patterns other than "first-k dense, rest MoE" are not
  expressible** with the current `dense_layers` field. Models that interleave
  MoE and dense layers in arbitrary patterns (if any show up) would need a
  new field.
- **Shared-expert width when `n_shared_experts > 1`.** DeepSeek-V3 and Kimi-K2
  both have `n_shared_experts == 1`, so `moe_shared_expert_intermediate_size`
  simply equals `moe_intermediate_size`. For a hypothetical model with
  `n_shared_experts > 1`, the convention in this repo is to store the
  **total** width (i.e. `n_shared_experts × moe_intermediate_size`), but
  that has not been exercised. Verify against the model card's param count
  before trusting the simulation.
- **Llama-family weights are gated on HuggingFace.** The canonical
  `meta-llama/Meta-Llama-3*` repos require authentication. Mirrors like
  `NousResearch/Meta-Llama-3-70B` or `hugging-quants/Meta-Llama-3.1-405B-BNB-NF4`
  expose the same (un-quantized) `config.json` values and are easier to
  pull programmatically.
