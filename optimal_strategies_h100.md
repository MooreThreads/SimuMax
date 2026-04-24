# Optimal Fitting Strategies per Model (H100 NVLink)

Results from `examples/search/search_fit_strategy.py`. For each model in `configs/models/`, the script enumerates `(tp, pp, ep, dp)` combinations under the constraint `world_size = tp * pp * dp` (DP is now a first-class search axis, not a post-hoc replication factor), calls `PerfLLM.search_best_parallel_strategy_with_recompute` per combo, and keeps the highest-MFU combination that fits in device memory. The winning strategy for each model is saved as `configs/strategy/<model_name>_optimal.json`.

## Run parameters

- System: `h100_nvlink` (H100 SXM5, 80 GB, NVLink4/NVSwitch intra-node, NDR400 IB inter-node; spec-based starter config — *not* shape-calibrated).
- Sequence length: 4096.
- Micro batch size: 1. Micro batch num per combo: `mbn = max(round(gbs_target / dp), pp)` so the pipeline is always valid (`mbn >= pp`).
- **Per-category target global batch size** (set to mimic realistic training runs):
  - Small (<80 B params): `gbs = 1024`, `max_world = 1024`
  - Medium (80–300 B): `gbs = 2048`, `max_world = 2048`
  - Large (≥300 B): `gbs = 4096`, `max_world = 4096`
- Effective gbs may deviate from the target (kept within `[gbs / 2, gbs * 2]`) because `mbn` is clamped up to `pp`.
- Dtype: bf16. ZeRO stage: 1. Sequence parallel: enabled.
- Memory reserved (gmi_error): 6 GB.
- Recompute types searched: `no_recompute`, `full_block`, `selective_recompute`.

## Search space

| Parameter | Candidates |
|---|---|
| `tp` | 1, 2, 4, 8 (filtered by `head_num % tp == 0` and `kv_head_num % tp == 0`) |
| `pp` | 1, 2, 4, 8, 16, 32 (filtered by `pp <= layer_num` and valid last-stage size) |
| `ep` (MoE) | 1, 2, 4, 8, 16, 32, 64 (filtered by `expert_num % ep == 0`) |
| `ep` (dense) | 1 |
| `dp` | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 (filtered by `tp*pp*dp <= max_world`) |

## Results

### Dense

| Model | Params | tp | pp | ep | dp | world | mbn | gbs | MFU | Recompute | Combos tried | Fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| mistral-large | 147B | 2 | 8 | 1 | 4 | 64 | 512 | 2048 | 0.612 | full_block × 11 | 144 | 38 |
| olmo2-32b | 34B | 1 | 4 | 1 | 4 | 16 | 256 | 1024 | 0.607 | full_block × 16 | 168 | 130 |
| gemma3-27b | 30B | 1 | 4 | 1 | 4 | 16 | 256 | 1024 | 0.603 | full_block × 15 | 150 | 112 |
| llama3-405b | 475B | 4 | 16 | 1 | 4 | 256 | 1024 | 4096 | 0.572 | full_block × 7 | 216 | 32 |
| gemma4-31b | 29B | 1 | 8 | 1 | 2 | 16 | 512 | 1024 | 0.566 | full_block × 7 | 128 | 88 |
| llama3-70b | 79B | 2 | 8 | 1 | 2 | 32 | 512 | 1024 | 0.564 | full_block × 6 | 150 | 76 |

### MoE (and MoE + MLA)

| Model | Type | Params | tp | pp | ep | dp | world | mbn | gbs | MFU | Recompute | Combos tried | Fit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| mixtral-8x22b | moe | 144B | 1 | 8 | 4 | 8 | 64 | 256 | 2048 | 0.500 | full_block × 5 | 480 | 119 |
| gpt-oss-120b | moe | 116B | 1 | 8 | 2 | 8 | 64 | 256 | 2048 | 0.348 | full_block × 5 | 672 | 317 |
| deepseek-r1 | moe + MLA | 701B | 2 | 16 | 8 | 8 | 256 | 512 | 4096 | 0.346 | full_block × 4 | 910 | 142 |
| qwen3-235b-a22b | moe | 234B | 1 | 32 | 1 | 4 | 128 | 512 | 2048 | 0.328 | full_block × 3 | 694 | 282 |
| glm-4.5-air | moe | 106B | 1 | 16 | 1 | 4 | 64 | 512 | 2048 | 0.327 | full_block × 2 | 771 | 405 |
| qwen3-coder-480b-a35b | moe | 478B | 1 | 16 | 8 | 16 | 256 | 256 | 4096 | 0.314 | full_block × 4 | 840 | 154 |
| glm-4.5 | moe | 358B | 1 | 16 | 8 | 8 | 128 | 512 | 4096 | 0.310 | full_block × 5 | 840 | 194 |
| minimax-m2 | moe | 228B | 1 | 8 | 8 | 8 | 64 | 256 | 2048 | 0.283 | full_block × 8 | 771 | 281 |
| ling-1t | moe | 1054B | 4 | 16 | 8 | 8 | 512 | 512 | 4096 | 0.225 | full_block × 5 | 910 | 103 |
| kimi-k2 | moe + MLA | 1045B | 4 | 16 | 8 | 8 | 512 | 512 | 4096 | 0.198 | full_block × 4 | 910 | 107 |

## What changed vs. the previous (DP=EP framing) table

- **DP is now a search axis.** Previously the simulator fixed `world = tp*pp*ep` (DP implicitly equal to EP), with DP replication handled outside the search. DP ∈ {1…4096} is now sweeped, and world is `tp*pp*dp`.
- **Realistic batch sizes.** Global batch is now pinned per category (1024 / 2048 / 4096) rather than tied to `ep*4*pp`. This changes which combos are feasible and restores DP as a real tuning knob.
- **TP finally grows for large models.** With MLA + TP support (see caveats) and a larger world budget, the search now picks `tp=4` for `llama3-405b`, `kimi-k2`, and `ling-1t`. The prior results rarely exceeded `tp=2`.
- **MFU across the board is higher**, since many models previously had to use absurd micro-batch-to-pp ratios or run with DP=EP that forced sub-optimal capacity planning. Dense models now cluster at **0.56–0.61 MFU**; MoE at **0.20–0.50**.
- **Some MoE winners prefer `ep=1`** (`qwen3-235b-a22b`, `glm-4.5-air`). With `ep=1` all experts are replicated on each rank — trading all-to-all comm for memory — which wins when capacity padding + dispatch dominates the cost of parallel experts.

## Caveats

- **MLA + TP is now supported.** The simulator used to assert `tp_size == 1` in `MLAAttention.__init__`. That has been replaced by a Megatron-style layout: down-projs use `disable_tensor_parallel=True` so they produce a full-seq, replicated latent; up-projs use a new `input_is_full_seq=True` flag on `LinearCol` that skips the SP input all-gather and falls back to TP-pattern bwd communication; `linear_out_proj` is now `LinearRow`. `deepseek-r1` now picks `tp=2` and `kimi-k2` picks `tp=4` (both were pinned to `tp=1` before).
- **H100 system config is spec-based, not calibrated.** The file warns of 10–20% error on iter time vs. a calibrated config. MFU absolute values should be read as relative comparisons, not ground truth.
- **MoE modeling uses megatron-style capacity padding** (`moe_pad_expert_input_to_capacity=True`, `capacity=1`, `dispatch_probs=True`), matching `examples/search/llm_search.py`.
- **Search is coarse.** `tp` and `pp` are powers of two; `pp` isn't restricted to exact divisors of `layer_num` but does enforce a non-empty last stage. Finer granularity (e.g., `pp=3,5,6`) could yield better fits — particularly for 61-layer MLA models where 61 is prime.
- **`mbn` is clamped up to `pp` for pipeline validity.** When `gbs / dp < pp`, effective `gbs` overshoots the target. The search accepts any `gbs_eff ∈ [gbs/2, 2*gbs]`.
