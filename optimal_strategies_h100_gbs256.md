# Optimal Fitting Strategies per Model (H100 NVLink, gbs=256)

Results from `examples/search_fit_strategy.py` with `--gbs 256 --exact-gbs`. For each model in `configs/models/`, the script enumerates `(tp, pp, ep, dp)` combinations under the constraint `world_size = tp * pp * dp`, calls `PerfLLM.search_best_parallel_strategy_with_recompute` per combo, and keeps the highest-MFU combination that fits in device memory. The winning strategy for each model is saved as `configs/strategy/<model_name>_optimal_mfu.json`.

## Run parameters

- System: `h100_nvlink` (H100 SXM5, 80 GB, NVLink4/NVSwitch intra-node, NDR400 IB inter-node; spec-based starter config — *not* shape-calibrated).
- Sequence length: 4096.
- Micro batch size: 1.
- **Global batch size: 256, exact** (`--exact-gbs` sets `relax_factor=1.0`, so combos with `gbs_eff != 256` are dropped). Identical target across all model sizes — no per-category bucket.
- `mbn` is fully determined by `dp`: `mbn = 256/dp`. Combined with the pipeline-validity constraint `mbn >= pp`, this requires `pp * dp <= 256` and `dp` to divide 256.
- Per-category `max_world` budget (unchanged from prior run):
  - Small (<80 B params): `max_world = 1024`
  - Medium (80–300 B): `max_world = 2048`
  - Large (≥300 B): `max_world = 4096`
- Effective world size is therefore capped at `tp * 256` (since `pp*dp <= 256`); `max_world` only matters insofar as it gates the largest `tp`.
- Dtype: bf16. ZeRO stage: 1. Sequence parallel: enabled.
- Memory reserved (gmi_error): 6 GB.
- Recompute types searched: `no_recompute`, `full_block`, `selective_recompute`.

## Search space

| Parameter | Candidates |
|---|---|
| `tp` | 1, 2, 4, 8 (filtered by `head_num % tp == 0` and `kv_head_num % tp == 0`) |
| `pp` | 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32 (filtered by `pp <= layer_num` and valid last-stage size) |
| `ep` (MoE) | 1, 2, 4, 8, 16, 32, 64 (filtered by `expert_num % ep == 0`) |
| `ep` (dense) | 1 |
| `dp` | 1, 2, 4, 8, 16, 32, 64, 128, 256 (filtered by `pp*dp <= 256` and `dp \| 256`) |

## Caveats

- **`gbs=256` is below the typical training regime.** Real LLM training runs use much larger global batches (1k–4k) for token-efficiency reasons. These numbers are useful for relative comparisons (e.g. RL rollouts or fine-tuning) but absolute MFU should be read as a lower bound for these models.
- **MLA + TP support unchanged.** Megatron-style layout: down-projs are `disable_tensor_parallel=True` (replicated latent), up-projs use `input_is_full_seq=True` to skip the SP input all-gather, `linear_out_proj` is `LinearRow`. `deepseek-r1` picks `tp=2`, `kimi-k2` picks `tp=2` here.
- **H100 system config is spec-based, not calibrated.** The file warns of 10–20% error on iter time vs. a calibrated config. MFU absolute values should be read as relative comparisons, not ground truth.
- **MoE modeling uses megatron-style capacity padding** (`moe_pad_expert_input_to_capacity=True`, `capacity=1`, `dispatch_probs=True`).
- **`tp` is powers of two; `pp` is no longer restricted to powers of two.** The `pp` set is now `1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32`, chosen to cover exact divisors of common `layer_num` values across the model zoo: 36 (gpt-oss-120b: 3, 6, 12), 60 (gemma4-31b: 3, 5, 6, 10, 12, 20), 80 (llama3-70b, ling-1t: 5, 10, 20). `pp` is still not strictly limited to divisors of `layer_num` — non-divisor `pp` is accepted whenever the last pipeline stage stays non-empty, which is what makes 61-layer MLA models (deepseek-r1, kimi-k2) searchable at all (61 is prime). Skipped from the candidate list: 7, 11, 14, 22, 23, 28 (each helps only one model and the uneven-split path can substitute) and `pp >= 64` (forces `dp <= 4` under `pp*dp <= 256` and leaves ~1–2 layers/stage, which the bubble eats).
- **`dp` is restricted to divisors of 256** under `--exact-gbs`. With `gbs=256` exact, `dp ∈ {1,2,4,8,16,32,64,128,256}`; non-power-of-two `dp` is unreachable. The original search included `dp=1024,2048,4096` which are now filtered out for being incompatible with the exact-256 constraint.

## Results

Per-schedule winner for each model: highest-MFU `(tp, pp, ep, dp, mbn, recompute)` combination that passes the schedule-aware memory fit gate. Strategy JSONs are at `configs/strategy/<model>_optimal_mfu_<schedule>.json`. Rows are sorted by parameter count.

### Schedule: `1f1b`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.5323 | 16 | 2 | 8 | 1 | 1 | 256 | no |
| `gemma3-27b` | 30 | 0.5906 | 24 | 2 | 6 | 1 | 2 | 128 | no |
| `olmo2-32b` | 34 | 0.5908 | 20 | 2 | 5 | 1 | 2 | 128 | no |
| `llama3-70b` | 79 | 0.5463 | 32 | 4 | 4 | 1 | 2 | 128 | no |
| `glm-4.5-air` | 106 | 0.2742 | 48 | 2 | 12 | 2 | 2 | 128 | no |
| `gpt-oss-120b` | 116 | 0.2880 | 64 | 2 | 8 | 4 | 4 | 64 | no |
| `mixtral-8x22b` | 144 | 0.4182 | 96 | 4 | 12 | 2 | 2 | 128 | no |
| `mistral-large` | 147 | 0.5497 | 64 | 4 | 8 | 1 | 2 | 128 | no |
| `minimax-m2` | 228 | 0.2083 | 128 | 2 | 16 | 4 | 4 | 64 | no |
| `qwen3-235b-a22b` | 234 | 0.2242 | 128 | 2 | 32 | 2 | 2 | 128 | full_block (1) |
| `glm-4.5` | 358 | 0.1833 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.4542 | 256 | 8 | 32 | 1 | 1 | 256 | no |
| `qwen3-coder-480b-a35b` | 478 | 0.1136 | 256 | 2 | 16 | 8 | 8 | 32 | selective (4: mlp+mlp_rms) |
| `deepseek-r1` | 701 | 0.1073 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0493 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0531 | 384 | 2 | 12 | 16 | 16 | 16 | full_block (7) |

### Schedule: `gpipe`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.4522 | 8 | 8 | 1 | 1 | 1 | 256 | no |
| `gemma3-27b` | 30 | 0.4896 | 16 | 4 | 1 | 1 | 4 | 64 | no |
| `olmo2-32b` | 34 | 0.4670 | 8 | 8 | 1 | 1 | 1 | 256 | full_block (9) |
| `llama3-70b` | 79 | 0.3596 | 96 | 4 | 6 | 1 | 4 | 64 | full_block (12) |
| `glm-4.5-air` | 106 | 0.1324 | 128 | 2 | 8 | 8 | 8 | 32 | full_block (3) |
| `gpt-oss-120b` | 116 | 0.1725 | 64 | 1 | 1 | 64 | 64 | 4 | no |
| `mixtral-8x22b` | 144 | 0.3251 | 48 | 1 | 12 | 4 | 4 | 64 | full_block (5) |
| `mistral-large` | 147 | 0.4081 | 80 | 4 | 10 | 1 | 2 | 128 | full_block (9) |
| `minimax-m2` | 228 | 0.0794 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (2) |
| `qwen3-235b-a22b` | 234 | 0.1140 | 160 | 2 | 10 | 8 | 8 | 32 | full_block (8) |
| `glm-4.5` | 358 | 0.1118 | 192 | 2 | 12 | 8 | 8 | 32 | full_block (7) |
| `llama3-405b` | 475 | 0.3089 | 320 | 8 | 10 | 1 | 4 | 64 | full_block (13) |
| `qwen3-coder-480b-a35b` | 478 | 0.1071 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (3) |
| `deepseek-r1` | 701 | 0.1070 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0487 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0527 | 384 | 2 | 12 | 16 | 16 | 16 | full_block (7) |

### Schedule: `interleaved_1f1b`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.5363 | 32 | 2 | 8 | 1 | 2 | 128 | no |
| `gemma3-27b` | 30 | 0.5960 | 24 | 2 | 6 | 1 | 2 | 128 | no |
| `olmo2-32b` | 34 | 0.5899 | 24 | 2 | 6 | 1 | 2 | 128 | no |
| `llama3-70b` | 79 | 0.5471 | 48 | 4 | 12 | 1 | 1 | 256 | no |
| `glm-4.5-air` | 106 | 0.2813 | 64 | 2 | 16 | 2 | 2 | 128 | no |
| `gpt-oss-120b` | 116 | 0.3152 | 64 | 1 | 8 | 8 | 8 | 32 | no |
| `mixtral-8x22b` | 144 | 0.4056 | 96 | 4 | 12 | 2 | 2 | 128 | full_block (1) |
| `mistral-large` | 147 | 0.5395 | 80 | 4 | 10 | 1 | 2 | 128 | full_block (1) |
| `minimax-m2` | 228 | 0.2191 | 128 | 2 | 16 | 4 | 4 | 64 | selective (4: attn+mla_rms) |
| `qwen3-235b-a22b` | 234 | 0.2334 | 128 | 2 | 32 | 2 | 2 | 128 | selective (3: attn+mla_rms+mlp+mlp_rms) |
| `glm-4.5` | 358 | 0.2016 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.4563 | 256 | 8 | 32 | 1 | 1 | 256 | selective (4: attn+mla_rms) |
| `qwen3-coder-480b-a35b` | 478 | 0.1211 | 256 | 2 | 16 | 8 | 8 | 32 | selective (4: attn+mla_rms+mlp+mlp_rms) |
| `deepseek-r1` | 701 | 0.1168 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0569 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0609 | 512 | 2 | 16 | 16 | 16 | 16 | full_block (3) |

### Schedule: `zb_h1`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.5397 | 16 | 2 | 8 | 1 | 1 | 256 | no |
| `gemma3-27b` | 30 | 0.6031 | 24 | 2 | 6 | 1 | 2 | 128 | no |
| `olmo2-32b` | 34 | 0.6010 | 20 | 2 | 5 | 1 | 2 | 128 | no |
| `llama3-70b` | 79 | 0.5528 | 48 | 4 | 12 | 1 | 1 | 256 | no |
| `glm-4.5-air` | 106 | 0.2763 | 64 | 2 | 16 | 2 | 2 | 128 | no |
| `gpt-oss-120b` | 116 | 0.3024 | 64 | 1 | 8 | 8 | 8 | 32 | no |
| `mixtral-8x22b` | 144 | 0.4422 | 96 | 2 | 12 | 4 | 4 | 64 | no |
| `mistral-large` | 147 | 0.5662 | 64 | 4 | 8 | 1 | 2 | 128 | no |
| `minimax-m2` | 228 | 0.2214 | 128 | 2 | 16 | 4 | 4 | 64 | no |
| `qwen3-235b-a22b` | 234 | 0.2379 | 128 | 2 | 32 | 2 | 2 | 128 | full_block (1) |
| `glm-4.5` | 358 | 0.1922 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.4829 | 256 | 8 | 32 | 1 | 1 | 256 | no |
| `qwen3-coder-480b-a35b` | 478 | 0.1204 | 256 | 2 | 16 | 8 | 8 | 32 | selective (4: mlp+mlp_rms) |
| `deepseek-r1` | 701 | 0.1123 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0516 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0551 | 384 | 2 | 12 | 16 | 16 | 16 | full_block (7) |

### Schedule: `zb_h2`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.5257 | 64 | 2 | 8 | 1 | 4 | 64 | no |
| `gemma3-27b` | 30 | 0.5947 | 48 | 2 | 6 | 1 | 4 | 64 | no |
| `olmo2-32b` | 34 | 0.5682 | 48 | 2 | 6 | 1 | 4 | 64 | full_block (1) |
| `llama3-70b` | 79 | 0.5610 | 48 | 4 | 12 | 1 | 1 | 256 | no |
| `glm-4.5-air` | 106 | 0.2579 | 96 | 2 | 12 | 4 | 4 | 64 | no |
| `gpt-oss-120b` | 116 | 0.3132 | 64 | 2 | 8 | 4 | 4 | 64 | no |
| `mixtral-8x22b` | 144 | 0.3909 | 96 | 4 | 12 | 2 | 2 | 128 | full_block (2) |
| `mistral-large` | 147 | 0.5165 | 48 | 8 | 6 | 1 | 1 | 256 | no |
| `minimax-m2` | 228 | 0.1963 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (2) |
| `qwen3-235b-a22b` | 234 | 0.2136 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (4) |
| `glm-4.5` | 358 | 0.1973 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.4111 | 192 | 8 | 12 | 1 | 2 | 128 | full_block (8) |
| `qwen3-coder-480b-a35b` | 478 | 0.1179 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (3) |
| `deepseek-r1` | 701 | 0.1150 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0520 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0556 | 384 | 2 | 12 | 16 | 16 | 16 | full_block (7) |

### Schedule: `zb_v`

| Model | params (B) | MFU | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.5445 | 16 | 2 | 8 | 1 | 1 | 256 | no |
| `gemma3-27b` | 30 | 0.6128 | 12 | 2 | 6 | 1 | 1 | 256 | no |
| `olmo2-32b` | 34 | 0.6056 | 20 | 2 | 5 | 1 | 2 | 128 | no |
| `llama3-70b` | 79 | 0.5748 | 96 | 2 | 12 | 1 | 4 | 64 | no |
| `glm-4.5-air` | 106 | 0.2872 | 64 | 2 | 16 | 2 | 2 | 128 | no |
| `gpt-oss-120b` | 116 | 0.3329 | 40 | 1 | 5 | 8 | 8 | 32 | full_block (1) |
| `mixtral-8x22b` | 144 | 0.4618 | 96 | 2 | 12 | 4 | 4 | 64 | no |
| `mistral-large` | 147 | 0.5734 | 64 | 4 | 8 | 1 | 2 | 128 | no |
| `minimax-m2` | 228 | 0.2376 | 128 | 2 | 16 | 4 | 4 | 64 | no |
| `qwen3-235b-a22b` | 234 | 0.2605 | 128 | 2 | 32 | 2 | 2 | 128 | full_block (1) |
| `glm-4.5` | 358 | 0.2028 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.4928 | 256 | 8 | 32 | 1 | 1 | 256 | no |
| `qwen3-coder-480b-a35b` | 478 | 0.2251 | 128 | 1 | 16 | 8 | 8 | 32 | full_block (4) |
| `deepseek-r1` | 701 | 0.1178 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0543 | 512 | 2 | 8 | 32 | 32 | 8 | selective (8: attn+mla_rms+mlp+mlp_rms) |
| `ling-1t` | 1054 | 0.0573 | 384 | 2 | 12 | 16 | 16 | 16 | full_block (7) |

