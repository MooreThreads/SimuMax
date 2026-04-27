# Optimal Fitting Strategies per Model (H100 NVLink, gbs=256)

Results from `examples/search_fit_strategy.py` with `--gbs 256 --exact-gbs`. For each model in `configs/models/`, the script enumerates `(tp, pp, ep, dp)` combinations under the constraint `world_size = tp * pp * dp`, calls `PerfLLM.search_best_parallel_strategy_with_recompute` per combo, and keeps the highest-MFU combination that fits in device memory. The winning strategy for each model is saved as `configs/strategy/<model_name>_optimal_mfu.json`.

## Results

Sorted by parameter count (smallest first). MFU is the headline metric; recompute notation: `fb(N)` = full-block on N layers, `sel(N)[...]` = selective on N layers with the listed components, `—` = no recompute. `last` is the layer count in the final pipeline stage when `layer_num % pp != 0`; `even` means uniform split.

| Model | Params (B) | Layers | MFU | World | tp | pp | ep | dp | mbn | Recompute | Last stage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| gemma4-31b | 29 | 60 | **0.5323** | 16 | 2 | 8 | 1 | 1 | 256 | — | 4 |
| gemma3-27b | 30 | 62 | **0.5906** | 24 | 2 | 6 | 1 | 2 | 128 | — | 7 |
| olmo2-32b | 34 | 64 | **0.5908** | 20 | 2 | 5 | 1 | 2 | 128 | — | 12 |
| llama3-70b | 79 | 80 | **0.5463** | 32 | 4 | 4 | 1 | 2 | 128 | — | even |
| glm-4.5-air | 106 | 46 | **0.2742** | 48 | 2 | 12 | 2 | 2 | 128 | — | 2 |
| gpt-oss-120b | 116 | 36 | **0.2880** | 64 | 2 | 8 | 4 | 4 | 64 | — | 1 |
| mixtral-8x22b | 144 | 56 | **0.4182** | 96 | 4 | 12 | 2 | 2 | 128 | — | 1 |
| mistral-large | 147 | 88 | **0.5497** | 64 | 4 | 8 | 1 | 2 | 128 | — | even |
| minimax-m2 | 228 | 62 | **0.2083** | 128 | 2 | 16 | 4 | 4 | 64 | — | 2 |
| qwen3-235b-a22b | 234 | 94 | **0.2242** | 128 | 2 | 32 | 2 | 2 | 128 | fb(1) | 1 |
| glm-4.5 | 358 | 92 | **0.1833** | 128 | 2 | 16 | 4 | 4 | 64 | fb(6) | 2 |
| llama3-405b | 475 | 128 | **0.4542** | 256 | 8 | 32 | 1 | 1 | 256 | — | even |
| qwen3-coder-480b-a35b | 478 | 62 | **0.1136** | 256 | 2 | 16 | 8 | 8 | 32 | sel(4)[mlp,mlp_rms] | 2 |
| deepseek-r1 | 701 | 61 | **0.1073** | 256 | 2 | 16 | 8 | 8 | 32 | fb(4) | 1 |
| kimi-k2 | 1045 | 61 | **0.0493** | 512 | 2 | 8 | 32 | 32 | 8 | sel(8)[attn,mla_rms,mlp,mlp_rms] | 5 |
| ling-1t | 1054 | 80 | **0.0531** | 384 | 2 | 12 | 16 | 16 | 16 | fb(7) | 3 |

### Headline observations

- **Non-power-of-two `pp` is doing real work.** 5 of 16 winners pick a `pp` outside `{1,2,4,8,16,32}` and would have been unreachable in the previous run: olmo2-32b (`pp=5`), gemma3-27b (`pp=6`), glm-4.5-air, mixtral-8x22b, and ling-1t (all `pp=12`).
- **Uneven pipeline splits are the norm, not the exception.** 13 of 16 winners pick a `pp` that does not divide `layer_num`, relying on the script's "non-empty last stage" path (e.g. olmo2-32b 64/5 → `[13,13,13,13,12]`, deepseek-r1 61/16 → 15×4 + 1, qwen3-235b-a22b 94/32 → 31×3 + 1). Only llama3-70b, mistral-large, and llama3-405b land on a `pp` that exactly divides `layer_num`.
- **Dense small models cluster around `tp ∈ {2,4}`, `pp ∈ {4..8}`, no recompute.** olmo2-32b, gemma3-27b, llama3-70b, mistral-large all reach MFU ≥ 0.55 without recompute.
- **MoE models pay a heavy MFU tax under `gbs=256`.** kimi-k2 (0.049), ling-1t (0.053), deepseek-r1 (0.107), qwen3-coder-480b-a35b (0.114) — all force large `ep` (8–32) and the resulting communication+capacity-padding overhead dominates. This is consistent with the doc's caveat that gbs=256 is below the typical training regime; expect these numbers to climb sharply at gbs=2k–4k.
- **`pp=32` only makes sense at very large param counts.** llama3-405b (128 layers, even split) and qwen3-235b-a22b (94 layers, last=1) are the only adopters. For others, the per-stage layer count drops too low and the bubble eats the gain.

### Search effort

Per-model `tried` (combos enumerated) and `fit` (combos that fit in memory):

| Model | Tried | Fit | Fit % |
|---|---:|---:|---:|
| deepseek-r1 | 844 | 59 | 7.0% |
| olmo2-32b | 242 | 200 | 82.6% |
| mistral-large | 216 | 80 | 37.0% |
| gemma3-27b | 208 | 167 | 80.3% |
| kimi-k2 | 844 | 29 | 3.4% |
| mixtral-8x22b | 728 | 228 | 31.3% |
| llama3-405b | 272 | 38 | 14.0% |
| minimax-m2 | 844 | 252 | 29.9% |
| glm-4.5 | 936 | 167 | 17.8% |
| gemma4-31b | 242 | 196 | 81.0% |
| qwen3-coder-480b-a35b | 816 | 90 | 11.0% |
| gpt-oss-120b | 844 | 403 | 47.7% |
| glm-4.5-air | 964 | 514 | 53.3% |
| llama3-70b | 261 | 160 | 61.3% |
| qwen3-235b-a22b | 753 | 230 | 30.5% |
| ling-1t | 1004 | 39 | 3.9% |

Total: ≈10.8k combos enumerated, ≈2.85k fit, run wall time ≈55 min.

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
