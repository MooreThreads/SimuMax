# Optimal Fitting Strategies per Model (H100 NVLink, gbs=256)

Results from `examples/search_fit_strategy.py` with `--gbs 256 --exact-gbs`. For each model in `configs/models/`, the script enumerates `(tp, pp, ep, dp)` combinations under the constraint `world_size = tp * pp * dp`, calls `PerfLLM.search_best_parallel_strategy_with_recompute` per combo, and keeps the highest-MFU combination that fits in device memory. The winning strategy for each model is saved as `configs/strategy_gbs256/<model_name>_optimal.json`.

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
| `pp` | 1, 2, 4, 8, 16, 32 (filtered by `pp <= layer_num` and valid last-stage size) |
| `ep` (MoE) | 1, 2, 4, 8, 16, 32, 64 (filtered by `expert_num % ep == 0`) |
| `ep` (dense) | 1 |
| `dp` | 1, 2, 4, 8, 16, 32, 64, 128, 256 (filtered by `pp*dp <= 256` and `dp \| 256`) |

## Results

### Dense

| Model | Params | tp | pp | ep | dp | world | mbn | gbs | MFU | Recompute | Combos tried | Fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| olmo2-32b | 34B | 2 | 2 | 1 | 4 | 16 | 64 | 256 | 0.592 | full_block × 27 | 150 | 117 |
| gemma3-27b | 30B | 1 | 4 | 1 | 4 | 16 | 64 | 256 | 0.582 | full_block × 15 | 135 | 102 |
| mistral-large | 147B | 4 | 8 | 1 | 1 | 32 | 256 | 256 | 0.574 | full_block × 9 | 120 | 34 |
| llama3-70b | 79B | 4 | 4 | 1 | 2 | 32 | 128 | 256 | 0.546 | No Recompute | 135 | 69 |
| gemma4-31b | 29B | 1 | 8 | 1 | 2 | 16 | 128 | 256 | 0.544 | full_block × 7 | 116 | 81 |
| llama3-405b | 475B | 8 | 16 | 1 | 1 | 128 | 256 | 256 | 0.517 | full_block × 5 | 156 | 22 |

### MoE (and MoE + MLA)

| Model | Type | Params | tp | pp | ep | dp | world | mbn | gbs | MFU | Recompute | Combos tried | Fit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| mixtral-8x22b | moe | 144B | 2 | 8 | 4 | 4 | 64 | 64 | 256 | 0.448 | full_block × 3 | 384 | 83 |
| gpt-oss-120b | moe | 116B | 2 | 8 | 4 | 4 | 64 | 64 | 256 | 0.288 | No Recompute | 504 | 209 |
| glm-4.5-air | moe | 106B | 1 | 16 | 2 | 2 | 32 | 128 | 256 | 0.283 | full_block × 3 | 564 | 261 |
| qwen3-235b-a22b | moe | 234B | 1 | 16 | 4 | 4 | 64 | 64 | 256 | 0.260 | full_block × 6 | 453 | 135 |
| glm-4.5 | moe | 358B | 2 | 16 | 4 | 4 | 128 | 64 | 256 | 0.247 | full_block × 6 | 540 | 82 |
| minimax-m2 | moe | 228B | 1 | 8 | 8 | 8 | 64 | 32 | 256 | 0.237 | full_block × 8 | 564 | 164 |
| deepseek-r1 | moe + MLA | 701B | 2 | 16 | 8 | 8 | 256 | 32 | 256 | 0.124 | full_block × 4 | 564 | 39 |
| qwen3-coder-480b-a35b | moe | 478B | 2 | 16 | 8 | 8 | 256 | 32 | 256 | 0.122 | full_block × 2 | 540 | 59 |
| ling-1t | moe | 1054B | 2 | 4 | 64 | 64 | 512 | 4 | 256 | 0.061 | full_block × 15 | 564 | 23 |
| kimi-k2 | moe + MLA | 1045B | 2 | 8 | 32 | 32 | 512 | 8 | 256 | 0.052 | full_block × 5 | 564 | 22 |

## What changed vs. the previous (per-category gbs) table

- **gbs is now uniformly 256** instead of 1024 / 2048 / 4096 by model bucket, and `--exact-gbs` removes the prior `[gbs/2, 2*gbs]` slack. This makes `mbn = 256/dp` a hard relation — no microbatch headroom to "absorb" suboptimal `dp` choices.
- **TP grows on dense large models.** With `mbn` capped at 256 (when `dp=1`) instead of the 1024+ used previously, the simulator now picks `tp=8, pp=16, dp=1` for `llama3-405b` and `tp=4, pp=8, dp=1` for `mistral-large`. The previous run had these at `tp=4` and `tp=2` respectively. With less DP/microbatch amortization available, leaning harder on TP becomes the right tradeoff.
- **DP collapses for dense models** to `dp ∈ {1, 2, 4}`. With `gbs=256`, larger `dp` would force `mbn < pp`, breaking pipeline validity. The DP search axis is essentially neutralized at this batch size.
- **MoE winners shrink to `(ep, dp)` pairs in the 4–8 range** for most models (e.g. `mixtral-8x22b`, `gpt-oss-120b`, `glm-4.5`, `qwen3-235b-a22b`). Only the trillion-param MLA/MoE models (`kimi-k2`, `ling-1t`) end up at `ep=32`/`64` to fit in memory.
- **Dense MFU drops modestly** (−1.5% to −5.5%): 0.59–0.61 → 0.52–0.59. Pipeline bubble share grows because there are far fewer microbatches per step (e.g. `mistral-large`: 512 → 256 mbn, `llama3-405b`: 1024 → 256).
- **MoE MFU drops more sharply** (−4.5% to −22%). The hardest hit are `deepseek-r1` (0.346 → 0.124), `qwen3-coder-480b-a35b` (0.314 → 0.122), `ling-1t` (0.225 → 0.061), and `kimi-k2` (0.198 → 0.052). At gbs=256 the trillion-param MoEs can't keep their expert-parallel groups fed: `mbn` ends up at 4–8, so all-to-all overheads dominate every step.
- **Some `ep=1` MoE winners are gone.** With the prior gbs, `qwen3-235b-a22b` and `glm-4.5-air` won at `ep=1`. At gbs=256 they shift to `ep=4` and `ep=2` respectively — replicating all experts on every rank no longer pays off when each step has so few microbatches.
- **Combos tried is roughly the same**, but **fit counts swing widely**: dense small models retain most fits (olmo2-32b: 130→117), while large dense and MLA models lose most of their search space (llama3-405b: 32→22, deepseek-r1: 142→39, kimi-k2: 107→22, ling-1t: 103→23).

## Caveats

- **`gbs=256` is below the typical training regime.** Real LLM training runs use much larger global batches (1k–4k) for token-efficiency reasons. These numbers are useful for relative comparisons (e.g. RL rollouts or fine-tuning) but absolute MFU should be read as a lower bound for these models.
- **MLA + TP support unchanged.** Megatron-style layout: down-projs are `disable_tensor_parallel=True` (replicated latent), up-projs use `input_is_full_seq=True` to skip the SP input all-gather, `linear_out_proj` is `LinearRow`. `deepseek-r1` picks `tp=2`, `kimi-k2` picks `tp=2` here.
- **H100 system config is spec-based, not calibrated.** The file warns of 10–20% error on iter time vs. a calibrated config. MFU absolute values should be read as relative comparisons, not ground truth.
- **MoE modeling uses megatron-style capacity padding** (`moe_pad_expert_input_to_capacity=True`, `capacity=1`, `dispatch_probs=True`).
- **Search is coarse.** `tp` and `pp` are powers of two; `pp` isn't restricted to exact divisors of `layer_num` but does enforce a non-empty last stage. Finer granularity (e.g., `pp=3,5,6`) could yield better fits — particularly for 61-layer MLA models where 61 is prime.
- **`dp` is restricted to divisors of 256** under `--exact-gbs`. With `gbs=256` exact, `dp ∈ {1,2,4,8,16,32,64,128,256}`; non-power-of-two `dp` is unreachable. The original search included `dp=1024,2048,4096` which are now filtered out for being incompatible with the exact-256 constraint.
