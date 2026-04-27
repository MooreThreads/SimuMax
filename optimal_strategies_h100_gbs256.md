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
