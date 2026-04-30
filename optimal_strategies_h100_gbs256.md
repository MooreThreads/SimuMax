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


## Disturbed-fit results (disturbance: `both`, 20 seeds)

Generated by the two-phase disturbed search in `examples/search_fit_strategy.py
--disturbance both` (see `Disturbed_Search_Plan.md` and `DIARY.md` §"Disturbed
(two-phase robust search)"). Same `(--gbs 256 --exact-gbs --system h100_nvlink)`
configuration as the nominal sweep at the top of this file. Disturbance config:
`configs/disturbance/both.json` (variable seq_len + per-op noise + stage/op
slowdowns). Search defaults: `--num-seeds 20 --candidate-top-k 30
--candidate-mfu-window 0.10 --seed-base 0`. Strict memory filter — a candidate
is kept only if peak memory ≤ accelerator budget on **every** seed; the saved
strategy uses `recompute_layer_num = max-across-seeds` for `full_block` winners
(pessimistic-seed level applied at every step). Per-cell artifacts:
`configs/strategy/<model>_optimal_mfu_<schedule>_both.json` (winner) and
`<…>_audit.json` (per-candidate stats, nominal-best, `git_rev`).

### Caveats specific to disturbed mode

- **Robust ≠ typical.** `mfu_mean` is the mean across 20 seeds at the
  pessimistic recompute level — often dramatically below the nominal MFU at the
  same `(model, schedule)`, because the saved level is the worst-seed level
  applied to every step. Compare against `mfu_std` and `peak_max` to gauge
  variability, not against the nominal table alone.
- **No-fit cells** mean *no* `(combo, recompute_family)` candidate fit memory
  on all 20 seeds within the Phase A short-list (top-30 with MFU window 10%).
  These cells have an audit JSON but no strategy JSON. Possible mitigations:
  bump `--candidate-top-k`, widen `--candidate-mfu-window`, or relax memory
  budget — none applied here.
- **Wide std flags `full_block` rerunning the level binary search per seed.**
  When the std is conspicuously larger (e.g. `gpipe` for `gpt-oss-120b`,
  `gemma3-27b`, `gemma4-31b`, or any cell on `ling-1t`), the per-seed evaluation
  found different levels feasible per seed; the saved level is the maximum.
- **Reproducibility ties to commit.** Disturbance sampling depends on the
  current SimuMax tree state; each audit JSON records `git_rev`. Same seed at
  a different commit may give different draws.

### Fit / no-fit summary across schedules

Out of 16 models per schedule, how many cells produced a winner under the
strict every-seed memory filter:

| schedule           | fit | no-fit |
|--------------------|----:|-------:|
| `interleaved_1f1b` |  16 |      0 |
| `1f1b`             |  14 |      2 |
| `zb_v`             |  14 |      2 |
| `zb_h1`            |  13 |      3 |
| `gpipe`            |  10 |      6 |
| `zb_h2`            |   9 |      7 |
| **total**          |  76 |     20 |

`interleaved_1f1b` is the most disturbance-robust schedule — every model finds
a candidate that fits across all 20 seeds. `zb_h2` is the most fragile (7/16
no-fit), driven mainly by the largest MoE models (`minimax-m2`,
`qwen3-235b-a22b`, `glm-4.5`, `qwen3-coder-480b-a35b`, `deepseek-r1`,
`ling-1t`) where the per-stage activation footprint plus the larger zb-h2
buffer leaves no surviving combo. `gpipe` is the second-most fragile for the
same reason. `1f1b` and `zb_h1` lose just `minimax-m2` and `deepseek-r1` (and,
for `zb_h1`, `ling-1t`).

### Schedule: `1f1b`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.2387 | 0.0276 | 74.0 | 16 | 4 | 2 | 1 | 2 | 128 | full_block (14) |
| `gemma3-27b` | 30 | 0.2590 | 0.0296 | 73.9 | 8 | 4 | 2 | 1 | 1 | 256 | full_block (23) |
| `olmo2-32b` | 34 | 0.2664 | 0.0305 | 73.9 | 16 | 4 | 2 | 1 | 2 | 128 | full_block (17) |
| `llama3-70b` | 79 | 0.2216 | 0.0177 | 74.0 | 24 | 8 | 3 | 1 | 1 | 256 | full_block (12) |
| `glm-4.5-air` | 106 | 0.0771 | 0.0072 | 73.9 | 48 | 2 | 6 | 4 | 4 | 64 | full_block (5) |
| `gpt-oss-120b` | 116 | 0.1100 | 0.0093 | 71.8 | 64 | 2 | 8 | 4 | 4 | 64 | no |
| `mixtral-8x22b` | 144 | 0.1479 | 0.0119 | 73.3 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (6) |
| `mistral-large` | 147 | 0.1927 | 0.0092 | 73.9 | 48 | 8 | 6 | 1 | 1 | 256 | full_block (5) |
| `minimax-m2` | 228 | — | — | — | — | — | — | — | — | — | no-fit |
| `qwen3-235b-a22b` | 234 | 0.0642 | 0.0060 | 72.9 | 96 | 2 | 12 | 4 | 4 | 64 | full_block (7) |
| `glm-4.5` | 358 | 0.0581 | 0.0056 | 73.5 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.1526 | 0.0098 | 73.4 | 256 | 8 | 16 | 1 | 2 | 128 | full_block (5) |
| `qwen3-coder-480b-a35b` | 478 | 0.0575 | 0.0069 | 73.5 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (3) |
| `deepseek-r1` | 701 | — | — | — | — | — | — | — | — | — | no-fit |
| `kimi-k2` | 1045 | 0.0324 | 0.0070 | 73.8 | 512 | 2 | 8 | 32 | 32 | 8 | full_block (7) |
| `ling-1t` | 1054 | 0.0497 | 0.0224 | 73.8 | 512 | 2 | 4 | 64 | 64 | 4 | full_block (19) |

### Schedule: `gpipe`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.3080 | 0.0568 | 72.4 | 16 | 8 | 1 | 1 | 2 | 128 | no |
| `gemma3-27b` | 30 | 0.3136 | 0.0615 | 73.8 | 8 | 8 | 1 | 1 | 1 | 256 | full_block (34) |
| `olmo2-32b` | 34 | 0.3425 | 0.0631 | 73.8 | 16 | 8 | 1 | 1 | 2 | 128 | full_block (2) |
| `llama3-70b` | 79 | 0.2392 | 0.0286 | 73.7 | 192 | 8 | 12 | 1 | 2 | 128 | full_block (5) |
| `glm-4.5-air` | 106 | 0.0788 | 0.0068 | 73.8 | 160 | 2 | 10 | 8 | 8 | 32 | full_block (3) |
| `gpt-oss-120b` | 116 | 0.1787 | 0.1261 | 73.8 | 64 | 1 | 1 | 64 | 64 | 4 | full_block (32) |
| `mixtral-8x22b` | 144 | 0.2140 | 0.0228 | 59.3 | 80 | 2 | 10 | 4 | 4 | 64 | full_block (6) |
| `mistral-large` | 147 | — | — | — | — | — | — | — | — | — | no-fit |
| `minimax-m2` | 228 | — | — | — | — | — | — | — | — | — | no-fit |
| `qwen3-235b-a22b` | 234 | — | — | — | — | — | — | — | — | — | no-fit |
| `glm-4.5` | 358 | 0.0755 | 0.0065 | 73.9 | 160 | 2 | 10 | 8 | 8 | 32 | full_block (10) |
| `llama3-405b` | 475 | 0.2186 | 0.0202 | 73.5 | 384 | 8 | 12 | 1 | 4 | 64 | full_block (11) |
| `qwen3-coder-480b-a35b` | 478 | — | — | — | — | — | — | — | — | — | no-fit |
| `deepseek-r1` | 701 | — | — | — | — | — | — | — | — | — | no-fit |
| `kimi-k2` | 1045 | 0.0329 | 0.0049 | 73.8 | 512 | 2 | 8 | 32 | 32 | 8 | full_block (7) |
| `ling-1t` | 1054 | — | — | — | — | — | — | — | — | — | no-fit |

### Schedule: `zb_h1`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.2486 | 0.0293 | 74.0 | 16 | 4 | 2 | 1 | 2 | 128 | full_block (20) |
| `gemma3-27b` | 30 | 0.2846 | 0.0141 | 73.7 | 20 | 4 | 5 | 1 | 1 | 256 | no |
| `olmo2-32b` | 34 | 0.2900 | 0.0341 | 74.0 | 16 | 4 | 2 | 1 | 2 | 128 | full_block (19) |
| `llama3-70b` | 79 | 0.2571 | 0.0224 | 74.0 | 24 | 8 | 3 | 1 | 1 | 256 | full_block (12) |
| `glm-4.5-air` | 106 | 0.0844 | 0.0086 | 73.9 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (3) |
| `gpt-oss-120b` | 116 | 0.1243 | 0.0104 | 71.8 | 64 | 2 | 8 | 4 | 4 | 64 | no |
| `mixtral-8x22b` | 144 | 0.1733 | 0.0141 | 73.5 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (6) |
| `mistral-large` | 147 | 0.2109 | 0.0137 | 73.6 | 80 | 4 | 10 | 1 | 2 | 128 | full_block (5) |
| `minimax-m2` | 228 | — | — | — | — | — | — | — | — | — | no-fit |
| `qwen3-235b-a22b` | 234 | 0.0682 | 0.0070 | 73.5 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (4) |
| `glm-4.5` | 358 | 0.0638 | 0.0063 | 73.7 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.1851 | 0.0124 | 73.9 | 256 | 8 | 16 | 1 | 2 | 128 | full_block (6) |
| `qwen3-coder-480b-a35b` | 478 | 0.0635 | 0.0076 | 73.5 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (3) |
| `deepseek-r1` | 701 | — | — | — | — | — | — | — | — | — | no-fit |
| `kimi-k2` | 1045 | 0.0336 | 0.0070 | 73.8 | 512 | 2 | 8 | 32 | 32 | 8 | full_block (7) |
| `ling-1t` | 1054 | — | — | — | — | — | — | — | — | — | no-fit |

### Schedule: `zb_h2`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.2925 | 0.0200 | 73.4 | 32 | 4 | 8 | 1 | 1 | 256 | full_block (1) |
| `gemma3-27b` | 30 | 0.3156 | 0.0352 | 73.4 | 40 | 2 | 5 | 1 | 4 | 64 | full_block (10) |
| `olmo2-32b` | 34 | 0.3459 | 0.0399 | 73.8 | 40 | 4 | 5 | 1 | 2 | 128 | full_block (4) |
| `llama3-70b` | 79 | 0.2979 | 0.0267 | 73.9 | 48 | 4 | 6 | 1 | 2 | 128 | full_block (10) |
| `glm-4.5-air` | 106 | 0.1079 | 0.0103 | 73.8 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (4) |
| `gpt-oss-120b` | 116 | — | — | — | — | — | — | — | — | — | no-fit |
| `mixtral-8x22b` | 144 | 0.2293 | 0.0209 | 73.6 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (7) |
| `mistral-large` | 147 | 0.3183 | 0.0245 | 73.7 | 80 | 8 | 10 | 1 | 1 | 256 | full_block (2) |
| `minimax-m2` | 228 | — | — | — | — | — | — | — | — | — | no-fit |
| `qwen3-235b-a22b` | 234 | — | — | — | — | — | — | — | — | — | no-fit |
| `glm-4.5` | 358 | — | — | — | — | — | — | — | — | — | no-fit |
| `llama3-405b` | 475 | 0.2626 | 0.0185 | 73.9 | 160 | 8 | 10 | 1 | 2 | 128 | full_block (13) |
| `qwen3-coder-480b-a35b` | 478 | — | — | — | — | — | — | — | — | — | no-fit |
| `deepseek-r1` | 701 | — | — | — | — | — | — | — | — | — | no-fit |
| `kimi-k2` | 1045 | 0.0359 | 0.0067 | 73.8 | 512 | 2 | 8 | 32 | 32 | 8 | full_block (7) |
| `ling-1t` | 1054 | — | — | — | — | — | — | — | — | — | no-fit |

### Schedule: `interleaved_1f1b`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.2503 | 0.0238 | 73.9 | 24 | 2 | 3 | 1 | 4 | 64 | full_block (14) |
| `gemma3-27b` | 30 | 0.3136 | 0.0193 | 70.7 | 20 | 4 | 5 | 1 | 1 | 256 | no |
| `olmo2-32b` | 34 | 0.3131 | 0.0327 | 73.8 | 24 | 4 | 3 | 1 | 2 | 128 | full_block (8) |
| `llama3-70b` | 79 | 0.2713 | 0.0261 | 73.9 | 24 | 8 | 3 | 1 | 1 | 256 | full_block (14) |
| `glm-4.5-air` | 106 | 0.1006 | 0.0090 | 73.7 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (3) |
| `gpt-oss-120b` | 116 | 0.1384 | 0.0124 | 73.9 | 64 | 2 | 8 | 4 | 4 | 64 | selective (5: attn+mla_rms) |
| `mixtral-8x22b` | 144 | 0.1963 | 0.0143 | 73.9 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (6) |
| `mistral-large` | 147 | 0.2590 | 0.0107 | 72.2 | 80 | 8 | 10 | 1 | 1 | 256 | no |
| `minimax-m2` | 228 | 0.0909 | 0.0110 | 71.2 | 64 | 1 | 8 | 8 | 8 | 32 | full_block (8) |
| `qwen3-235b-a22b` | 234 | 0.0865 | 0.0086 | 73.9 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (5) |
| `glm-4.5` | 358 | 0.0818 | 0.0083 | 73.5 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.2120 | 0.0129 | 73.4 | 256 | 8 | 16 | 1 | 2 | 128 | full_block (5) |
| `qwen3-coder-480b-a35b` | 478 | 0.0742 | 0.0075 | 72.5 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (3) |
| `deepseek-r1` | 701 | 0.0717 | 0.0074 | 72.6 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | 0.0390 | 0.0072 | 74.0 | 512 | 2 | 8 | 32 | 32 | 8 | full_block (7) |
| `ling-1t` | 1054 | 0.0601 | 0.0235 | 73.9 | 512 | 2 | 4 | 64 | 64 | 4 | full_block (18) |

### Schedule: `zb_v`

| Model | params (B) | mfu_mean | mfu_std | peak_max GB | world | tp | pp | ep | dp | mbn | recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `gemma4-31b` | 29 | 0.2807 | 0.0262 | 73.1 | 64 | 2 | 8 | 1 | 4 | 64 | full_block (1) |
| `gemma3-27b` | 30 | 0.3690 | 0.0398 | 73.4 | 48 | 2 | 6 | 1 | 4 | 64 | full_block (2) |
| `olmo2-32b` | 34 | 0.3500 | 0.0302 | 66.0 | 24 | 4 | 6 | 1 | 1 | 256 | no |
| `llama3-70b` | 79 | 0.3313 | 0.0203 | 72.2 | 48 | 4 | 12 | 1 | 1 | 256 | full_block (1) |
| `glm-4.5-air` | 106 | 0.1061 | 0.0093 | 73.9 | 48 | 2 | 6 | 4 | 4 | 64 | full_block (6) |
| `gpt-oss-120b` | 116 | 0.1672 | 0.0152 | 69.2 | 64 | 2 | 8 | 4 | 4 | 64 | no |
| `mixtral-8x22b` | 144 | 0.2297 | 0.0182 | 73.7 | 64 | 2 | 8 | 4 | 4 | 64 | full_block (5) |
| `mistral-large` | 147 | 0.2895 | 0.0215 | 73.8 | 80 | 4 | 10 | 1 | 2 | 128 | full_block (5) |
| `minimax-m2` | 228 | 0.0819 | 0.0123 | 72.6 | 128 | 1 | 16 | 8 | 8 | 32 | full_block (3) |
| `qwen3-235b-a22b` | 234 | 0.0899 | 0.0094 | 73.9 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (4) |
| `glm-4.5` | 358 | 0.0804 | 0.0083 | 72.3 | 128 | 2 | 16 | 4 | 4 | 64 | full_block (6) |
| `llama3-405b` | 475 | 0.2699 | 0.0189 | 73.5 | 256 | 8 | 16 | 1 | 2 | 128 | full_block (4) |
| `qwen3-coder-480b-a35b` | 478 | — | — | — | — | — | — | — | — | — | no-fit |
| `deepseek-r1` | 701 | 0.0696 | 0.0081 | 71.9 | 256 | 2 | 16 | 8 | 8 | 32 | full_block (4) |
| `kimi-k2` | 1045 | — | — | — | — | — | — | — | — | — | no-fit |
| `ling-1t` | 1054 | 0.0582 | 0.0235 | 73.9 | 512 | 2 | 4 | 64 | 64 | 4 | full_block (18) |

