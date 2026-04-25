"""
For each model in configs/models/, search the highest-MFU (tp, pp, ep, dp, recompute)
combination at a fixed (approximate) global batch size within a max world-size budget.

Framing: world_size = tp * pp * dp; ep is a partition within dp for MoE (requires ep | dp).
micro_batch_num is chosen as close as possible to gbs_target / dp while respecting the
pipeline validity constraint (mbn >= pp). Combos whose effective gbs would drift beyond
GBS_RELAX_FACTOR away from the target are dropped.

Output: configs/strategy/<model_name>_optimal.json, matching the StrategyConfig schema.
"""
import argparse
import json
import math
import os
import re
import sys
from contextlib import contextmanager

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    RELEASE_MODELS,
    RELEASE_STRATEGY,
    get_simu_system_config,
)


SEQ_LEN = 4096
MICRO_BATCH_SIZE = 1
DTYPE = "bf16"
GMI_ERROR = 6  # GB reserved, same default as llm_search.py

GBS_RELAX_FACTOR = 2.0  # accept combos whose effective gbs is within [gbs/f, gbs*f]

# Auto-sizing: pick gbs and max_world per model by rough total-param count.
# Small (<80B) ~ 4M tokens/step, medium (80-300B) ~ 8M, large (>=300B) ~ 16M, at seq=4096.
# max_world tracks gbs so each category has a similar dp/mbn headroom.
GBS_SMALL, MAX_WORLD_SMALL = 1024, 1024
GBS_MEDIUM, MAX_WORLD_MEDIUM = 2048, 2048
GBS_LARGE, MAX_WORLD_LARGE = 4096, 4096
SMALL_THRESHOLD_B = 80.0
LARGE_THRESHOLD_B = 300.0

TP_CANDIDATES = [1, 2, 4, 8]
PP_CANDIDATES = [1, 2, 4, 8, 16, 32]
EP_CANDIDATES_MOE = [1, 2, 4, 8, 16, 32, 64]
EP_CANDIDATES_DENSE = [1]
DP_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RECOMPUTE_TYPES = ["no_recompute", "full_block", "selective_recompute"]


def estimate_params_b(model_cfg: ModelConfig) -> float:
    """Rough total-param count in billions. Used only for picking the batch-size category."""
    h = model_cfg.hidden_size
    L = model_cfg.layer_num
    V = getattr(model_cfg, "vocab_size", 32000) or 32000
    ffn = getattr(model_cfg, "ffn_hidden_size", None) or getattr(model_cfg, "intermediate_size", 4 * h)
    attn = 4 * h * h * L
    if model_cfg.model_type == "moe":
        expert_ffn = getattr(model_cfg, "moe_ffn_hidden_size", None) or getattr(model_cfg, "expert_ffn_hidden_size", ffn)
        expert_num = getattr(model_cfg, "expert_num", 1) or 1
        moe_params = expert_num * 3 * h * expert_ffn * L
        shared_ffn = getattr(model_cfg, "shared_expert_ffn_hidden_size", 0) or 0
        shared_params = 3 * h * shared_ffn * L if shared_ffn else 0
        total = attn + moe_params + shared_params + V * h
    else:
        total = attn + 3 * h * ffn * L + V * h
    return total / 1e9


def auto_gbs(model_cfg: ModelConfig) -> int:
    """Map a model to small/medium/large gbs by rough param count."""
    pb = estimate_params_b(model_cfg)
    if pb < SMALL_THRESHOLD_B:
        return GBS_SMALL
    if pb < LARGE_THRESHOLD_B:
        return GBS_MEDIUM
    return GBS_LARGE


def auto_max_world(model_cfg: ModelConfig) -> int:
    """Max world_size by the same small/medium/large bucket, matching gbs."""
    pb = estimate_params_b(model_cfg)
    if pb < SMALL_THRESHOLD_B:
        return MAX_WORLD_SMALL
    if pb < LARGE_THRESHOLD_B:
        return MAX_WORLD_MEDIUM
    return MAX_WORLD_LARGE


@contextmanager
def suppress_stdout():
    """PerfLLM.search prints a lot; silence it per-combo so progress stays readable."""
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()


def enumerate_combos(model_cfg: ModelConfig, gbs_target: int, max_world: int,
                     relax_factor: float = GBS_RELAX_FACTOR):
    """Yield valid (tp, pp, ep, dp, mbn, gbs_eff) for this model.

    Picks mbn so that mbn*dp is as close as possible to gbs_target, subject to
    mbn >= pp (pipeline validity). Combos where the resulting gbs_eff would fall
    outside [gbs_target/relax_factor, gbs_target*relax_factor] are dropped. Pass
    relax_factor=1.0 to require gbs_eff == gbs_target exactly.
    """
    head_num = model_cfg.head_num
    kv_head_num = model_cfg.kv_head_num
    layer_num = model_cfg.layer_num
    is_moe = model_cfg.model_type == "moe"
    expert_num = getattr(model_cfg, "expert_num", 0) or 0

    tp_list = [tp for tp in TP_CANDIDATES if head_num % tp == 0 and kv_head_num % tp == 0]

    pp_list = []
    for pp in PP_CANDIDATES:
        if pp > layer_num:
            continue
        if pp == 1:
            pp_list.append(pp)
            continue
        layers_per_stage = math.ceil(layer_num / pp)
        last_stage = layer_num - layers_per_stage * (pp - 1)
        if last_stage > 0:
            pp_list.append(pp)

    if is_moe:
        ep_list = [ep for ep in EP_CANDIDATES_MOE if expert_num == 0 or expert_num % ep == 0]
    else:
        ep_list = EP_CANDIDATES_DENSE

    gbs_min = gbs_target / relax_factor
    gbs_max = gbs_target * relax_factor

    for tp in tp_list:
        for pp in pp_list:
            for ep in ep_list:
                for dp in DP_CANDIDATES:
                    world = tp * pp * dp
                    if world > max_world:
                        continue
                    if is_moe and (dp < ep or dp % ep != 0):
                        continue
                    # Closest mbn to target that still feeds the pipeline.
                    mbn_ideal = max(1, round(gbs_target / dp))
                    mbn = max(mbn_ideal, pp)
                    gbs_eff = mbn * dp
                    if gbs_eff < gbs_min or gbs_eff > gbs_max:
                        continue
                    yield tp, pp, ep, dp, mbn, gbs_eff


PARALLELISM_RE = re.compile(
    r"tp(?P<tp>\d+)\.ep(?P<ep>\d+)\.pp(?P<pp>\d+)\.dp(?P<dp>\d+)"
    r"\.etp(?P<etp>\d+)\.edp\d+, world_size:(?P<world>\d+)"
)
MBC_RE = re.compile(r"mbc(\d+)")
MBS_RE = re.compile(r"mbs(\d+)")


def parse_parallelism(parallelism_str: str):
    m = PARALLELISM_RE.search(parallelism_str)
    if not m:
        raise ValueError(f"Could not parse parallelism string: {parallelism_str}")
    tp = int(m.group("tp"))
    ep = int(m.group("ep"))
    pp = int(m.group("pp"))
    etp = int(m.group("etp"))
    world = int(m.group("world"))
    mbn = int(MBC_RE.search(parallelism_str).group(1))
    mbs = int(MBS_RE.search(parallelism_str).group(1))
    return {
        "tp_size": tp,
        "pp_size": pp,
        "ep_size": ep,
        "etp_size": etp,
        "world_size": world,
        "micro_batch_num": mbn,
        "micro_batch_size": mbs,
    }


def parse_recompute_status(status: str):
    """Returns dict with recompute fields, or {} for no-recompute."""
    if status == "No Recompute":
        return {"enable_recompute": False, "recompute_granularity": None, "recompute_layer_num": 0}

    if status.startswith("full_block"):
        m = re.search(r"recompute_layer_num=(\d+)", status)
        return {
            "enable_recompute": True,
            "recompute_granularity": "full_block",
            "recompute_layer_num": int(m.group(1)),
        }

    if status.startswith("selective_recompute"):
        def flag(name):
            m = re.search(rf"{name}=(True|False)", status)
            return m.group(1) == "True" if m else False

        layer_m = re.search(r"recompute_layer_num=(\d+)", status)
        return {
            "enable_recompute": True,
            "recompute_granularity": "selective_recompute",
            "recompute_layer_num": int(layer_m.group(1)),
            "attn_recompute": flag("attn"),
            "mla_rms_recompute": flag("attn_rms"),
            "mlp_recompute": flag("mlp"),
            "mlp_rms_recompute": flag("mlp_rms"),
            "recompute_variance": flag("recompute_variance"),
        }

    raise ValueError(f"Unrecognized recompute_status: {status}")


def build_strategy_json(best: dict) -> dict:
    parallelism = parse_parallelism(best["parallelism"])
    recompute = parse_recompute_status(best["recompute_status"])

    strategy = {
        "seq_len": SEQ_LEN,
        "micro_batch_size": parallelism["micro_batch_size"],
        "micro_batch_num": parallelism["micro_batch_num"],
        "dtype": DTYPE,
        "world_size": parallelism["world_size"],
        "tp_size": parallelism["tp_size"],
        "pp_size": parallelism["pp_size"],
        "ep_size": parallelism["ep_size"],
        "etp_size": parallelism["etp_size"],
        "moe_dispatcher_policy": "all2all",
        "enable_sequence_parallel": True,
        "zero_state": 1,
        "enable_dropout": False,
        "use_fused_norm": True,
        "use_math_sdp": False,
        "use_flash_sdp": True,
        "use_fp32_accum_grad": True,
        "enable_recompute": recompute["enable_recompute"],
        "mem_factor": 0.94,
    }
    if recompute["enable_recompute"]:
        strategy["recompute_granularity"] = recompute["recompute_granularity"]
        strategy["recompute_layer_num"] = recompute["recompute_layer_num"]
        if recompute["recompute_granularity"] == "selective_recompute":
            strategy["attn_recompute"] = recompute["attn_recompute"]
            strategy["mla_rms_recompute"] = recompute["mla_rms_recompute"]
            strategy["mlp_recompute"] = recompute["mlp_recompute"]
            strategy["mlp_rms_recompute"] = recompute["mlp_rms_recompute"]
            strategy["recompute_variance"] = recompute["recompute_variance"]
    return strategy


def make_perf_model(model_config_path: str, system_config_path: str) -> PerfLLM:
    # Any valid StrategyConfig works as a skeleton — search overrides world_size, tp/pp/ep,
    # mbs, mbn, and recompute_* per combo. Pick an existing file from configs/strategy/.
    template_path = RELEASE_STRATEGY["llama3_70b_optimal_mfu"]
    perf = PerfLLM()
    perf.configure(
        strategy_config=StrategyConfig.init_from_config_file(template_path),
        model_config=ModelConfig.init_from_config_file(model_config_path),
        system_config=SystemConfig.init_from_config_file(system_config_path),
    )
    # Align with megatron defaults as done in llm_search.py (affects MoE capacity modeling).
    perf.model_config.moe_pad_expert_input_to_capacity = True
    perf.model_config.capacity = 1
    perf.model_config.padded_vocab_size = True
    perf.model_config.make_vocab_size_divisible_by = 128
    perf.strategy.dispatch_probs = True
    perf.strategy.seq_len = SEQ_LEN
    return perf


def search_for_model(model_name: str, model_config_path: str, system_config_path: str,
                     gbs_target: int, max_world: int, verbose: bool = False,
                     relax_factor: float = GBS_RELAX_FACTOR):
    perf = make_perf_model(model_config_path, system_config_path)
    mcfg = perf.model_config

    best_overall = {}
    best_mfu = -1.0
    tried = 0
    fit = 0

    combos = list(enumerate_combos(mcfg, gbs_target=gbs_target, max_world=max_world,
                                   relax_factor=relax_factor))
    print(f"[{model_name}] {len(combos)} (tp, pp, ep, dp) combos to search "
          f"(gbs≈{gbs_target}, max_world={max_world})")

    for tp, pp, ep, dp, mbn, gbs_eff in combos:
        world_size = tp * pp * dp
        tried += 1

        # Fresh PerfLLM per combo: the search mutates strategy fields (and resets
        # recompute_layer_num on exit), so isolating avoids state leak.
        perf = make_perf_model(model_config_path, system_config_path)
        all_search_result = {}

        def _run():
            return perf.search_best_parallel_strategy_with_recompute(
                world_size=world_size,
                gmi_error=GMI_ERROR,
                micro_batch_size=MICRO_BATCH_SIZE,
                global_batch_size=gbs_eff,
                all_search_result=all_search_result,
                tp_search_list=[tp],
                ep_search_list=[ep],
                pp_search_list=[pp],
                recompute_search_type=RECOMPUTE_TYPES,
                use_reserved_memory=False,
                dump_path=None,
            )

        try:
            if verbose:
                best = _run()
            else:
                with suppress_stdout():
                    best = _run()
        except Exception as e:
            print(f"  [tp={tp} pp={pp} ep={ep} dp={dp} gbs={gbs_eff}] ERROR: {type(e).__name__}: {e}")
            continue

        if best and "mfu" in best:
            fit += 1
            mfu = best["mfu"]
            if mfu > best_mfu:
                best_mfu = mfu
                best_overall = best
                print(f"  [tp={tp} pp={pp} ep={ep} dp={dp} gbs={gbs_eff}] NEW BEST "
                      f"mfu={mfu:.4f} recompute='{best['recompute_status']}'")

    print(f"[{model_name}] tried={tried} fit={fit} best_mfu={best_mfu:.4f}" if best_overall
          else f"[{model_name}] tried={tried} fit=0 — NO COMBO FITS")
    return best_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="h100_nvlink", help="System config name")
    parser.add_argument("--model", default=None,
                        help="Run only this model (by name as in configs/models/). Otherwise all.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full search output per combo (noisy).")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: configs/strategy/ colocated with models).")
    parser.add_argument("--gbs", type=int, default=None,
                        help="Target global batch size. If omitted, picked by model size: "
                             f"<{SMALL_THRESHOLD_B:g}B->{GBS_SMALL}, "
                             f"<{LARGE_THRESHOLD_B:g}B->{GBS_MEDIUM}, else {GBS_LARGE}.")
    parser.add_argument("--max-world", type=int, default=None,
                        help="Maximum world_size to consider. If omitted, picked by model size: "
                             f"<{SMALL_THRESHOLD_B:g}B->{MAX_WORLD_SMALL}, "
                             f"<{LARGE_THRESHOLD_B:g}B->{MAX_WORLD_MEDIUM}, else {MAX_WORLD_LARGE}.")
    parser.add_argument("--exact-gbs", action="store_true",
                        help="Require gbs_eff == gbs_target exactly (relax_factor=1.0). "
                             "By default, accepts combos within [gbs/2, gbs*2].")
    args = parser.parse_args()
    relax_factor = 1.0 if args.exact_gbs else GBS_RELAX_FACTOR

    system_config_path = get_simu_system_config(args.system)

    # Output directory: same as where configs/strategy/ lives in the repo.
    strategy_dir = args.output_dir or RELEASE_STRATEGY["root"]
    os.makedirs(strategy_dir, exist_ok=True)

    model_map = {k: v for k, v in RELEASE_MODELS.items() if k != "root"}
    if args.model is not None:
        if args.model not in model_map:
            raise SystemExit(f"Model '{args.model}' not found. Available: {list(model_map)}")
        model_map = {args.model: model_map[args.model]}

    summary = []
    for model_name, model_path in model_map.items():
        mcfg = ModelConfig.init_from_config_file(model_path)
        gbs_target = args.gbs if args.gbs is not None else auto_gbs(mcfg)
        max_world = args.max_world if args.max_world is not None else auto_max_world(mcfg)
        print(f"\n=== {model_name} (params≈{estimate_params_b(mcfg):.0f}B, gbs={gbs_target}, max_world={max_world}) ===")
        best = search_for_model(model_name, model_path, system_config_path,
                                gbs_target=gbs_target, max_world=max_world,
                                verbose=args.verbose, relax_factor=relax_factor)
        if not best:
            summary.append((model_name, None, "no-fit"))
            continue

        # Use the stored model_name from the config for the filename (e.g., "deepseek_r1").
        mcfg = ModelConfig.init_from_config_file(model_path)
        out_name = f"{mcfg.model_name}_optimal.json"
        out_path = os.path.join(strategy_dir, out_name)

        strategy_json = build_strategy_json(best)
        with open(out_path, "w") as f:
            json.dump(strategy_json, f, indent=4)
        print(f"  -> wrote {out_path}")
        summary.append((model_name, out_path, f"mfu={best['mfu']:.4f}"))

    print("\n=== Summary ===")
    for name, path, status in summary:
        print(f"  {name}: {status}{' -> ' + path if path else ''}")


if __name__ == "__main__":
    main()
