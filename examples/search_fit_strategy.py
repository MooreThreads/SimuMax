"""
For each model in configs/models/, search the highest-MFU (tp, pp, ep, dp, recompute)
combination at a fixed (approximate) global batch size within a max world-size budget.

Framing: world_size = tp * pp * dp; ep is a partition within dp for MoE (requires ep | dp).
micro_batch_num is chosen as close as possible to gbs_target / dp while respecting the
pipeline validity constraint (mbn >= pp). Combos whose effective gbs would drift beyond
GBS_RELAX_FACTOR away from the target are dropped.

Output: configs/strategy/<model_name>_optimal_mfu.json, matching the StrategyConfig schema.

Disturbance mode (opt-in via --disturbance) implements the two-phase search described in
``Disturbed_Search_Plan.md``: Phase A is the nominal sweep, kept as a fast pre-filter,
emitting one candidate per (tp, pp, ep, dp, recompute_family); Phase B re-evaluates each
short-listed candidate under N seeded draws of the chosen DisturbanceConfig and selects
the highest-mean-MFU candidate whose peak memory stays within bounds across every seed.
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM
from simumax.core.utils import HumanReadableSize
from simumax.utils import (
    RELEASE_DISTURBANCE,
    RELEASE_MODELS,
    RELEASE_STRATEGY,
    get_simu_disturbance_config,
    get_simu_pp_scheduling_config,
    get_simu_system_config,
)


SUPPORTED_SCHEDULES = (
    "1f1b",
    "gpipe",
    "zb_h1",
    "zb_h2",
    "interleaved_1f1b",
    "zb_v",
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
PP_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]
EP_CANDIDATES_MOE = [1, 2, 4, 8, 16, 32, 64]
EP_CANDIDATES_DENSE = [1]
DP_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RECOMPUTE_TYPES = ["no_recompute", "full_block", "selective_recompute"]

# Skeleton strategy used to construct PerfLLM. The search overrides world_size,
# tp/pp/ep, mbs, mbn, and recompute_* per combo, so the specific values here
# don't matter as long as they form a valid StrategyConfig. Inlined to avoid a
# disk dependency on configs/strategy_buggy/llama3_70b_optimal_mfu.json.
TEMPLATE_STRATEGY = {
    "seq_len": 4096,
    "micro_batch_size": 1,
    "micro_batch_num": 128,
    "dtype": "bf16",
    "world_size": 32,
    "tp_size": 4,
    "pp_size": 4,
    "ep_size": 1,
    "etp_size": 1,
    "moe_dispatcher_policy": "all2all",
    "enable_sequence_parallel": True,
    "zero_state": 1,
    "enable_dropout": False,
    "use_fused_norm": True,
    "use_math_sdp": False,
    "use_flash_sdp": True,
    "use_fp32_accum_grad": True,
    "enable_recompute": False,
    "mem_factor": 0.94,
}


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

    # Carry over pipeline-split fields the search set when layer_num % pp != 0.
    # Without these, reload of a non-divisible-pp config trips the strict
    # divisibility assertion in get_num_layers_to_build (perf_llm.py).
    if best.get("num_layers_in_first_pipeline_stage") is not None:
        strategy["num_layers_in_first_pipeline_stage"] = best["num_layers_in_first_pipeline_stage"]
    if best.get("num_layers_in_last_pipeline_stage") is not None:
        strategy["num_layers_in_last_pipeline_stage"] = best["num_layers_in_last_pipeline_stage"]
    return strategy


def make_perf_model(
    model_config_path: str,
    system_config_path: str,
    pp_scheduling_config_path: Optional[str] = None,
    disturbance_config_path: Optional[str] = None,
) -> PerfLLM:
    perf = PerfLLM()
    pp_scheduling = (
        PipelineScheduleConfig.init_from_config_file(pp_scheduling_config_path)
        if pp_scheduling_config_path is not None
        else None
    )
    perf.configure(
        strategy_config=StrategyConfig.init_from_dict(TEMPLATE_STRATEGY),
        model_config=ModelConfig.init_from_config_file(model_config_path),
        system_config=SystemConfig.init_from_config_file(system_config_path),
        pp_scheduling_config=pp_scheduling,
        disturbance_config=disturbance_config_path,
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
                     relax_factor: float = GBS_RELAX_FACTOR,
                     pp_scheduling_config_path: Optional[str] = None):
    perf = make_perf_model(model_config_path, system_config_path,
                           pp_scheduling_config_path)
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
        perf = make_perf_model(model_config_path, system_config_path,
                               pp_scheduling_config_path)
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


# ---------------------------------------------------------------------------
# Disturbance mode
# ---------------------------------------------------------------------------
#
# Two-phase search per ``Disturbed_Search_Plan.md``:
#   - Phase A: enumerate combos, run the existing nominal search, mine
#     ``all_search_result`` to emit one PhaseACandidate per
#     (tp, pp, ep, dp, recompute_family). Apply the retention rule
#     (top-K + MFU window) across the union of family records.
#   - Phase B: for each retained candidate, evaluate under N seeded
#     disturbance draws. A candidate is kept only if peak memory fits
#     across every seed; ranking is by mean MFU across seeds. Default
#     early-exit on the first memory-failing seed; opt-out via
#     ``--full-seed-eval``.

FAMILIES = ("no_recompute", "full_block", "selective_recompute")


@dataclass
class PhaseACandidate:
    """One nominal-fitting record at granularity (tp, pp, ep, dp, family)."""
    # Combo
    tp: int
    pp: int
    ep: int
    dp: int
    mbn: int
    mbs: int
    etp: int
    world_size: int
    gbs_eff: int
    # Recompute family (fixed per candidate; different families are different candidates)
    recompute_family: str
    recompute_layer_num_nominal: int
    selective_flags: Optional[Dict[str, bool]]
    recompute_variance: bool
    # Pipeline split
    num_layers_in_first_pipeline_stage: Optional[int]
    num_layers_in_last_pipeline_stage: Optional[int]
    # Nominal metrics (Phase A)
    nominal_mfu: float
    nominal_peak_mem_gb: float

    def combo_key(self) -> str:
        return (
            f"tp{self.tp}.pp{self.pp}.ep{self.ep}.dp{self.dp}"
            f".mbn{self.mbn}.{self.recompute_family}"
        )


def _classify_family(recompute_status: str) -> str:
    if recompute_status == "No Recompute":
        return "no_recompute"
    if recompute_status.startswith("full_block"):
        return "full_block"
    if recompute_status.startswith("selective_recompute"):
        return "selective_recompute"
    raise ValueError(f"Unrecognized recompute_status: {recompute_status!r}")


def _peak_mem_to_gb(value: Any) -> float:
    """Coerce a peak_mem field (numeric bytes, '12.34 GB' string, or per-stage
    dict of either) to a single GB float (max across stages when nested)."""
    if value is None:
        raise ValueError("peak_mem is None")
    if isinstance(value, dict):
        return max(_peak_mem_to_gb(v) for v in value.values())
    if isinstance(value, (int, float)):
        return float(value) / (1024 ** 3)
    if isinstance(value, str):
        try:
            return HumanReadableSize.from_string(
                value, HumanReadableSize.BYTE_UNITS, 1024, target_unit="GB"
            ).get_value()
        except Exception:
            m = re.match(r"\s*([\d.]+)\s*([KMGT]?B)\s*", value)
            if not m:
                raise
            n = float(m.group(1))
            unit = m.group(2)
            factor = {"B": 1 / 1024 ** 3, "KB": 1 / 1024 ** 2,
                      "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}[unit]
            return n * factor
    raise TypeError(f"Cannot convert {type(value)} to GB: {value!r}")


def _per_family_best_records(all_search_result: Dict[str, list]) -> Dict[str, dict]:
    """Group ``all_search_result`` (a parallel-list dict produced by
    ``merge_dict``) by recompute family and keep the highest-MFU record per
    family. Returns ``{family: record_dict}``; family absent if no fitting
    record was emitted for it."""
    if not all_search_result or "mfu" not in all_search_result:
        return {}
    keys = list(all_search_result.keys())
    n = len(all_search_result["mfu"])
    out: Dict[str, dict] = {}
    for i in range(n):
        record = {k: all_search_result[k][i] for k in keys}
        family = _classify_family(record["recompute_status"])
        if family not in out or record["mfu"] > out[family]["mfu"]:
            out[family] = record
    return out


def _record_to_phase_a_candidate(
    record: dict,
    tp: int, pp: int, ep: int, dp: int, mbn: int, gbs_eff: int,
) -> PhaseACandidate:
    parallelism = parse_parallelism(record["parallelism"])
    recompute = parse_recompute_status(record["recompute_status"])
    family = _classify_family(record["recompute_status"])

    if family == "no_recompute":
        layer_num = 0
        flags = None
        # ``search_best_parallel_strategy_with_recompute`` sets recompute_variance=True
        # for the no-recompute branch (line ~4145 in perf_llm.py).
        recompute_variance = True
    elif family == "full_block":
        layer_num = recompute["recompute_layer_num"]
        flags = None
        recompute_variance = False  # megatron-LM full recompute does not support variance
    else:  # selective_recompute
        layer_num = recompute["recompute_layer_num"]
        flags = {
            "attn_recompute": recompute["attn_recompute"],
            "mla_rms_recompute": recompute["mla_rms_recompute"],
            "mlp_recompute": recompute["mlp_recompute"],
            "mlp_rms_recompute": recompute["mlp_rms_recompute"],
        }
        recompute_variance = recompute.get("recompute_variance", False)

    return PhaseACandidate(
        tp=tp, pp=pp, ep=ep, dp=dp, mbn=mbn,
        mbs=parallelism["micro_batch_size"],
        etp=parallelism["etp_size"],
        world_size=parallelism["world_size"],
        gbs_eff=gbs_eff,
        recompute_family=family,
        recompute_layer_num_nominal=layer_num,
        selective_flags=flags,
        recompute_variance=recompute_variance,
        num_layers_in_first_pipeline_stage=record.get("num_layers_in_first_pipeline_stage"),
        num_layers_in_last_pipeline_stage=record.get("num_layers_in_last_pipeline_stage"),
        nominal_mfu=float(record["mfu"]),
        nominal_peak_mem_gb=_peak_mem_to_gb(record["peak_mem"]),
    )


def phase_a_enumerate_candidates(
    model_name: str,
    model_config_path: str,
    system_config_path: str,
    gbs_target: int,
    max_world: int,
    relax_factor: float,
    pp_scheduling_config_path: Optional[str],
) -> List[PhaseACandidate]:
    """Phase A: nominal sweep that emits one candidate per
    (tp, pp, ep, dp, recompute_family). Disturbance is *not* engaged here.
    Mining is via ``all_search_result`` so we do not rerun the inner search."""
    # Construct once just to read model_config for combo enumeration.
    perf = make_perf_model(model_config_path, system_config_path,
                           pp_scheduling_config_path)
    mcfg = perf.model_config
    combos = list(enumerate_combos(mcfg, gbs_target=gbs_target, max_world=max_world,
                                   relax_factor=relax_factor))
    print(f"[{model_name}] Phase A: {len(combos)} (tp, pp, ep, dp) combos × up to "
          f"{len(FAMILIES)} families to enumerate")

    candidates: List[PhaseACandidate] = []
    tried = 0
    for tp, pp, ep, dp, mbn, gbs_eff in combos:
        world_size = tp * pp * dp
        tried += 1
        perf = make_perf_model(model_config_path, system_config_path,
                               pp_scheduling_config_path)
        all_search_result: Dict[str, list] = {}

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
                recompute_search_type=list(FAMILIES),
                use_reserved_memory=False,
                dump_path=None,
            )

        try:
            with suppress_stdout():
                _run()
        except Exception as e:
            print(f"  [tp={tp} pp={pp} ep={ep} dp={dp} gbs={gbs_eff}] ERROR: "
                  f"{type(e).__name__}: {e}")
            continue

        per_family = _per_family_best_records(all_search_result)
        for family, record in per_family.items():
            try:
                cand = _record_to_phase_a_candidate(record, tp, pp, ep, dp, mbn, gbs_eff)
            except Exception as e:
                print(f"  [tp={tp} pp={pp} ep={ep} dp={dp} {family}] skipped: "
                      f"could not parse record ({type(e).__name__}: {e})")
                continue
            candidates.append(cand)

    print(f"[{model_name}] Phase A: tried={tried} combos, "
          f"{len(candidates)} (combo, family) candidates fit at nominal")
    return candidates


def apply_candidate_to_strategy(perf: PerfLLM, c: PhaseACandidate) -> None:
    """Set every strategy field that defines this candidate. Mirrors what
    ``search_best_parallel_strategy_with_recompute`` would set up internally."""
    s = perf.strategy
    s.world_size = c.world_size
    s.tp_size = c.tp
    s.pp_size = c.pp
    s.ep_size = c.ep
    s.etp_size = c.etp
    s.micro_batch_size = c.mbs
    s.micro_batch_num = c.mbn
    s.num_layers_in_first_pipeline_stage = c.num_layers_in_first_pipeline_stage
    s.num_layers_in_last_pipeline_stage = c.num_layers_in_last_pipeline_stage

    if c.recompute_family == "no_recompute":
        s.enable_recompute = False
        s.recompute_granularity = None
        s.recompute_layer_num = 0
        s.recompute_variance = c.recompute_variance
    elif c.recompute_family == "full_block":
        s.enable_recompute = True
        s.recompute_granularity = "full_block"
        s.recompute_layer_num = c.recompute_layer_num_nominal
        s.recompute_variance = False
    else:  # selective_recompute
        s.enable_recompute = True
        s.recompute_granularity = "selective_recompute"
        s.recompute_layer_num = c.recompute_layer_num_nominal
        for k, v in (c.selective_flags or {}).items():
            setattr(s, k, v)
        s.recompute_variance = c.recompute_variance


def evaluate_under_disturbance(
    candidate: PhaseACandidate,
    seed: int,
    model_config_path: str,
    system_config_path: str,
    pp_scheduling_config_path: Optional[str],
    disturbance_config_path: str,
) -> Dict[str, Any]:
    """Run one (candidate, seed) evaluation under disturbance.

    Behavior per family:
      - ``no_recompute`` / ``selective_recompute``: configuration is fully
        determined by the candidate; we run a single ``run_estimate`` /
        ``analysis_mem`` / ``analysis_cost`` and check the memory bound.
      - ``full_block``: the level may need to rise under this seed's
        draws; we re-run the existing binary search over recompute level
        (``search_best_recompute_layer_num``). If no level fits, the
        candidate fails for this seed.
    """
    perf = make_perf_model(model_config_path, system_config_path,
                           pp_scheduling_config_path,
                           disturbance_config_path=disturbance_config_path)
    perf.disturbance.seed = seed
    apply_candidate_to_strategy(perf, candidate)

    accelerator_mem_gb = perf.system.accelerator.mem_gbs - GMI_ERROR

    if candidate.recompute_family == "full_block":
        all_search_result: Dict[str, list] = {}
        try:
            with suppress_stdout():
                best = perf.search_best_recompute_layer_num(
                    layer_num=perf.model_config.layer_num,
                    use_reserved_memory=False,
                    gmi_error=GMI_ERROR,
                    best_mfu=-1.0,
                    all_search_result=all_search_result,
                )
        except Exception as e:
            return {"fits": False, "peak_mem_gb": None, "mfu": None,
                    "recompute_layer_num": None,
                    "error": f"{type(e).__name__}: {e}"}
        if not best:
            # No level fits at this seed.
            return {"fits": False, "peak_mem_gb": None, "mfu": None,
                    "recompute_layer_num": None}
        return {
            "fits": True,
            "peak_mem_gb": _peak_mem_to_gb(best["peak_mem"]),
            "mfu": float(best["mfu"]),
            "recompute_layer_num": parse_recompute_status(
                best["recompute_status"])["recompute_layer_num"],
        }

    # no_recompute or selective_recompute: fixed config, single shot.
    try:
        with suppress_stdout():
            perf.run_estimate()
            mem_result = perf.analysis_mem()
            cost_result = perf.analysis_cost()
            peak_list = perf.get_pp_stage_peak_mem(mem_result, "peak_mem", toG=True)
    except Exception as e:
        return {"fits": False, "peak_mem_gb": None, "mfu": None,
                "recompute_layer_num": candidate.recompute_layer_num_nominal,
                "error": f"{type(e).__name__}: {e}"}

    peak_mem_gb = max(peak_list.values())
    if peak_mem_gb > accelerator_mem_gb:
        return {"fits": False, "peak_mem_gb": peak_mem_gb, "mfu": None,
                "recompute_layer_num": candidate.recompute_layer_num_nominal}
    return {
        "fits": True,
        "peak_mem_gb": peak_mem_gb,
        "mfu": float(cost_result.data["mfu_6nd_with_attn"]),
        "recompute_layer_num": candidate.recompute_layer_num_nominal,
    }


def aggregate_seed_results(
    candidate: PhaseACandidate,
    seed_results: List[Dict[str, Any]],
    seeds_attempted: List[int],
    full_seed_eval: bool,
) -> Dict[str, Any]:
    """Build the audit record for one candidate after running its seed loop.

    ``seed_results`` is in the same order as ``seeds_attempted``; under default
    early-exit it may be shorter than ``num_seeds`` (stops at first failure)."""
    record: Dict[str, Any] = {
        "combo": {
            "tp": candidate.tp, "pp": candidate.pp, "ep": candidate.ep,
            "dp": candidate.dp, "mbn": candidate.mbn, "mbs": candidate.mbs,
            "etp": candidate.etp, "world_size": candidate.world_size,
            "gbs_eff": candidate.gbs_eff,
            "recompute_family": candidate.recompute_family,
            "recompute_layer_num_nominal": candidate.recompute_layer_num_nominal,
            "selective_flags": candidate.selective_flags,
        },
        "nominal_mfu": candidate.nominal_mfu,
        "nominal_peak_mem_gb": candidate.nominal_peak_mem_gb,
        "n_seeds_attempted": len(seeds_attempted),
        "n_seeds_evaluated": len(seed_results),
        "early_exit": (not full_seed_eval),
    }

    failures = [(i, r) for i, r in enumerate(seed_results) if not r["fits"]]
    fits_all = (len(failures) == 0 and len(seed_results) == len(seeds_attempted))
    record["fits_all_seeds"] = fits_all

    if failures:
        first_idx, first = failures[0]
        record["failed_at_seed"] = seeds_attempted[first_idx]
        record["peak_mem_at_failure_gb"] = first.get("peak_mem_gb")
        record["n_seeds_failed_memory"] = len(failures)
        if "error" in first:
            record["first_failure_error"] = first["error"]

    fitting = [r for r in seed_results if r["fits"]]
    if fitting:
        mfus = [r["mfu"] for r in fitting]
        peaks = [r["peak_mem_gb"] for r in fitting]
        levels = [r["recompute_layer_num"] for r in fitting
                  if r["recompute_layer_num"] is not None]
        record.update({
            "mfu_mean": statistics.mean(mfus),
            "mfu_std": statistics.stdev(mfus) if len(mfus) >= 2 else 0.0,
            "mfu_min": min(mfus),
            "mfu_max": max(mfus),
            "peak_mem_mean_gb": statistics.mean(peaks),
            "peak_mem_std_gb": statistics.stdev(peaks) if len(peaks) >= 2 else 0.0,
            "peak_mem_max_gb": max(peaks),
            "recompute_layer_num_max": max(levels) if levels else None,
        })
    return record


def phase_b_evaluate_candidates(
    model_name: str,
    candidates: List[PhaseACandidate],
    model_config_path: str,
    system_config_path: str,
    pp_scheduling_config_path: Optional[str],
    disturbance_config_path: str,
    num_seeds: int,
    seed_base: int,
    full_seed_eval: bool,
) -> List[Dict[str, Any]]:
    audit: List[Dict[str, Any]] = []
    seeds = list(range(seed_base, seed_base + num_seeds))

    for idx, cand in enumerate(candidates, start=1):
        seed_results: List[Dict[str, Any]] = []
        for s in seeds:
            res = evaluate_under_disturbance(
                cand, s,
                model_config_path, system_config_path,
                pp_scheduling_config_path, disturbance_config_path,
            )
            seed_results.append(res)
            if not res["fits"] and not full_seed_eval:
                break  # early exit: candidate is dead under strict filter

        audit_record = aggregate_seed_results(cand, seed_results, seeds, full_seed_eval)
        audit.append(audit_record)

        if audit_record["fits_all_seeds"]:
            print(
                f"  [{idx}/{len(candidates)}] {cand.combo_key()} "
                f"FITS_ALL mfu_mean={audit_record['mfu_mean']:.4f} "
                f"±{audit_record['mfu_std']:.4f} "
                f"peak_max={audit_record['peak_mem_max_gb']:.1f}GB"
            )
        else:
            failed_at = audit_record.get("failed_at_seed", "?")
            print(
                f"  [{idx}/{len(candidates)}] {cand.combo_key()} "
                f"REJECTED at seed={failed_at} "
                f"(evaluated {audit_record['n_seeds_evaluated']}/"
                f"{audit_record['n_seeds_attempted']} seeds)"
            )
    return audit


def filter_candidates(
    candidates: List[PhaseACandidate],
    candidate_top_k: int,
    candidate_mfu_window: float,
) -> Tuple[List[PhaseACandidate], float]:
    if not candidates:
        return [], -1.0
    nominal_best = max(c.nominal_mfu for c in candidates)
    cutoff = nominal_best * (1.0 - candidate_mfu_window)
    sorted_cands = sorted(candidates, key=lambda c: -c.nominal_mfu)
    in_window = [c for c in sorted_cands if c.nominal_mfu >= cutoff]
    short_list = in_window[:candidate_top_k]
    return short_list, nominal_best


def candidate_to_strategy_json(
    candidate: PhaseACandidate,
    recompute_layer_num_override: Optional[int] = None,
) -> dict:
    """Build a StrategyConfig-shaped dict from a candidate. For ``full_block``
    candidates Phase B records the max level across seeds; pass it as
    ``recompute_layer_num_override`` so the saved strategy reflects the
    pessimistic-seed level (otherwise a saved config would OOM on the seed
    that demanded the larger level)."""
    strategy = {
        "seq_len": SEQ_LEN,
        "micro_batch_size": candidate.mbs,
        "micro_batch_num": candidate.mbn,
        "dtype": DTYPE,
        "world_size": candidate.world_size,
        "tp_size": candidate.tp,
        "pp_size": candidate.pp,
        "ep_size": candidate.ep,
        "etp_size": candidate.etp,
        "moe_dispatcher_policy": "all2all",
        "enable_sequence_parallel": True,
        "zero_state": 1,
        "enable_dropout": False,
        "use_fused_norm": True,
        "use_math_sdp": False,
        "use_flash_sdp": True,
        "use_fp32_accum_grad": True,
        "enable_recompute": candidate.recompute_family != "no_recompute",
        "mem_factor": 0.94,
    }

    if candidate.recompute_family == "full_block":
        strategy["recompute_granularity"] = "full_block"
        strategy["recompute_layer_num"] = (
            recompute_layer_num_override
            if recompute_layer_num_override is not None
            else candidate.recompute_layer_num_nominal
        )
    elif candidate.recompute_family == "selective_recompute":
        strategy["recompute_granularity"] = "selective_recompute"
        strategy["recompute_layer_num"] = candidate.recompute_layer_num_nominal
        flags = candidate.selective_flags or {}
        strategy["attn_recompute"] = flags.get("attn_recompute", False)
        strategy["mla_rms_recompute"] = flags.get("mla_rms_recompute", False)
        strategy["mlp_recompute"] = flags.get("mlp_recompute", False)
        strategy["mlp_rms_recompute"] = flags.get("mlp_rms_recompute", False)
        strategy["recompute_variance"] = candidate.recompute_variance

    if candidate.num_layers_in_first_pipeline_stage is not None:
        strategy["num_layers_in_first_pipeline_stage"] = (
            candidate.num_layers_in_first_pipeline_stage
        )
    if candidate.num_layers_in_last_pipeline_stage is not None:
        strategy["num_layers_in_last_pipeline_stage"] = (
            candidate.num_layers_in_last_pipeline_stage
        )
    return strategy


def _git_rev() -> Optional[str]:
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def search_for_model_disturbed(
    model_name: str,
    model_config_path: str,
    system_config_path: str,
    gbs_target: int,
    max_world: int,
    relax_factor: float,
    pp_scheduling_config_path: Optional[str],
    disturbance_config_path: str,
    num_seeds: int,
    seed_base: int,
    candidate_top_k: int,
    candidate_mfu_window: float,
    full_seed_eval: bool,
) -> Tuple[Optional[PhaseACandidate], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Returns (winner_candidate, winner_audit_record, full_audit_dict).
    ``winner_*`` are None if no candidate fits across all seeds."""
    candidates = phase_a_enumerate_candidates(
        model_name, model_config_path, system_config_path,
        gbs_target=gbs_target, max_world=max_world,
        relax_factor=relax_factor,
        pp_scheduling_config_path=pp_scheduling_config_path,
    )

    short_list, nominal_best_mfu = filter_candidates(
        candidates, candidate_top_k, candidate_mfu_window
    )

    audit: Dict[str, Any] = {
        "model": model_name,
        "schedule": os.path.basename(pp_scheduling_config_path or "")
                    .replace(".json", "") or None,
        "disturbance": os.path.basename(disturbance_config_path)
                       .replace(".json", ""),
        "num_seeds": num_seeds,
        "seed_base": seed_base,
        "candidate_top_k": candidate_top_k,
        "candidate_mfu_window": candidate_mfu_window,
        "full_seed_eval": full_seed_eval,
        "gbs_target": gbs_target,
        "max_world": max_world,
        "n_phase_a_candidates": len(candidates),
        "n_phase_a_shortlist": len(short_list),
        "nominal_best_mfu": nominal_best_mfu if candidates else None,
        "git_rev": _git_rev(),
    }

    if not candidates:
        audit["selected"] = None
        audit["reason"] = "no_combo_fits_at_nominal"
        audit["candidates"] = []
        return None, None, audit

    print(f"[{model_name}] Phase A: nominal_best_mfu={nominal_best_mfu:.4f}, "
          f"short-list size={len(short_list)} "
          f"(window={candidate_mfu_window}, top_k={candidate_top_k})")

    print(f"[{model_name}] Phase B: re-evaluating {len(short_list)} candidates "
          f"under {num_seeds} seeds "
          f"(seed_base={seed_base}, full_seed_eval={full_seed_eval})")
    audit_records = phase_b_evaluate_candidates(
        model_name, short_list,
        model_config_path, system_config_path,
        pp_scheduling_config_path, disturbance_config_path,
        num_seeds=num_seeds, seed_base=seed_base,
        full_seed_eval=full_seed_eval,
    )
    audit["candidates"] = audit_records

    survivors = [(c, r) for c, r in zip(short_list, audit_records)
                 if r["fits_all_seeds"]]
    if not survivors:
        audit["selected"] = None
        audit["reason"] = "no_combo_fits_all_seeds"
        return None, None, audit

    winner_cand, winner_record = max(survivors, key=lambda cr: cr[1]["mfu_mean"])
    audit["selected"] = {
        "combo": winner_record["combo"],
        "mfu_mean": winner_record["mfu_mean"],
        "mfu_std": winner_record["mfu_std"],
        "peak_mem_mean_gb": winner_record["peak_mem_mean_gb"],
        "peak_mem_max_gb": winner_record["peak_mem_max_gb"],
        "recompute_layer_num_max": winner_record.get("recompute_layer_num_max"),
    }
    return winner_cand, winner_record, audit


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
    parser.add_argument("--schedule", default="1f1b",
                        choices=SUPPORTED_SCHEDULES,
                        help="Pipeline schedule under which the strategy will run. "
                             "The fit gate uses schedule-aware peak memory, so "
                             "different schedules yield different optimal strategies. "
                             "Default '1f1b' reproduces the legacy behavior.")
    parser.add_argument("--disturbance", default=None,
                        help="Disturbance config name (resolved under "
                             "configs/disturbance/) or absolute path. When set, "
                             "engages the two-phase disturbed search per "
                             "Disturbed_Search_Plan.md. When unset, runs the "
                             "nominal-only path (default; behavior unchanged).")
    parser.add_argument("--num-seeds", type=int, default=20,
                        help="Number of seeded draws in Phase B. Ignored when "
                             "--disturbance is absent. Default 20 — defensible for "
                             "ranking within a schedule under expected variance; "
                             "bump to 50 for paper-grade close cross-schedule "
                             "comparisons.")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="First disturbance seed in Phase B; subsequent seeds "
                             "are seed_base + i. Reproduces or extends a prior sweep.")
    parser.add_argument("--candidate-top-k", type=int, default=30,
                        help="Hard cap on Phase A short-list size (per "
                             "(model, schedule)). Applied across all "
                             "(combo, recompute_family) records.")
    parser.add_argument("--candidate-mfu-window", type=float, default=0.10,
                        help="Keep all (combo, family) records within this "
                             "fractional MFU gap of nominal-best (e.g. 0.10 keeps "
                             "everything within 10%% of the nominal winner). The "
                             "intersection with --candidate-top-k is taken.")
    parser.add_argument("--full-seed-eval", action="store_true",
                        help="Disable the early-exit-on-memory-failure optimization "
                             "in Phase B. When set, every candidate runs all N seeds "
                             "regardless of intermediate memory violations. Useful for "
                             "diagnosing near-misses or post-hoc relaxing the filter; "
                             "costs strictly more, never less.")
    args = parser.parse_args()
    relax_factor = 1.0 if args.exact_gbs else GBS_RELAX_FACTOR

    system_config_path = get_simu_system_config(args.system)
    pp_scheduling_config_path = get_simu_pp_scheduling_config(args.schedule)

    disturbance_config_path: Optional[str] = None
    disturbance_tag: Optional[str] = None
    if args.disturbance is not None:
        if os.path.isabs(args.disturbance) and os.path.isfile(args.disturbance):
            disturbance_config_path = args.disturbance
            disturbance_tag = os.path.basename(args.disturbance).replace(".json", "")
        else:
            disturbance_config_path = get_simu_disturbance_config(args.disturbance)
            disturbance_tag = args.disturbance

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

        if disturbance_config_path is None:
            print(f"\n=== {model_name} (params≈{estimate_params_b(mcfg):.0f}B, "
                  f"gbs={gbs_target}, max_world={max_world}, "
                  f"schedule={args.schedule}) ===")
            best = search_for_model(model_name, model_path, system_config_path,
                                    gbs_target=gbs_target, max_world=max_world,
                                    verbose=args.verbose, relax_factor=relax_factor,
                                    pp_scheduling_config_path=pp_scheduling_config_path)
            if not best:
                summary.append((model_name, None, "no-fit"))
                continue

            # Schedule-suffixed filename. Even for ``--schedule 1f1b`` the new
            # output goes to ``*_optimal_mfu_1f1b.json`` rather than silently
            # overwriting the existing ``*_optimal_mfu.json`` release artifacts
            # (see ``Schedule_Aware_Memory_Plan.md`` §5.5.3 / §7.1). The user
            # promotes a fresh result to the unsuffixed filename by hand once
            # the anti-regression check has passed.
            mcfg = ModelConfig.init_from_config_file(model_path)
            out_name = f"{mcfg.model_name}_optimal_mfu_{args.schedule}.json"
            out_path = os.path.join(strategy_dir, out_name)

            strategy_json = build_strategy_json(best)
            with open(out_path, "w") as f:
                json.dump(strategy_json, f, indent=4)
            print(f"  -> wrote {out_path}")
            summary.append((model_name, out_path, f"mfu={best['mfu']:.4f}"))
            continue

        # Disturbance mode
        print(f"\n=== {model_name} (params≈{estimate_params_b(mcfg):.0f}B, "
              f"gbs={gbs_target}, max_world={max_world}, "
              f"schedule={args.schedule}, disturbance={disturbance_tag}, "
              f"N={args.num_seeds}) ===")
        winner_cand, winner_record, audit = search_for_model_disturbed(
            model_name, model_path, system_config_path,
            gbs_target=gbs_target, max_world=max_world,
            relax_factor=relax_factor,
            pp_scheduling_config_path=pp_scheduling_config_path,
            disturbance_config_path=disturbance_config_path,
            num_seeds=args.num_seeds,
            seed_base=args.seed_base,
            candidate_top_k=args.candidate_top_k,
            candidate_mfu_window=args.candidate_mfu_window,
            full_seed_eval=args.full_seed_eval,
        )

        # Audit JSON is always written (even on no-fit) so users can see why.
        out_name_base = (
            f"{mcfg.model_name}_optimal_mfu_{args.schedule}_{disturbance_tag}"
        )
        audit_path = os.path.join(strategy_dir, f"{out_name_base}_audit.json")
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=4, default=lambda o: list(o) if isinstance(o, tuple) else o)
        print(f"  -> wrote audit {audit_path}")

        if winner_cand is None:
            reason = audit.get("reason", "no_winner")
            print(f"[{model_name}] NO WINNER ({reason}) — strategy JSON not written")
            summary.append((model_name, audit_path, f"no-fit ({reason})"))
            continue

        strategy_json = candidate_to_strategy_json(
            winner_cand,
            recompute_layer_num_override=winner_record.get("recompute_layer_num_max"),
        )
        out_path = os.path.join(strategy_dir, f"{out_name_base}.json")
        with open(out_path, "w") as f:
            json.dump(strategy_json, f, indent=4)
        print(f"  -> wrote {out_path}")
        summary.append((
            model_name, out_path,
            f"mfu_mean={winner_record['mfu_mean']:.4f} "
            f"±{winner_record['mfu_std']:.4f} "
            f"(peak_max={winner_record['peak_mem_max_gb']:.1f}GB)",
        ))

    print("\n=== Summary ===")
    for name, path, status in summary:
        print(f"  {name}: {status}{' -> ' + path if path else ''}")


if __name__ == "__main__":
    main()
