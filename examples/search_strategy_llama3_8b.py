from pathlib import Path
from pprint import pprint

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_perf_model():
    strategy_dict = StrategyConfig.read_json_file(
        str(REPO_ROOT / "configs/strategy/tp1_pp2_dp4_mbs1.json")
    )
    strategy_dict["enable_recompute"] = False
    strategy_dict["recompute_granularity"] = None
    strategy_dict["recompute_layer_num"] = 0

    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_dict(strategy_dict),
        model_config=ModelConfig.init_from_config_file(
            str(REPO_ROOT / "configs/models/llama3-8b.json")
        ),
        system_config=SystemConfig.init_from_config_file(
            str(REPO_ROOT / "configs/system/a100_pcie.json")
        ),
    )
    perf_model.model_config.padded_vocab_size = True
    perf_model.model_config.make_vocab_size_divisible_by = 128
    return perf_model


def search_batch_settings(perf_model: PerfLLM):
    print("=== Search batch settings under a fixed global batch size ===")
    (
        all_search_micro_batch_size,
        all_search_micro_batch_num,
        all_peak_cached_mem_list,
        all_cost_list,
    ) = perf_model.search_max_micro_batch_size_fixed_gbs(
        pp_size=perf_model.strategy.pp_size,
        dp_size=perf_model.strategy.dp_size,
        global_batch_size=32,
        gmi_error=10,
        use_reserved_memory=True,
        save_all=False,
        verbose=False,
    )
    result = {
        "micro_batch_size": all_search_micro_batch_size[0],
        "micro_batch_num": all_search_micro_batch_num[0],
        "peak_mem_gib_by_stage": all_peak_cached_mem_list[0],
        "iter_time": all_cost_list[0].data["duration_time_per_iter"],
        "throughput_per_accelerator": all_cost_list[0].data["throughput_per_accelerator"],
    }
    pprint(result)


def search_parallel_strategy(perf_model: PerfLLM):
    print("\n=== Search a small parallel-strategy space ===")
    all_search_result = {}
    best_strategy = perf_model.search_best_parallel_strategy(
        world_size=8,
        gmi_error=10,
        micro_batch_size=1,
        global_batch_size=32,
        all_search_result=all_search_result,
        tp_search_list=[1, 2],
        ep_search_list=[1],
        pp_search_list=[2],
        recompute_search_type=["no_recompute"],
        use_reserved_memory=True,
        dump_path=None,
        verbose=False,
    )
    pprint(best_strategy)


def main():
    perf_model = build_perf_model()
    search_batch_settings(perf_model)
    search_parallel_strategy(perf_model)


if __name__ == "__main__":
    main()
