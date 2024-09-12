from simumax.tuning.strategy_searcher import StrategySearcher
from simumax.core.config import SystemConfig, ModelConfig


def main():
    candidate_strategy = {
        "seq_len": [8192],
        "world_size": [16384],
        "micro_batch_size": [1],
        "micro_batch_num": [1],
        "dtype": ["bf16"],
        "tp_size": [1, 2, 4, 8],
        "enable_sequence_parallel": [True],
        "interleaving_size": [1],
        "zero_state": [1],
        "use_fused_norm": [True],
        "no_sync": [True],
        "use_math_sdp": [False],
        "use_flash_sdp": [True],
        "use_fp32_accum_grad": [True],
        "use_fused_swiglu": [True],
        "enable_recompute": [False, True],
        "recompute_granularity": ["full_block"],
        "skip_ckpt_micro_batch_num": [0],
        "mem_factor": [0.94],
    }
    system_config_file = "../configs/system/a100_bf16.json"
    model_config_file = "../configs/models/llama3-405b_padding_128.json"

    system_config = SystemConfig.init_from_config_file(system_config_file)
    model_config = ModelConfig.init_from_config_file(model_config_file)

    searcher = StrategySearcher(model_config=model_config, system_config=system_config)
    res_list = searcher.search(candidate_strategy=candidate_strategy, topk=5)
    if not res_list:
        print("No strategy found.")
        return
    for i, res in enumerate(res_list):
        strategy, mfu, throughput_per_accelerator, breakdown_result, peak_mem = res
        print(
            f"# Rank {i+1} strategy:\n"
            f"## {strategy.to_dict()}\n"
            f"### mfu: {mfu} peak_mem: {peak_mem} throughput_per_accelerator: {throughput_per_accelerator}\n"
        )

if __name__ == "__main__":
    main()
