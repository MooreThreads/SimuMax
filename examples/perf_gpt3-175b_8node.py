import os
import json

from simumax.core.perf_llm import PerfLLM


def main():
    system_config_file = "../configs/system/a100_bf16.json"
    strategy_config_file = "../configs/strategy/tp8_pp8_dp1_seq2k_mbs1_ckpt_sdp.json"
    model_config_file = "../configs/models/gpt3-175b.json"

    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=strategy_config_file,
        model_config=model_config_file,
        system_config=system_config_file,
        debug_points=[
            "(1)LLMBlock -> (0)LayerNorm",
            "(1)LLMBlock -> (1)Attention -> (0)LinearCol",
            "(1)LLMBlock -> (1)Attention -> (1)CoreAttention",
            "(1)LLMBlock -> (1)Attention -> (2)LinearRow",
            "(1)LLMBlock -> (2)LayerNorm",
            "(1)LLMBlock -> (3)MLP -> (0)LinearCol",
            "(1)LLMBlock -> (3)MLP -> (1)Swiglu",
            "(1)LLMBlock -> (3)MLP -> (2)LinearRow",
            "(1)LLMBlock -> (3)MLP",
            "(2)LLMBlock",
        ],
    )

    perf_model.run_estimate()
    os.makedirs("./tmp", exist_ok=True)


    base_info = {}
    base_info["arch"] = str(perf_model.model_chunk_dict)
    base_info["all_param"] = perf_model.model_config.param_numel
    base_info["act_param"] = perf_model.model_config.activated_param_numel
    with open("./tmp/model_arch", "w") as f:
        f.write(base_info["arch"])
    with open("./tmp/base_info.json", "w") as f:
        f.write(json.dumps(base_info, indent=2, sort_keys=False, ensure_ascii=False))

    mem_result = perf_model.analysis_mem()
    with open("./tmp/mem_result.json", "w") as f:
        f.write(str(mem_result))

    compute_result = perf_model.analysis_cost()

    with open("./tmp/compute_result.json", "w") as f:
        f.write(str(compute_result))
    print("max micro batch size: ", perf_model.search_max_micro_batch_size())


if __name__ == "__main__":
    main()
