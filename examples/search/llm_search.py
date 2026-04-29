import os
# os.environ['ENABLE_SIMU_GRAPH'] = "1"
from argparse import ArgumentParser
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config
from simumax.core.graph import  SimuONNXGraphBuilder, visualize_with_graphviz
from simumax.core.tensor import FakeTensor

def set_megatron_config(perf_model:PerfLLM):
    perf_model.model_config.moe_pad_expert_input_to_capacity = True
    perf_model.model_config.capacity = 1
    perf_model.model_config.padded_vocab_size = True
    perf_model.model_config.make_vocab_size_divisible_by = 128
    
    perf_model.strategy.dispatch_probs = True
    # perf_model.strategy.fp8 = True

def search(model, system):
    # Use the interface of SimuMax to configure the model_config_path, strategy_config_path and system_config_path, you can alse pase the path directly, sush as: 
    # system_config_file = '../configs/system/a100_pcie.json'
    strategy_config_file = get_simu_strategy_config('tp1_pp2_dp4_mbs1') # default is tp1_pp2_dp4_mbs1
    model_config_file = get_simu_model_config(model)
    system_config_file = get_simu_system_config(system)
    
    
    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_config_file(strategy_config_file),
        model_config=ModelConfig.init_from_config_file(model_config_file),
        system_config=SystemConfig.init_from_config_file(system_config_file),
    )

    
    set_megatron_config(perf_model) # Optinionally, just for align the configuration with the benchmark

    all_search_result = {}
    perf_model.search_best_parallel_strategy_with_recompute(
        world_size=2048,
        gmi_error=6, # 6G memory reserved
        micro_batch_size=1,
        global_batch_size= 2048*8,
        all_search_result=all_search_result,
        tp_search_list=[1],
        ep_search_list=[8, 16, 32, 64],
        recompute_search_type=['no_recompute', 'full_block', 'selective_recompute'],
        use_reserved_memory=False,
        dump_path=f"search_{perf_model.model_config.model_name}_{perf_model.system.sys_name}"
    )

if __name__ == "__main__":
    search('deepseekv3', 'a100_pcie')
    search('deepseekv2', 'a100_pcie')