from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import get_simu_model_config, get_simu_strategy_config, get_simu_system_config

def set_megatron_config(perf_model:PerfLLM):
    perf_model.model_config.padded_vocab_size = True
    perf_model.model_config.make_vocab_size_divisible_by = 128

def main():
    # Use the interface of SimuMax to configure the model_config_path, strategy_config_path and system_config_path, you can alse pase the path directly, sush as: 
    # system_config_file = '../configs/system/a100_pcie.json'
    strategy_config_file = get_simu_strategy_config('tp4_pp1_dp2_mbs1')
    model_config_file = get_simu_model_config('llama3-8b')
    system_config_file = get_simu_system_config('a100_pcie')
    
    
    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_config_file(strategy_config_file),
        model_config=ModelConfig.init_from_config_file(model_config_file),
        system_config=SystemConfig.init_from_config_file(system_config_file),
    )

    model_name = perf_model.model_config.model_name
    system_name = perf_model.system.sys_name
    set_megatron_config(perf_model) # Optinionally, just for align the configuration with the benchmark
    perf_model.run_estimate()
    perf_model.analysis(f'{model_name}_{system_name}')

if __name__ == "__main__":
    main()