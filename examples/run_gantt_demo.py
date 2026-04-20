"""Render the 1F1B pipeline Gantt for Llama3-70B on the shipped A100-PCIe system.

Writes two things:
  - ./llama3_70b_a100_pcie_bf16/  (standard SimuMax JSON outputs)
  - ./corrected_1F1B_pipeline.png (Gantt chart of the 1F1B schedule)
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; no GUI window pops up

from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import (
    PerfLLM,
    FIRST_CHUNK,
    MIDDLE_CHUNK,
    LAST_CHUNK,
)
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def main():
    strategy_path = get_simu_strategy_config("llama70b_tp8_pp4_dp1")
    model_path    = get_simu_model_config("llama3-70b")
    system_path   = get_simu_system_config("a100_pcie")

    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_config_file(strategy_path),
        model_config=ModelConfig.init_from_config_file(model_path),
        system_config=SystemConfig.init_from_config_file(system_path),
    )

    perf_model.run_estimate()

    save_dir = f"{perf_model.model_config.model_name}_{perf_model.system.sys_name}"
    perf_model.analysis(save_dir)

    pp = perf_model.strategy.pp_size
    mbc = perf_model.strategy.micro_batch_num

    fwd_first, bwd_first = perf_model._compute_single_batch_fwd_bwd_time(FIRST_CHUNK)

    if pp == 1:
        forward_times, backward_times = [fwd_first], [bwd_first]
    else:
        fwd_last, bwd_last = perf_model._compute_single_batch_fwd_bwd_time(LAST_CHUNK)
        if pp == 2:
            forward_times  = [fwd_first, fwd_last]
            backward_times = [bwd_first, bwd_last]
        else:
            fwd_mid, bwd_mid = perf_model._compute_single_batch_fwd_bwd_time(MIDDLE_CHUNK)
            forward_times  = [fwd_first] + [fwd_mid] * (pp - 2) + [fwd_last]
            backward_times = [bwd_first] + [bwd_mid] * (pp - 2) + [bwd_last]

    iter_time = perf_model.calculate_1f1b_bubble(
        pp=pp,
        mbc=mbc,
        forward_times=forward_times,
        backward_times=backward_times,
        draw=True,
    )

    print()
    print(f"pp={pp}, mbc={mbc}")
    print(f"forward_times  (s) = {forward_times}")
    print(f"backward_times (s) = {backward_times}")
    print(f"simulated iter time from 1F1B scheduler = {iter_time:.6f} s")
    print("Gantt saved to ./corrected_1F1B_pipeline.png")


if __name__ == "__main__":
    main()
