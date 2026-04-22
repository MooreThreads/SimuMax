"""Render the pipeline Gantt chart for a chosen strategy.

Dispatches on the strategy's ``pp_schedule`` field (e.g. ``1f1b``, ``zb_h2``)
and writes the appropriate Gantt chart plus the standard SimuMax JSON
outputs.

Examples
--------
    python examples/run_gantt_demo.py
    python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_zbh2
    python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100 \\
        --model llama3-70b --system h100_nvlink --output my_chart.png
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend; no GUI window pops up

from simumax.core.config import DisturbanceConfig, ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_disturbance_config,
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)

SUPPORTED_SCHEDULES = ["1f1b", "zb_h1", "zb_h2", "gpipe",
                       "interleaved_1f1b", "zb_v"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="llama70b_tp8_pp4_dp100",
                        help="Strategy config name (without .json).")
    parser.add_argument("--model", default="llama3-70b",
                        help="Model config name (without .json).")
    parser.add_argument("--system", default="h100_nvlink",
                        help="System config name (without .json).")
    parser.add_argument("--disturbance", default=None,
                        help="Disturbance config name (without .json), "
                             "e.g. 'default'. Omit to run with no disturbance.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path for the Gantt chart. "
                             "Defaults to a schedule-specific filename.")
    parser.add_argument("--list-schedules", action="store_true",
                        help="Print the supported pp_schedule values and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_schedules:
        print("Supported pp_schedule values:")
        for s in SUPPORTED_SCHEDULES:
            print(f"  - {s}  -> default Gantt: "
                  f"{PerfLLM.default_gantt_filename(s)}")
        sys.exit(0)

    strategy_path = get_simu_strategy_config(args.strategy)
    model_path = get_simu_model_config(args.model)
    system_path = get_simu_system_config(args.system)
    disturbance_config = None
    if args.disturbance is not None:
        disturbance_config = DisturbanceConfig.init_from_config_file(
            get_simu_disturbance_config(args.disturbance)
        )

    perf_model = PerfLLM()
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_config_file(strategy_path),
        model_config=ModelConfig.init_from_config_file(model_path),
        system_config=SystemConfig.init_from_config_file(system_path),
        disturbance_config=disturbance_config,
    )
    perf_model.run_estimate()

    save_dir = f"{perf_model.model_config.model_name}_{perf_model.system.sys_name}"
    perf_model.analysis(save_dir)

    schedule = perf_model.strategy.pp_schedule
    iter_time = perf_model.draw_pp_gantt(output_path=args.output)

    pp = perf_model.strategy.pp_size
    mbc = perf_model.strategy.micro_batch_num
    out_path = args.output or PerfLLM.default_gantt_filename(schedule)

    print()
    print(f"schedule = {schedule}")
    print(f"pp = {pp}, mbc = {mbc}")
    print(f"simulated iter time = {iter_time:.6f} s")
    print(f"Gantt saved to {out_path}")


if __name__ == "__main__":
    main()
