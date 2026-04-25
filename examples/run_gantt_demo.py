"""Render the pipeline Gantt chart for a chosen strategy + schedule.

Strategy and pipeline schedule are independent configs: the same
parallelism strategy can be simulated under any supported schedule by
selecting a different ``--schedule``. ``--strategy`` defaults to
``<model>_optimal_mfu`` (with ``-``/``.`` → ``_``); override explicitly
when you want a different strategy.

Examples
--------
    python examples/run_gantt_demo.py --model llama3-70b
    python examples/run_gantt_demo.py --model llama3-70b --schedule zb_h2
    python examples/run_gantt_demo.py --model llama3-70b \\
        --strategy llama3_70b_optimal_mfu --schedule interleaved_1f1b \\
        --system h100_nvlink --output my_chart.png
"""

import argparse
import statistics
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend; no GUI window pops up

from gymnasium.utils import seeding

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    PipelineScheduleConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.core.perf_llm import PerfLLM
from simumax.utils import (
    get_simu_disturbance_config,
    get_simu_model_config,
    get_simu_pp_scheduling_config,
    get_simu_strategy_config,
    get_simu_system_config,
)

SUPPORTED_SCHEDULES = ["1f1b", "zb_h1", "zb_h2", "gpipe",
                       "interleaved_1f1b", "zb_v"]


def _default_strategy_for(model: str) -> str:
    return model.replace("-", "_").replace(".", "_") + "_optimal_mfu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None,
                        help="Model config name (without .json). "
                             "Required unless --list-schedules is passed.")
    parser.add_argument("--strategy", default=None,
                        help="Strategy config name (without .json). "
                             "Defaults to <model>_optimal_mfu.")
    parser.add_argument("--schedule", default="1f1b",
                        help="Pipeline schedule config name (without .json), "
                             f"one of: {', '.join(SUPPORTED_SCHEDULES)}.")
    parser.add_argument("--system", default="h100_nvlink",
                        help="System config name (without .json).")
    parser.add_argument("--disturbance", default=None,
                        help="Disturbance config name (without .json), "
                             "e.g. 'default'. Omit to run with no disturbance.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path for the Gantt chart. "
                             "Defaults to a schedule-specific filename.")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes to simulate. With >1, the "
                             "disturbance seed is varied per episode (matching "
                             "eval_agents.py) and a summary of pp_utilization "
                             "is printed; the Gantt is only drawn when 1.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed driving per-episode disturbance seeds "
                             "(matches eval_agents.py --seed semantics).")
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

    if args.model is None:
        sys.exit("--model is required (or pass --list-schedules)")

    if args.n_episodes < 1:
        sys.exit("--n-episodes must be >= 1")
    if args.n_episodes > 1 and args.disturbance is None:
        sys.exit("--n-episodes > 1 requires --disturbance "
                 "(without it every episode is identical)")

    strategy_name = args.strategy or _default_strategy_for(args.model)
    strategy_path = get_simu_strategy_config(strategy_name)
    schedule_path = get_simu_pp_scheduling_config(args.schedule)
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
        pp_scheduling_config=PipelineScheduleConfig.init_from_config_file(schedule_path),
        disturbance_config=disturbance_config,
    )

    schedule = perf_model.pp_scheduling.pp_schedule
    pp = perf_model.strategy.pp_size
    mbc = perf_model.strategy.micro_batch_num

    if args.n_episodes == 1:
        perf_model.run_estimate()
        save_dir = f"{perf_model.model_config.model_name}_{perf_model.system.sys_name}"
        perf_model.analysis(save_dir)
        iter_time = perf_model.draw_pp_gantt(output_path=args.output)
        out_path = args.output or PerfLLM.default_gantt_filename(schedule)

        print()
        print(f"schedule = {schedule}")
        print(f"pp = {pp}, mbc = {mbc}")
        print(f"simulated iter time = {iter_time:.6f} s")
        print(f"Gantt saved to {out_path}")
        return

    # Multi-episode: mirror eval_agents.py seeding — a single Generator seeded
    # with --seed feeds one fresh int per episode into disturbance.seed, exactly
    # like PipelineSchedulingEnv.reset() does in the RL eval path.
    rng, _ = seeding.np_random(args.seed)
    utils: list[float] = []
    mfus: list[float] = []
    for ep in range(args.n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        perf_model.disturbance.seed = ep_seed
        perf_model.run_estimate()
        # analysis_cost() runs convert_final_result_to_human_format, so most
        # numeric fields end up stringified. pp_utilization and mfu stay floats.
        cost = perf_model.analysis_cost().data
        utils.append(float(cost["pp_utilization"]))
        mfus.append(float(cost["mfu"]))

    util_std = statistics.stdev(utils) if len(utils) > 1 else 0.0
    mfu_std = statistics.stdev(mfus) if len(mfus) > 1 else 0.0
    print()
    print(f"schedule = {schedule}")
    print(f"pp = {pp}, mbc = {mbc}")
    print(f"episodes = {args.n_episodes}, base seed = {args.seed}")
    print(f"pp_utilization: mean={statistics.fmean(utils):.4f} "
          f"std={util_std:.4f} min={min(utils):.4f} max={max(utils):.4f}")
    print(f"mfu:            mean={statistics.fmean(mfus):.4f} "
          f"std={mfu_std:.4f} min={min(mfus):.4f} max={max(mfus):.4f}")
    print("Gantt skipped (n_episodes > 1).")


if __name__ == "__main__":
    main()
