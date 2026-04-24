"""Phase 1 training entry point: MaskablePPO on SimuMax-driven env.

Fixed-sequence, no-disturbance setting — trains a policy to schedule
forward/backward/weight tasks for a given model × strategy × system
deployment. Set ``seq_len_std`` in the strategy JSON or pass a
``--disturbance`` config to enable Phase 2 stochasticity.
"""

from __future__ import annotations

import argparse

from simumax.rl.env.env import RLEnvConfig
from simumax.rl.train import PPOTrainingConfig, train
from simumax.rl.env.types import RewardMode
from simumax.utils import (
    get_simu_disturbance_config,
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
    get_simu_training_config,
)


def _default_strategy_for(model: str) -> str:
    return model.replace("-", "_").replace(".", "_") + "_optimal_mfu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy config name (without .json); defaults to <model>_optimal_mfu",
    )
    parser.add_argument("--system", default="h100_nvlink")
    parser.add_argument(
        "--disturbance",
        default=None,
        help="Disturbance config name (without .json); omit for no disturbance",
    )
    parser.add_argument(
        "--training",
        default="default",
        help="Training config name (without .json) from configs/training/",
    )
    parser.add_argument("--log-dir", default="logs/rl_env")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--reward-mode",
        choices=[m.value for m in RewardMode],
        default=RewardMode.UTILIZATION.value,
    )
    args = parser.parse_args()

    strategy = args.strategy or _default_strategy_for(args.model)

    ppo_config = PPOTrainingConfig.init_from_config_file(
        get_simu_training_config(args.training)
    )
    env_config = RLEnvConfig(
        strategy_config=get_simu_strategy_config(strategy),
        model_config=get_simu_model_config(args.model),
        system_config=get_simu_system_config(args.system),
        disturbance_config=(
            get_simu_disturbance_config(args.disturbance)
            if args.disturbance is not None else None
        ),
        reward_mode=RewardMode(args.reward_mode),
        seed=ppo_config.seed,
    )
    train(
        env_config=env_config,
        ppo_config=ppo_config,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
