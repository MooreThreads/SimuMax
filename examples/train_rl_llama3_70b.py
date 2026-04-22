"""Phase 1 training entry point: MaskablePPO on SimuMax-driven env.

Fixed-sequence, no-disturbance setting — trains a policy to schedule
forward/backward/weight tasks on a llama3-8b × tp1_pp2_dp4_mbs1 × a100
deployment. Flip ``seq_len_std`` / ``op_duration_std`` / etc. in the
strategy JSON to enable Phase 2 stochasticity.
"""

from __future__ import annotations

import argparse

from simumax.rl_env.env import RLEnvConfig
from simumax.rl_env.train import PPOTrainingConfig, train
from simumax.rl_env.types import RewardMode
from simumax.utils import (
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="llama70b_tp8_pp4_dp100")
    parser.add_argument("--model", default="llama3-70b")
    parser.add_argument("--system", default="h100_nvlink")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="logs/rl_env")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--reward-mode",
        choices=[m.value for m in RewardMode],
        default=RewardMode.UTILIZATION.value,
    )
    args = parser.parse_args()

    env_config = RLEnvConfig(
        strategy_config=get_simu_strategy_config(args.strategy),
        model_config=get_simu_model_config(args.model),
        system_config=get_simu_system_config(args.system),
        reward_mode=RewardMode(args.reward_mode),
        seed=args.seed,
    )
    ppo_config = PPOTrainingConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    train(
        env_config=env_config,
        ppo_config=ppo_config,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
