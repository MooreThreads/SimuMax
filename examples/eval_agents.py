"""Evaluate baseline and/or trained agents on the SimuMax-driven RL env.

Examples
--------
    python examples/eval_static_agents.py --agents gpipe 1f1b zb_h1 zb_h2
    python examples/eval_static_agents.py --agents zb_h2 --n-episodes 20 \\
        --disturbance default --render-dir /tmp/gantts
    python examples/eval_static_agents.py \\
        --agents 1f1b zb_h2 ppo_best \\
        --ppo-checkpoint ppo_best=/tmp/runs/foo/best_model.zip \\
        --display
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simumax.core.config import (
    DisturbanceConfig,
    ModelConfig,
    StrategyConfig,
    SystemConfig,
)
from simumax.rl.agents import AGENT_REGISTRY
from simumax.rl.env.env import RLEnvConfig
from simumax.rl.env.types import RewardMode
from simumax.rl.eval import evaluate, format_summary
from simumax.utils import (
    get_simu_disturbance_config,
    get_simu_model_config,
    get_simu_strategy_config,
    get_simu_system_config,
)


def _parse_ppo_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--ppo-checkpoint expects LABEL=PATH; got {value!r}"
        )
    label, _, path = value.partition("=")
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--ppo-checkpoint needs non-empty LABEL and PATH; got {value!r}"
        )
    if label in AGENT_REGISTRY:
        raise argparse.ArgumentTypeError(
            f"PPO label {label!r} collides with a built-in static agent; pick another"
        )
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strategy",
        default="tp1_pp2_dp4_mbs1",
        help="Strategy config name (without .json)",
    )
    parser.add_argument(
        "--model", default="llama3-8b", help="Model config name (without .json)"
    )
    parser.add_argument(
        "--system", default="a100_pcie", help="System config name (without .json)"
    )
    parser.add_argument(
        "--disturbance",
        default=None,
        help="Disturbance config name (without .json); omit for no disturbance",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=sorted(AGENT_REGISTRY),
        help=(
            "Agents to evaluate. Names default to the static registry "
            f"({sorted(AGENT_REGISTRY)}); any name you also pass via "
            "--ppo-checkpoint is loaded as a PPO checkpoint instead."
        ),
    )
    parser.add_argument(
        "--ppo-checkpoint",
        action="append",
        default=[],
        type=_parse_ppo_checkpoint,
        metavar="LABEL=PATH",
        help=(
            "Load a trained MaskablePPO checkpoint under LABEL. Repeatable. "
            "LABEL must also appear in --agents to be evaluated."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes per agent (default 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed — all agents replay identical episodes",
    )
    parser.add_argument(
        "--render-dir",
        default=None,
        help="If set, dump a Gantt PNG per (agent, episode)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open a blocking matplotlib window per episode "
        "(combine with --render-dir to also save PNGs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy = StrategyConfig.init_from_config_file(
        get_simu_strategy_config(args.strategy)
    )
    model = ModelConfig.init_from_config_file(get_simu_model_config(args.model))
    system = SystemConfig.init_from_config_file(get_simu_system_config(args.system))
    disturbance = None
    if args.disturbance is not None:
        disturbance = DisturbanceConfig.init_from_config_file(
            get_simu_disturbance_config(args.disturbance)
        )

    env_config = RLEnvConfig(
        strategy_config=strategy,
        model_config=model,
        system_config=system,
        disturbance_config=disturbance,
        reward_mode=RewardMode.UTILIZATION,
        seed=args.seed,
    )

    render_dir = Path(args.render_dir) if args.render_dir is not None else None
    ppo_checkpoints = dict(args.ppo_checkpoint)

    agent_names = list(args.agents)
    # Catch typos where a --ppo-checkpoint label wasn't also selected.
    unused = sorted(set(ppo_checkpoints) - set(agent_names))
    if unused:
        raise SystemExit(
            f"--ppo-checkpoint labels not in --agents: {unused}. "
            f"Add them to --agents or drop the checkpoint."
        )
    # Catch unknown names — anything not in the registry and not a PPO label.
    unknown = [
        n for n in agent_names if n not in AGENT_REGISTRY and n not in ppo_checkpoints
    ]
    if unknown:
        raise SystemExit(
            f"Unknown agent name(s): {unknown}. "
            f"Known static: {sorted(AGENT_REGISTRY)}; "
            f"PPO labels: {sorted(ppo_checkpoints)}"
        )

    results = evaluate(
        agent_names=agent_names,
        env_config=env_config,
        n_episodes=args.n_episodes,
        base_seed=args.seed,
        render_dir=render_dir,
        display=args.display,
        ppo_checkpoints=ppo_checkpoints,
    )

    print(
        f"Agent eval on {args.strategy} / {args.model} / {args.system} "
        f"over {args.n_episodes} episode(s) (seed={args.seed}):"
    )
    print(format_summary(results))
    if render_dir is not None:
        print(f"Gantts written to {render_dir}")


if __name__ == "__main__":
    main()
