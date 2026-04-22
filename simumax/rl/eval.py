"""Run agents against the RL env and collect iter times.

Handles both :mod:`simumax.rl.agents`'s static baselines (looked up via
``make_agent``) and trained :class:`~simumax.rl.agents.ppo.PPOAgent`
checkpoints (supplied by label -> path). The two families share the
same ``reset()`` / ``act()`` contract so the run loop is identical.

Shares the ``SimuMaxBackend`` across agents so the expensive
``run_estimate()`` only fires once. Each agent runs on a fresh env
instance seeded with the same ``base_seed``, so all agents see the
same per-episode disturbance / seq-len draws — apples-to-apples.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simumax.rl.agents import PPOAgent, make_agent
from simumax.rl.env.backend import SimuMaxBackend
from simumax.rl.env.env import PipelineSchedulingEnv, RLEnvConfig


@dataclass
class EvalResult:
    agent_name: str
    iter_times: list[float]

    def summary(self) -> dict[str, float]:
        n = len(self.iter_times)
        mean = statistics.fmean(self.iter_times) if n else 0.0
        std = statistics.stdev(self.iter_times) if n > 1 else 0.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": min(self.iter_times) if n else 0.0,
            "max": max(self.iter_times) if n else 0.0,
        }


def _run_episode(env: PipelineSchedulingEnv, agent, max_steps: int) -> float:
    # First reset on a fresh env pulls seed from env_config; subsequent
    # resets advance np_random naturally.
    obs, info = env.reset()
    agent.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = agent.act(obs, info["action_mask"])
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if steps > max_steps:
            msg = (
                f"Agent {type(agent).__name__} exceeded {max_steps} steps "
                f"without terminating — likely a schedule bug."
            )
            raise RuntimeError(msg)
    return float(env.current_time)


def evaluate(
    agent_names: list[str],
    env_config: RLEnvConfig,
    n_episodes: int = 1,
    base_seed: int = 0,
    render_dir: Optional[Path] = None,
    display: bool = False,
    ppo_checkpoints: Optional[dict[str, str]] = None,
) -> list[EvalResult]:
    """Run each agent for ``n_episodes`` episodes; return per-agent iter times.

    Names appearing in ``ppo_checkpoints`` are loaded via
    :class:`~simumax.rl.agents.PPOAgent`; all others are resolved through
    :func:`~simumax.rl.agents.make_agent`. ``render_dir`` and ``display``
    are independent — pass both to both save PNGs and open a GUI window
    per episode (``plt.show()`` blocks until closed).
    """
    backend = SimuMaxBackend(
        strategy_config=env_config.strategy_config,
        model_config=env_config.model_config,
        system_config=env_config.system_config,
        disturbance_config=env_config.disturbance_config,
    )
    pp = backend.num_gpus
    mbc = backend.num_microbatches

    # Allow plenty of headroom: ~3 ops per (mb, stage) + no-ops between.
    max_steps = (mbc * pp * 3 + 1) * 20

    ckpts = ppo_checkpoints or {}

    results: list[EvalResult] = []
    for name in agent_names:
        if name in ckpts:
            agent = PPOAgent(ckpts[name])
        else:
            agent = make_agent(name, pp, mbc)
        # Fresh env per agent so np_random starts from base_seed for all.
        agent_env_config = RLEnvConfig(**{**env_config.__dict__, "seed": base_seed})
        env = PipelineSchedulingEnv(agent_env_config, backend=backend)
        iter_times: list[float] = []
        for ep in range(n_episodes):
            t = _run_episode(env, agent, max_steps)
            iter_times.append(t)
            if render_dir is not None or display:
                out_path: Optional[str] = None
                if render_dir is not None:
                    render_dir.mkdir(parents=True, exist_ok=True)
                    out_path = str(render_dir / f"{name}_ep{ep:03d}.png")
                env.render(
                    out_path,
                    title=f"{name} ep {ep} (t={t:.6f}s)",
                    display=display,
                )
        results.append(EvalResult(agent_name=name, iter_times=iter_times))
    return results


def format_summary(results: list[EvalResult]) -> str:
    ordered = sorted(results, key=lambda r: r.summary()["mean"])
    lines = [
        f"  {'agent':<10} {'n':>4} {'mean (s)':>14} {'std':>14} "
        f"{'min':>14} {'max':>14}"
    ]
    for r in ordered:
        s = r.summary()
        lines.append(
            f"  {r.agent_name:<10} {int(s['n']):>4d} "
            f"{s['mean']:>14.6f} {s['std']:>14.6f} "
            f"{s['min']:>14.6f} {s['max']:>14.6f}"
        )
    return "\n".join(lines)
