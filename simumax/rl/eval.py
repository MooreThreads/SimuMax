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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from simumax.rl.agents import PPOAgent, make_agent
from simumax.rl.env.backend import SimuMaxBackend
from simumax.rl.env.env import PipelineSchedulingEnv, RLEnvConfig


@dataclass
class EvalResult:
    agent_name: str
    iter_times: list[float]
    pp_utilizations: list[float] = field(default_factory=list)
    mfus: list[float] = field(default_factory=list)

    def _stats(self, values: list[float]) -> dict[str, float]:
        n = len(values)
        mean = statistics.fmean(values) if n else 0.0
        std = statistics.stdev(values) if n > 1 else 0.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": min(values) if n else 0.0,
            "max": max(values) if n else 0.0,
        }

    def summary(self) -> dict[str, float]:
        return self._stats(self.iter_times)

    def utilization_summary(self) -> dict[str, float]:
        return self._stats(self.pp_utilizations)

    def mfu_summary(self) -> dict[str, float]:
        return self._stats(self.mfus)


def _run_episode(
    env: PipelineSchedulingEnv, agent, max_steps: int
) -> tuple[float, float, float]:
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
    return (
        float(info["iter_time"]),
        float(info["pp_utilization"]),
        float(info["mfu"]),
    )


def evaluate(
    agent_names: list[str],
    env_config: RLEnvConfig,
    n_episodes: int = 1,
    base_seed: int = 0,
    render_dir: Optional[Path] = None,
    display: bool = False,
    ppo_checkpoints: Optional[dict[str, str]] = None,
    show_progress: bool = False,
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
        pp_scheduling_config=env_config.pp_scheduling_config,
        disturbance_config=env_config.disturbance_config,
    )
    pp = backend.num_gpus
    mbc = backend.num_microbatches

    # Allow plenty of headroom: ~3 ops per (mb, stage) + no-ops between.
    max_steps = (mbc * pp * 3 + 1) * 20

    ckpts = ppo_checkpoints or {}

    results: list[EvalResult] = []
    total = len(agent_names) * n_episodes
    progress_ctx = (
        tqdm(total=total, desc="episodes", unit="ep")
        if show_progress
        else nullcontext()
    )
    with progress_ctx as pbar:
        for name in agent_names:
            if name in ckpts:
                agent = PPOAgent(ckpts[name])
            else:
                agent = make_agent(name, pp, mbc)
            # Fresh env per agent so np_random starts from base_seed for all.
            agent_env_config = RLEnvConfig(
                **{**env_config.__dict__, "seed": base_seed}
            )
            env = PipelineSchedulingEnv(agent_env_config, backend=backend)
            iter_times: list[float] = []
            pp_utilizations: list[float] = []
            mfus: list[float] = []
            for ep in range(n_episodes):
                if pbar is not None:
                    pbar.set_postfix_str(f"{name} ep{ep}")
                t, u, mfu = _run_episode(env, agent, max_steps)
                iter_times.append(t)
                pp_utilizations.append(u)
                mfus.append(mfu)
                if render_dir is not None or display:
                    out_path: Optional[str] = None
                    if render_dir is not None:
                        render_dir.mkdir(parents=True, exist_ok=True)
                        out_path = str(render_dir / f"{name}_ep{ep:03d}.png")
                    env.render(
                        out_path,
                        title=f"{name} ep {ep} (t={t:.6f}s, u={u:.4f}, mfu={mfu:.4f})",
                        display=display,
                    )
                if pbar is not None:
                    pbar.update(1)
            results.append(
                EvalResult(
                    agent_name=name,
                    iter_times=iter_times,
                    pp_utilizations=pp_utilizations,
                    mfus=mfus,
                )
            )
    return results


def format_summary(results: list[EvalResult]) -> str:
    ordered = sorted(results, key=lambda r: r.summary()["mean"])
    lines = [
        f"  {'agent':<10} {'n':>4} "
        f"{'iter mean (s)':>14} {'iter std':>14} "
        f"{'iter min':>14} {'iter max':>14} "
        f"{'util mean':>12} {'util std':>12} "
        f"{'util min':>12} {'util max':>12} "
        f"{'mfu mean':>12} {'mfu std':>12} "
        f"{'mfu min':>12} {'mfu max':>12}"
    ]
    for r in ordered:
        s = r.summary()
        u = r.utilization_summary()
        m = r.mfu_summary()
        lines.append(
            f"  {r.agent_name:<10} {int(s['n']):>4d} "
            f"{s['mean']:>14.6f} {s['std']:>14.6f} "
            f"{s['min']:>14.6f} {s['max']:>14.6f} "
            f"{u['mean']:>12.4f} {u['std']:>12.4f} "
            f"{u['min']:>12.4f} {u['max']:>12.4f} "
            f"{m['mean']:>12.4f} {m['std']:>12.4f} "
            f"{m['min']:>12.4f} {m['max']:>12.4f}"
        )
    return "\n".join(lines)
