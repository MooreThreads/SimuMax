"""MaskablePPO training helpers for the SimuMax-driven RL env.

Minimal port of ``rlpp/deep_rl/train.py`` — just enough to produce a
working training run for Phase 1. Stochastic env / hparam callbacks
from rlpp are intentionally dropped; add them back when we need more
telemetry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from simumax.rl.env.env import PipelineSchedulingEnv, RLEnvConfig
from simumax.rl.feature_extractor import PipelineFeatureExtractor


def _linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)

    return schedule


@dataclass
class PPOTrainingConfig:
    total_timesteps: int = 500_000
    n_envs: int = 8
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lr_init: float = 3e-4
    lr_final: float = 1e-5
    clip_range_init: float = 0.2
    clip_range_final: float = 0.05
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    features_dim: int = 64
    net_arch_pi: tuple[int, ...] = (128, 128)
    net_arch_vf: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "auto"
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000
    eval_episodes: int = 5

    @classmethod
    def init_from_dict(cls, config_dict: dict[str, Any]) -> "PPOTrainingConfig":
        # JSON has no tuple type; net_arch_* arrive as lists and must be
        # tuples to stay hashable / immutable like the dataclass defaults.
        known = {f.name for f in fields(cls)}
        unknown = set(config_dict) - known
        if unknown:
            raise ValueError(
                f"Unknown PPOTrainingConfig keys in JSON: {sorted(unknown)}"
            )
        normalized = dict(config_dict)
        for key in ("net_arch_pi", "net_arch_vf"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(int(x) for x in normalized[key])
        return cls(**normalized)

    @classmethod
    def init_from_config_file(cls, config_file: str) -> "PPOTrainingConfig":
        with open(config_file, "r", encoding="utf-8") as reader:
            return cls.init_from_dict(json.load(reader))


def make_env_fn(
    env_config: RLEnvConfig,
    rank: int,
    seed: int,
) -> Callable[[], PipelineSchedulingEnv]:
    """Env factory for ``SubprocVecEnv``.

    ``RLEnvConfig`` holds file paths / enum values so it pickles
    cleanly across worker processes. Each worker builds its own
    ``SimuMaxBackend`` (and thus pays ``build()`` + ``_run()`` once).
    """

    def _init() -> PipelineSchedulingEnv:
        env = PipelineSchedulingEnv(env_config=env_config)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed + rank)
    return _init


def train(
    env_config: RLEnvConfig,
    ppo_config: PPOTrainingConfig,
    log_dir: str = "logs",
    run_name: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> MaskablePPO:
    run_name = run_name or f"ppo_seed{ppo_config.seed}"
    run_dir = Path(log_dir) / run_name
    tb_log_dir = str(run_dir / "tensorboard")
    checkpoint_dir = str(run_dir / "checkpoints")
    best_model_dir = str(run_dir / "best_model")
    eval_log_dir = str(run_dir / "eval")
    for d in (checkpoint_dir, best_model_dir, eval_log_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    train_env = VecMonitor(
        SubprocVecEnv(
            [
                make_env_fn(env_config, rank=i, seed=ppo_config.seed)
                for i in range(ppo_config.n_envs)
            ]
        )
    )
    eval_env = VecMonitor(
        SubprocVecEnv(
            [
                make_env_fn(env_config, rank=i, seed=ppo_config.seed + 10_000)
                for i in range(min(4, ppo_config.n_envs))
            ]
        )
    )

    lr = _linear_schedule(ppo_config.lr_init, ppo_config.lr_final)
    clip = _linear_schedule(
        ppo_config.clip_range_init, ppo_config.clip_range_final
    )

    policy_kwargs: dict[str, Any] = {
        "features_extractor_class": PipelineFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": ppo_config.features_dim},
        "net_arch": {
            "pi": list(ppo_config.net_arch_pi),
            "vf": list(ppo_config.net_arch_vf),
        },
    }

    if resume_path is not None:
        model = MaskablePPO.load(
            resume_path,
            env=train_env,
            tensorboard_log=tb_log_dir,
            device=ppo_config.device,
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=lr,
            n_steps=ppo_config.n_steps,
            batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            clip_range=clip,
            ent_coef=ppo_config.ent_coef,
            vf_coef=ppo_config.vf_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            target_kl=ppo_config.target_kl,
            normalize_advantage=ppo_config.normalize_advantage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            seed=ppo_config.seed,
            verbose=1,
            device=ppo_config.device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, ppo_config.checkpoint_freq // ppo_config.n_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo",
    )
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(1, ppo_config.eval_freq // ppo_config.n_envs),
        n_eval_episodes=ppo_config.eval_episodes,
        deterministic=True,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    model.learn(
        total_timesteps=ppo_config.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=resume_path is None,
    )

    model.save(str(run_dir / "final_model"))
    train_env.close()
    eval_env.close()
    return model
