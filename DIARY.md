  GPipe:
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_gpipe

  1F1B (default):
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100

  1F1B Interleaved:
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_il1f1b

  ZB-H1:
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_zbh1

  ZB-H2:
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_zbh2

  ZB-V:
  uv run python examples/run_gantt_demo.py --strategy llama70b_tp8_pp4_dp100_zbv

  ## RL env: train & evaluate

  The RL env lives under `simumax/rl/` — `env/` is the Gymnasium env,
  `agents/` holds static baselines (GPipe, 1F1B, their overlap variants,
  ZB-H1, ZB-H2) plus a `PPOAgent` adapter over MaskablePPO checkpoints.
  Both entrypoints below pick up config JSONs from `configs/` by name.

  ### A) Train a PPO agent

  CLI: `examples/train_rl_llama3_70b.py` wraps `simumax.rl.train.train`.

    uv run python examples/train_rl_llama3_70b.py \
        --strategy llama70b_tp8_pp4_dp100 \
        --model llama3-70b \
        --system h100_nvlink \
        --training default \
        --log-dir logs/rl_env \
        --run-name my_run

  Optional: `--disturbance <name>` for stochastic episodes;
  `--reward-mode {utilization,makespan,bubble}`.

  Outputs under `logs/rl_env/<run-name>/`:
    - `tensorboard/` — learning curves (`tensorboard --logdir …`)
    - `checkpoints/ppo_*_steps.zip` — periodic snapshots
    - `best_model/best_model.zip` — best on eval callback
    - `final_model.zip` — last step

  Hyperparameters come from `configs/training/<name>.json` via
  `PPOTrainingConfig` (n_envs, n_steps, lr schedule, net_arch, …).

  ### B) Evaluate agents (static or PPO)

  CLI: `examples/eval_agents.py` wraps `simumax.rl.eval.evaluate`. All
  selected agents share one `SimuMaxBackend` and replay identical
  episodes from `--seed`, so comparisons are apples-to-apples.

  Static baselines only:

    uv run python examples/eval_agents.py \
        --strategy llama70b_tp8_pp4_dp100 \
        --model llama3-70b --system h100_nvlink \
        --agents gpipe gpipe_overlap 1f1b 1f1b_overlap zb_h1 zb_h2 \
        --n-episodes 20

  Mix in a trained PPO checkpoint (label can be anything not clashing
  with a built-in static agent); repeat `--ppo-checkpoint` for more:

    uv run python examples/eval_agents.py \
        --agents zb_h2 ppo_best \
        --ppo-checkpoint ppo_best=logs/rl_env/my_run/best_model/best_model.zip \
        --n-episodes 20 --disturbance default

  Rendering — combine freely:
    - `--render-dir /tmp/gantts` saves `<agent>_ep<NNN>.png` per episode
    - `--display` opens a blocking matplotlib window per episode

  Registry keys (see `simumax/rl/agents/__init__.py`):
    gpipe, gpipe_overlap, 1f1b, 1f1b_overlap, zb_h1, zb_h2
  Classic variants (`gpipe`, `1f1b`) emulate fused-backward semantics by
  gating B on rank g until W on rank g+1 is DONE; `_overlap` variants
  let W and B on neighboring ranks run concurrently (faster, matches
  what the B/W-split env naturally enables).