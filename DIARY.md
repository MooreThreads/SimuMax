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

  uv run python examples/run_gantt_demo.py --strategy llama3_70b_optimal_mfu --model llama3-70b --system h100_nvlink --schedule zb_h2 --disturbance both


  ## RL env: train & evaluate

  The RL env lives under `simumax/rl/` — `env/` is the Gymnasium env,
  `agents/` holds static baselines (GPipe, 1F1B, their overlap variants,
  ZB-H1, ZB-H2) plus a `PPOAgent` adapter over MaskablePPO checkpoints.
  Both entrypoints below pick up config JSONs from `configs/` by name.

  ### A) Train a PPO agent

  CLI: `examples/train_rl.py` wraps `simumax.rl.train.train`.

    uv run python examples/train_rl.py \
        --model llama3-70b \
        --system h100_nvlink \
        --training default \
        --log-dir logs/rl_env \
        --run-name my_run

  `--strategy` defaults to `<model>_optimal_mfu` (with `-`/`.` → `_`);
  override explicitly when you want a non-optimal-MFU strategy.

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
        --model llama3-70b --system h100_nvlink \
        --agents gpipe gpipe_overlap 1f1b 1f1b_overlap zb_h1 zb_h2 \
        --n-episodes 20

  Mix in a trained PPO checkpoint (label can be anything not clashing
  with a built-in static agent); repeat `--ppo-checkpoint` for more:

    uv run python examples/eval_agents.py \
        --agents zb_h2 ppo_best \
        --ppo-checkpoint ppo_best=logs/rl_env/my_run/best_model/best_model.zip \
        --n-episodes 20 --disturbance both

  Rendering — combine freely:
    - `--render-dir /tmp/gantts` saves `<agent>_ep<NNN>.png` per episode
    - `--display` opens a blocking matplotlib window per episode

  Registry keys (see `simumax/rl/agents/__init__.py`):
    gpipe, gpipe_overlap, 1f1b, 1f1b_overlap, zb_h1, zb_h2
  Classic variants (`gpipe`, `1f1b`) emulate fused-backward semantics by
  gating B on rank g until W on rank g+1 is DONE; `_overlap` variants
  let W and B on neighboring ranks run concurrently (faster, matches
  what the B/W-split env naturally enables).


  ## Ablation sweep: MFU + PP utilization across models × schedules

  CLI: `examples/run_ablation_sweep.py`. For each
  `(model, <model>_optimal_mfu strategy, pp_schedule)` cell on a chosen
  system, runs three phases and writes long-format Parquet shards:

    1. `nominal`            — no disturbances, 1 deterministic episode
    2. `baseline_disturbed` — full base disturbance, N episodes (seed-only sweep)
    3. `ablation`           — one-axis-at-a-time sweep over disturbance fields,
                              all other axes zeroed:
       - `seq_len_std`        : 0.2..2.0 × seq_len_mean (10 pts)
       - `op_duration_std`    : 0.01..0.10 step 0.01 (10 pts)
       - `stage_slowdown_prob`: 0.05..0.50 step 0.05 (10 pts)
       - `op_slowdown_prob`   : 0.01..0.10 step 0.01 (10 pts)

  Per-episode disturbance seeds are derived from `--seed` via the same
  Generator-fed scheme as `run_gantt_demo.py --n-episodes`. Each shard is
  the unit of resume: re-running skips shards already on disk.

  Launch (all 16 models × 6 schedules, 100 episodes, 8 workers):

    uv run python examples/run_ablation_sweep.py \
        --output-dir results/h100_nvlink \
        --workers 8

  Useful flags: `--models`, `--schedules` (comma list or `all`),
  `--phases nominal,baseline,ablation`, `--episodes`, `--seed`,
  `--system`, `--base-disturbance`, `--no-resume`.

  Outputs under `<output-dir>/`:
    - `manifest.json`               — config snapshot, git sha, ablation grids
    - `nominal.parquet`
    - `baseline_disturbed.parquet`
    - `ablation_<axis>.parquet`     — one per axis
    - `_shards/`                    — per-(cell, phase[, axis]) shards
    - `failed.json` / `skipped.json` (only on failures / sanity-check skips)

  Row schema highlights: `model, pp_schedule, phase, ablation_axis,
  ablation_value` (multiplier-of-mean for `seq_len_std`, raw value for the
  others), `ablation_value_absolute` (what's actually passed to
  `DisturbanceConfig`), `episode_idx, disturbance_seed, mfu,
  pp_utilization, iter_time_s`, plus `pp_size, micro_batch_num`,
  `seq_len_strategy`, and the base disturbance context (`base_seq_len_mean`,
  `base_stage_slowdown_k`, `base_op_slowdown_k`,
  `base_op_slowdown_max_count`, …) so each file is self-describing for
  re-plotting without joining the manifest.
  what the B/W-split env naturally enables).