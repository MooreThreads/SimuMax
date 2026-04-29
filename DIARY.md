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

    uv run python examples/train_rl.py --model llama3-70b --system h100_nvlink --training default --log-dir logs/rl_env --run-name my_run

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

    uv run python examples/eval_agents.py --model llama3-70b --system h100_nvlink --agents gpipe gpipe_overlap 1f1b 1f1b_overlap zb_h1 zb_h2 --n-episodes 20

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


  ## Search: optimal-MFU strategies per (model, schedule)

  CLI: `examples/search_fit_strategy.py`. For each model in `configs/models/`,
  searches the highest-MFU `(tp, pp, ep, dp, recompute)` combo at a fixed
  (approximate) global batch size within a max world-size budget, and writes
  `configs/strategy/<model>_optimal_mfu_<schedule>[_<disturbance>].json`.
  Filename is always schedule-suffixed; disturbance suffix is added when
  `--disturbance` is set, and a nominal artifact is never overwritten.

  ### A) Nominal (no disturbance)

  Runs the deterministic combo sweep, picks the highest-MFU config that fits
  peak memory under the chosen schedule.

    uv run python examples/search_fit_strategy.py --model llama3-70b --schedule 1f1b

  Loops:
    - over schedules: `--schedule {1f1b, gpipe, zb_h1, zb_h2, interleaved_1f1b, zb_v}`
    - over models: omit `--model` to run all in `configs/models/`.

  Output: `configs/strategy/llama3_70b_optimal_mfu_1f1b.json`.

  ### B) Disturbed (two-phase robust search)

  Adds `--disturbance <name>` and runs Phase A (nominal screen, emits one
  candidate per `(combo, recompute_family)`) + Phase B (re-evaluates the
  top-K candidates under N seeded draws of the disturbance config). Strict
  memory filter: a candidate is kept only if peak memory ≤ accelerator
  budget on *every* seed. Winner = highest mean MFU among survivors. Full
  design in `Disturbed_Search_Plan.md`.

    uv run python examples/search_fit_strategy.py \
        --model llama3-70b --schedule 1f1b \
        --disturbance both --num-seeds 20

  Outputs:
    - `<model>_optimal_mfu_<schedule>_<disturbance>.json` — winner strategy
      (recompute_layer_num = max across seeds, so the saved config doesn't
      OOM on the worst seed it was tested under)
    - `<model>_optimal_mfu_<schedule>_<disturbance>_audit.json` — per-candidate
      stats (mean / std / min / max for MFU and peak_mem), nominal-best,
      Phase A short-list metadata, `git_rev`

  If no candidate fits across all seeds, the strategy JSON is **not** written;
  the audit JSON gets `"selected": null` and `"reason": "no_combo_fits_all_seeds"`.

  Useful flags:
    - `--num-seeds <int>` (default 20) — N seeds per candidate. Bump to 50
      for tight cross-schedule comparisons (smaller SE on close means).
    - `--seed-base <int>` (default 0) — first seed; reproduce or extend a sweep.
    - `--candidate-top-k <int>` (default 30) — Phase A short-list cap.
    - `--candidate-mfu-window <float>` (default 0.10) — keep all
      `(combo, family)` records within this fraction of nominal-best.
    - `--full-seed-eval` — disable Phase B early-exit on first memory failure
      (diagnostic only; otherwise dead candidates short-circuit).
    - `--exact-gbs`, `--gbs <int>`, `--max-world <int>` — override the
      model-size-based defaults.
    - `--output-dir <path>` — override `configs/strategy/`.

  Cost: Phase A ≈ one nominal sweep; Phase B = K × N evaluations (default
  30 × 20 = 600). Order-of-magnitude ~5h per `(model, schedule)` at defaults.
  Parallelize at `(model, schedule)` granularity by launching multiple
  instances; the script itself is single-threaded and mutates `PerfLLM` state.

  Full release set, all schedules, disturbance `both`, 20 seeds:

    for s in 1f1b gpipe zb_h1 zb_h2 interleaved_1f1b zb_v; do
      uv run python examples/search_fit_strategy.py --schedule $s --disturbance both
    done

  ### Caveats

  - The `--candidate-mfu-window 0.10` cutoff is a heuristic. For
    paper-grade headline pairs run one full naive sweep
    (`--candidate-mfu-window 1.0 --candidate-top-k 9999`) on a
    representative `(model, schedule)` to confirm the heuristic agrees
    with exhaustive top-1.
  - Mean MFU under disturbance can be dramatically lower than nominal MFU
    because the saved `recompute_layer_num` is the max-across-seeds (the
    pessimistic-seed level), and that level applies to every step at
    runtime. Robust ≠ typical; disclose `mfu_mean ± mfu_std` from the
    audit, not just the headline.
  - The disturbance config used at search time and at sweep / eval time
    must match; reproducibility also depends on the SimuMax commit
    (sampler changes silently change per-seed draws). The audit records
    `git_rev` for this reason.


  ## Sweep: MFU + PP utilization across models × schedules

  CLI: `examples/run_sweep.py`. For each `(model, pp_schedule)` cell on a
  chosen system, runs one or both of these phases and writes long-format
  Parquet shards. Strategy selection is schedule-aware *and* phase-aware:

    1. `nominal`  — no disturbances, 1 deterministic episode. Uses strategy
                    `<model>_optimal_mfu_<schedule>`.
    2. `baseline` — full base disturbance, N episodes (seed-only sweep).
                    Uses strategy `<model>_optimal_mfu_<schedule>_both`
                    (optimum found under disturbance) and the disturbance
                    config from `--base-disturbance` (default `both`).

  Per-episode disturbance seeds are derived from `--seed` via the same
  Generator-fed scheme as `run_gantt_demo.py --n-episodes`. Each shard is
  the unit of resume: re-running skips shards already on disk. Missing
  strategy files (or `configure()` sanity rejections) land in
  `skipped.json` so a partial strategy rollout doesn't fail the sweep.

  Launch (all 16 models × 6 schedules, both phases, 100 episodes, 8 workers):

    uv run python examples/run_sweep.py \
        --output-dir results/h100_nvlink \
        --workers 8

  Nominal only (e.g. before `_both` strategies are ready):

    uv run python examples/run_sweep.py \
        --output-dir results/h100_nvlink \
        --phases nominal --episodes 1 --workers 8

  Useful flags: `--models`, `--schedules` (comma list or `all`),
  `--phases nominal,baseline`, `--episodes`, `--seed`, `--system`,
  `--base-disturbance`, `--no-resume`.

  Outputs under `<output-dir>/`:
    - `manifest.json`               — config snapshot, git sha, sweep settings
    - `nominal.parquet`
    - `baseline_disturbed.parquet`
    - `_shards/`                    — per-(cell, phase) shards
    - `failed.json` / `skipped.json` (only on failures / sanity-check or
                                      missing-strategy skips)

  Row schema highlights: `model, pp_schedule, phase, episode_idx,
  disturbance_seed, mfu, pp_utilization, iter_time_s`, plus `pp_size,
  micro_batch_num, seq_len_strategy`, and the base disturbance context
  (`base_seq_len_mean`, `base_stage_slowdown_k`, `base_op_slowdown_k`,
  `base_op_slowdown_max_count`, …) so each file is self-describing for
  re-plotting without joining the manifest.

  Note: the legacy `ablation` phase (one-axis-at-a-time disturbance sweep)
  has been removed; recover via git history if needed.


  uv run python scripts/plot.py nominal --results-dir results/h100_nvlink --figs-dir paper/figs

  uv run python scripts/plot.py baseline --results-dir results/h100_nvlink --figs-dir paper/figs

  uv run python scripts/plot.py ablation --results-dir results/h100_nvlink --figs-dir paper/figs

  uv run python scripts/plot.py scatter --results-dir results/h100_nvlink --figs-dir paper/figs

