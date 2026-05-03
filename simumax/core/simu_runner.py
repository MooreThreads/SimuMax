"""Simulator replay orchestration helpers."""

from __future__ import annotations

import os
import time
import pickle
from types import SimpleNamespace

from simumax.core.base_struct import BarrierBackend, SimuContext, SimuSystem, SimuThread
from simumax.core.generate_tracing import process_log_file
from simumax.core.simu_artifacts import (
    append_memory_events_to_trace,
    export_simu_memory_artifacts,
    should_enable_simu_memory_timeline,
)
from simumax.core.simu_memory import SimuMemoryTracker
from simumax.core.transformer.pipeline_schedule import OptimizerSimulator, PpSchedule
from simumax.core.utils import get_pp_stage_representative_rank, get_rank_group


def run_simulation(perf_model, save_path, merge_lanes=True):
    """Run simulator replay for a configured PerfLLM-like object."""

    model_base = perf_model.model_chunk_dict["first_stage_chunk"]
    simu = SimuSystem()
    t0 = time.time()
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "log.log")
    output_json_path = os.path.join(save_path, "tracing_logs.json")
    if os.path.exists(log_path):
        os.remove(log_path)
    ctx = SimuContext(BarrierBackend(), merge_lanes=merge_lanes, log_path=log_path)
    if should_enable_simu_memory_timeline(perf_model.strategy, perf_model._vp_size()):
        ctx.memory_tracker = SimuMemoryTracker()

    if merge_lanes:
        simu_ranks = perf_model.strategy.pp_size
    else:
        simu_ranks = perf_model.strategy.world_size

    for rank_i in range(simu_ranks):
        rank = (
            get_pp_stage_representative_rank(rank_i, perf_model.strategy)
            if merge_lanes
            else rank_i
        )
        thread = SimuThread(rank=rank)

        args = SimpleNamespace(thread_state=thread.thread_state, rank=rank, microbatch=0)
        rank_info = get_rank_group(rank, model_base.strategy)
        if rank_info["pp_rank"] == 0:
            model_base = perf_model.model_chunk_dict["first_stage_chunk"]
            model_name = "first_stage_chunk"
            stage_key = "first_stage_chunk"
        elif rank_info["pp_rank"] < model_base.strategy.pp_size - 1:
            model_base = perf_model.model_chunk_dict["middle_stage_chunk"]
            model_name = "middle_stage_chunk"
            stage_key = "middle_stage_chunk"
        else:
            model_base = perf_model.model_chunk_dict["last_stage_chunk"]
            model_name = "last_stage_chunk"
            stage_key = "last_stage_chunk"

        vp_size = perf_model._vp_size()
        if vp_size > 1 and perf_model.vpp_stage_chunk_names.get(stage_key):
            stage_models = [
                perf_model.vpp_chunk_dict[name]
                for name in perf_model.vpp_stage_chunk_names[stage_key]
            ]
        else:
            stage_models = [model_base]

        pp_simu = PpSchedule(perf_model.strategy, perf_model.system, stage_models)
        if ctx.memory_tracker is not None:
            stage_static_bytes = sum(model.get_model_info().all for model in stage_models)
            ctx.memory_tracker.init_rank(rank, stage_static_bytes)

        thread.job = pp_simu.prefill_batch(args, com_buff=None)

        op_block = OptimizerSimulator(perf_model, model_name)
        op_block.prefill(args, com_buff=None)
        thread.job.append(op_block.prefill_fwd())

        simu.threads.append(thread)

    simu.simu(ctx)

    print("wall time", time.time() - t0)

    process_log_file(log_path, output_json_path)
    if ctx.memory_tracker is not None:
        append_memory_events_to_trace(output_json_path, ctx.memory_tracker)
        export_simu_memory_artifacts(save_path, ctx.memory_tracker, pickle_module=pickle)
