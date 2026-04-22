"""performance model for LLM"""

from abc import ABC, abstractmethod
import os
import math
import json
from copy import deepcopy
from typing import List, Union, Dict, Optional
from sympy import divisors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simumax.core.base_struct import PathDebugContext
from simumax.core.config import DisturbanceConfig, StrategyConfig, SystemConfig, ModelConfig, set_capture_graph_only, TMP_PATH, SIMU_CHECK, SIMU_DEBUG, ENABLE_SIMU_GRAPH
from simumax.core.base_struct import InputOutputInfo, TensorSize, Result
from simumax.core.transformer.language_model import LLMModel, PeakPoint
from simumax.core.graph import SimuONNXGraphBuilder, visualize_with_graphviz
from simumax.core.utils import (
    HumanReadableSize,
    human_readable_bytes,
    convert_final_result_to_human_format,
    merge_dict,
    rm_tmp
)

FIRST_CHUNK = "first_stage_chunk"
MIDDLE_CHUNK = "middle_stage_chunk"
LAST_CHUNK = "last_stage_chunk"

# Substream indices for the four stochastic features; must stay stable so a
# given ``DisturbanceConfig.seed`` reproduces identical draws across runs.
_SEED_SEQ_LEN = 0
_SEED_OP_DURATION = 1
_SEED_OP_SLOWDOWN = 2
_SEED_STAGE_SLOWDOWN = 3
_NUM_DISTURBANCE_STREAMS = 4

class PerfBase(ABC):
    """
    Abstract class for performance model
    """

    dtype_to_element_size = {"fp32": 4, "fp16": 2, "bf16": 2}

    def __init__(self) -> None:
        self.is_configured = False
        self.strategy = None
        self.disturbance = None
        self.model_config = None
        self.system = None
        self.graph = None

        self.debug_points = []
        self.debug_points_last_stage = []

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def _run(self):
        pass

    def _set_strategy_config(self, strategy: StrategyConfig):
        strategy.sanity_check()
        self.strategy = strategy

    def _set_disturbance_config(self, disturbance: DisturbanceConfig):
        disturbance.sanity_check()
        self.disturbance = disturbance

    def _set_model_config(self, model_config: ModelConfig):
        model_config.sanity_check()
        self.model_config = model_config
        self.model_config.maybe_pad_vocab_size(self.strategy.tp_size, True)

    def _set_system_config(self, system: SystemConfig):
        system.sanity_check()
        self.system = system

    @abstractmethod
    def search_max_micro_batch_size(self):
        pass

    def configure(
        self,
        strategy_config: Union[StrategyConfig, str] = None,
        model_config: Union[ModelConfig, str] = None,
        system_config: Union[SystemConfig, str] = None,
        disturbance_config: Union[DisturbanceConfig, str, None] = None,
        debug_points: List[str] = None,
        debug_points_last_stage=None,
    ):
        """
        Configure the performance model, including strategy, model and system config.
        And check the sanity of the configuration.

        ``disturbance_config`` is optional: when omitted an all-defaults
        ``DisturbanceConfig`` is used, which reduces to nominal (no-noise,
        no-slowdown) behaviour.
        """
        if not isinstance(strategy_config, StrategyConfig):
            strategy_config = StrategyConfig.init_from_config_file(strategy_config)
        self._set_strategy_config(strategy_config)

        if disturbance_config is None:
            disturbance_config = DisturbanceConfig()
        elif not isinstance(disturbance_config, DisturbanceConfig):
            disturbance_config = DisturbanceConfig.init_from_config_file(disturbance_config)
        self._set_disturbance_config(disturbance_config)

        if not isinstance(model_config, ModelConfig):
            model_config = ModelConfig.init_from_config_file(model_config)
        self._set_model_config(model_config)


        if not isinstance(system_config, SystemConfig):
            system_config = SystemConfig.init_from_config_file(system_config)
        self._set_system_config(system_config)


        self.debug_points = debug_points if debug_points is not None else []
        self.debug_points_last_stage = (
            debug_points_last_stage if debug_points_last_stage is not None else []
        )

        self._cross_sanity_check()
        self.is_configured = True

    def analysis_pcie_net(self, re_analysis):
        def pcie_decision_helper(size):
            if size <= 2:
                return "intra_node_pcie_2x"
            elif size <= 4:
                return "intra_node_pcie_4x"
            elif size <= 8:
                return "intra_node_pcie_8x"
            else:
                return "inter_node"
            
        world_size = self.strategy.world_size
        tp_size = self.strategy.tp_size
        etp_size = self.strategy.etp_size
        edp_size = self.strategy.edp_size
        ep_size = self.strategy.ep_size
        pp_size = self.strategy.pp_size
        dp_size = self.strategy.dp_size
        num_gpu_per_nodes = self.system.num_per_node    
        
        # 1. analysis pp_net
        if self.strategy.pp_net == "auto" or re_analysis:
            self.strategy.pp_net = pcie_decision_helper(tp_size*dp_size*pp_size)
        
        # 2. analysis ep_net 
        if self.strategy.ep_net == "auto" or re_analysis:
            self.strategy.ep_net = pcie_decision_helper(ep_size * etp_size)

        # 3. analysis tp_net
        if self.strategy.tp_net == "auto" or re_analysis:
            self.strategy.tp_net = pcie_decision_helper(tp_size)    
        # 4. analysis etp_net
        if self.strategy.etp_net == 'auto' or re_analysis:
            self.strategy.etp_net = pcie_decision_helper(etp_size)

        # 5. analysis dp_net
        if self.strategy.dp_net == "auto" or re_analysis:
            self.strategy.dp_net = pcie_decision_helper(tp_size*dp_size)

        # 6. analysis edp_net
        if self.strategy.edp_net == "auto" or re_analysis:
            self.strategy.edp_net = pcie_decision_helper(etp_size * ep_size * edp_size)

    def analysis_high_link_net(self, re_analysis):
        world_size = self.strategy.world_size
        tp_size = self.strategy.tp_size
        etp_size = self.strategy.etp_size
        ep_size = self.strategy.ep_size
        edp_size = self.strategy.edp_size
        pp_size = self.strategy.pp_size
        dp_size = self.strategy.dp_size
        num_gpu_per_nodes = self.system.num_per_node    
        
        # 1. analysis pp_net
        pp_nodes_per_group = world_size // pp_size
        if self.strategy.pp_net == "auto" or re_analysis:
            if pp_nodes_per_group < num_gpu_per_nodes:
                self.strategy.pp_net = "high_intra_node"
            else:
                self.strategy.pp_net = "inter_node"
        
        # 2. analysis ep_net 
        if self.strategy.ep_net == "auto" or re_analysis:
            condition = (ep_size*etp_size <= num_gpu_per_nodes) # When etp *ep exceeds the number of nodes, the communication bandwidth will be reduced, and the default communication between machines will be carried out.
            self.strategy.ep_net = "high_intra_node" if condition else "inter_node"

        # 3. analysis tp_net
        if self.strategy.tp_net == "auto" or re_analysis:
            condition = (tp_size <= num_gpu_per_nodes)
            self.strategy.tp_net = "high_intra_node" if condition else "inter_node"
        # 4. analysis etp_net
        if self.strategy.etp_net == 'auto' or re_analysis:
            condition = etp_size <= num_gpu_per_nodes
            self.strategy.etp_net = "high_intra_node" if condition else "inter_node"

        # 5. analysis dp_net
        if self.strategy.dp_net == "auto" or re_analysis:
            condition = (tp_size * dp_size <= num_gpu_per_nodes)
            self.strategy.dp_net = "high_intra_node" if condition else "inter_node"

        # 6. analysis edp_net
        if self.strategy.edp_net == "auto" or re_analysis:
            condition = etp_size * ep_size * edp_size <= num_gpu_per_nodes
            self.strategy.edp_net = "high_intra_node" if condition else "inter_node"
        
    def analysis_net(self, re_analysis = False):
        if self.system.intra_with_pcie:
            self.analysis_pcie_net(re_analysis)
        else:
            self.analysis_high_link_net(re_analysis)
    
    def capture(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Capture graph...")
        builder = SimuONNXGraphBuilder()
        builder.reset()
        set_capture_graph_only(True)
        self._run()
        set_capture_graph_only(False)
        graph = builder.graph
        graph.export_json(os.path.join(save_path, 'model_graph.json'))
        print("Capture graph done.")
        return graph
    
    def run_estimate(self, capture_graph = False, save_path='./'):
        assert self.is_configured, "should call configure() first"
        self.model_config.maybe_pad_vocab_size(self.strategy.tp_size)
        self.analysis_net(re_analysis = True)
        self.build()
        if capture_graph:
            self.graph = self.capture(save_path)

        # Sample per-microbatch seq_lens exactly once per simulation.
        # Constant array when seq_len_std == 0 (current behaviour).
        if hasattr(self, "_sample_seq_lens"):
            self.seq_lens = self._sample_seq_lens()

        self._run()

        # Sample per-task disturbance multipliers. All three features
        # degrade to no-ops when their primary knob is 0 (default), so
        # this block is invisible for un-configured simulations.
        # Ordering matters: A and C are independent of the scheduler and
        # must be sampled before Feature B's dry run applies them.
        if hasattr(self, "_sample_op_disturbance"):
            self._sample_op_disturbance()
        if hasattr(self, "_sample_stage_disturbance"):
            self._sample_stage_disturbance()
class PerfLLM(PerfBase):

    """Performance model for LLM"""

    def __init__(self) -> None:
        super().__init__()
        self.model_chunk_dict = {}
        self.path_debug_context = PathDebugContext()
        self.path_debug_context_last_stage = PathDebugContext()
        self.pp_state_peak_point = dict(
            first_stage_chunk=dict(),
            middle_stage_chunk=dict(),
            last_stage_chunk=dict()
        )
        # Per-microbatch sampled sequence lengths (np.int64 array, len == micro_batch_num).
        # Populated in run_estimate via _sample_seq_lens(). Constant-valued when
        # disturbance.seq_len_std == 0 (default).
        self.seq_lens: np.ndarray = None
        # Cache for per-(chunk, seq_len_int) timing tuples.
        # Keys: (chunk_name, s_int). Values: (fwd_time, b_time, w_time).
        self._chunk_cost_cache: Dict[tuple, tuple] = {}
        # Disturbance tables (None when disabled). Each is a dict keyed by
        # kind ("F" / "B" / "W") whose value is a 2D ndarray of shape
        # (n_rank_units, mbc) where n_rank_units is pp for physical-rank
        # schedules and V*pp for virtual-stage schedules.
        self.op_noise_mult = None            # Feature A multipliers (float)
        self.op_slowdown_mask = None         # Feature C triggered mask (bool)
        self.stage_slowdown_mult = None      # Feature B multipliers (float)
        # Physical rank selected by Feature B (None when no slowdown fired).
        self._slowed_rank: Optional[int] = None
        # Sampled-event records for auditing (populated alongside the tables).
        self.op_slowdown_records = []
        # Most recent scheduler output, stashed by each calculate_*_bubble
        # method. Retained for downstream inspection.
        self._last_schedules = None
        os.makedirs(TMP_PATH, exist_ok=True)

    def __del__(self):
        try:
            import shutil
            if not SIMU_CHECK:
                if os.path.exists(TMP_PATH):
                    shutil.rmtree(TMP_PATH)
        except Exception as e:
            print(f"Error deleting file: {e}")

    def _disturbance_rng(self, stream_idx: int) -> np.random.Generator:
        """Return a reproducible RNG for one of the four disturbance substreams.

        ``SeedSequence(seed).spawn(N)`` is deterministic, so callers that only
        need one substream can call this without pre-spawning.
        """
        seed = self.disturbance.seed
        ss = np.random.SeedSequence(seed).spawn(_NUM_DISTURBANCE_STREAMS)
        return np.random.default_rng(ss[stream_idx])

    def _sample_seq_lens(self) -> np.ndarray:
        """Sample one sequence length per microbatch.

        Returns a length-``micro_batch_num`` int64 array. When
        ``disturbance.seq_len_std == 0`` (default) the array is constant and
        the simulation degenerates to the original single-seq_len behaviour.

        With non-zero std the values are drawn from
        ``Normal(seq_len_mean, seq_len_std)``, rounded to int, clipped to
        ``[seq_len_min, seq_len_max]`` (``seq_len_max`` defaults to
        ``10 * seq_len_mean`` if unset), and when sequence parallelism is on
        snapped *down* to a multiple of ``tp_size`` (bumped up to ``tp_size``
        if rounding would yield 0). ``seq_len_mean`` falls back to
        ``strategy.seq_len`` when unset.
        """
        strategy = self.strategy
        disturbance = self.disturbance
        mbc = strategy.micro_batch_num
        mean = disturbance.seq_len_mean if disturbance.seq_len_mean is not None else strategy.seq_len
        std = disturbance.seq_len_std
        if std == 0.0:
            return np.full(mbc, int(mean), dtype=np.int64)

        rng = self._disturbance_rng(_SEED_SEQ_LEN)
        raw = rng.normal(mean, std, size=mbc)
        lo = disturbance.seq_len_min if disturbance.seq_len_min is not None else 1
        hi = disturbance.seq_len_max if disturbance.seq_len_max is not None else int(10 * mean)
        clipped = np.clip(np.rint(raw), lo, hi).astype(np.int64)
        if strategy.enable_sequence_parallel and strategy.tp_size > 1:
            tp = strategy.tp_size
            clipped = np.maximum(tp, (clipped // tp) * tp)
        print(
            f"[SimuMax] Variable seq_len: N(mean={mean}, std={std}), "
            f"mbc={mbc} -> "
            f"min={int(clipped.min())}, "
            f"mean={float(clipped.mean()):.1f}, "
            f"max={int(clipped.max())}; values={clipped.tolist()}"
        )
        return clipped

    # ------------------------------------------------------------------
    # Disturbance injection (Features A / B / C)
    # ------------------------------------------------------------------

    def _disturbance_shape(self):
        """Return ``(n_rank_units, kinds)`` matching the per-task timing
        tables consumed by the scheduler for the current ``pp_schedule``.

        ``n_rank_units`` is ``pp`` for physical-rank schedules (1f1b,
        gpipe, zb_h1, zb_h2) and ``V*pp`` for virtual-stage schedules
        (interleaved_1f1b, zb_v). ``kinds`` includes ``"W"`` only for
        schedules that split the backward into B / W (zb_h1, zb_h2, zb_v).
        """
        schedule = self.strategy.pp_schedule
        pp = self.strategy.pp_size
        if schedule == "interleaved_1f1b":
            return self.strategy.interleaving_size * pp, ("F", "B")
        if schedule == "zb_v":
            return 2 * pp, ("F", "B", "W")
        if schedule in ("zb_h1", "zb_h2"):
            return pp, ("F", "B", "W")
        # 1f1b, gpipe
        return pp, ("F", "B")

    def _is_virtual_stage_schedule(self):
        return self.strategy.pp_schedule in ("interleaved_1f1b", "zb_v")

    def _vs_to_rank_owned(self):
        """Return ``(V, vs_to_rank_fn, owned_vs)``.

        ``owned_vs[r]`` is the ordered list of global-vs indices assigned
        to physical rank ``r``. For physical-rank schedules ``V == 1`` and
        ``owned_vs[r] == [r]``.
        """
        schedule = self.strategy.pp_schedule
        pp = self.strategy.pp_size
        if schedule == "interleaved_1f1b":
            V = self.strategy.interleaving_size
            vs_to_rank = lambda gvs: gvs % pp
        elif schedule == "zb_v":
            V = 2
            vs_to_rank = lambda gvs: gvs if gvs < pp else (2 * pp - 1 - gvs)
        else:
            V = 1
            vs_to_rank = lambda gvs: gvs
        owned_vs = [[] for _ in range(pp)]
        for gvs in range(V * pp):
            owned_vs[vs_to_rank(gvs)].append(gvs)
        return V, vs_to_rank, owned_vs

    def _compose_mult(self, kind, idx, m):
        """Composed (A × B × C) multiplier for a single task.

        ``idx`` is the row index into the multiplier tables: physical rank
        for 1f1b/gpipe/zb_h1/zb_h2, or global vs index for
        interleaved_1f1b/zb_v.
        """
        mult = 1.0
        if self.op_noise_mult is not None and kind in self.op_noise_mult:
            mult *= float(self.op_noise_mult[kind][idx, m])
        if self.op_slowdown_mask is not None and kind in self.op_slowdown_mask:
            if self.op_slowdown_mask[kind][idx, m]:
                mult *= self.disturbance.op_slowdown_k
        if self.stage_slowdown_mult is not None and kind in self.stage_slowdown_mult:
            mult *= float(self.stage_slowdown_mult[kind][idx, m])
        return mult

    def _sample_op_disturbance(self):
        """Populate ``op_noise_mult`` (Feature A) and ``op_slowdown_mask``
        (Feature C). Both default to ``None`` when disabled so the timing
        application code can short-circuit.
        """
        strategy = self.strategy
        disturbance = self.disturbance
        n_rank_units, kinds = self._disturbance_shape()
        mbc = strategy.micro_batch_num

        # Feature A — per-task Gaussian multiplier.
        if disturbance.op_duration_std > 0.0:
            rng_a = self._disturbance_rng(_SEED_OP_DURATION)
            mult = {}
            for kind in kinds:
                raw = rng_a.normal(
                    loc=1.0,
                    scale=disturbance.op_duration_std,
                    size=(n_rank_units, mbc),
                )
                mult[kind] = np.clip(
                    raw,
                    disturbance.op_duration_min_factor,
                    disturbance.op_duration_max_factor,
                )
            self.op_noise_mult = mult
            flat = np.concatenate([m.ravel() for m in mult.values()])
            print(
                f"[SimuMax] Op-duration noise: N(1, {disturbance.op_duration_std}), "
                f"shape=({len(kinds)}, {n_rank_units}, {mbc}) -> "
                f"min={float(flat.min()):.3f}, "
                f"mean={float(flat.mean()):.3f}, "
                f"max={float(flat.max()):.3f}"
            )
        else:
            self.op_noise_mult = None

        # Feature C — independent Bernoulli per task.
        self.op_slowdown_records = []
        if disturbance.op_slowdown_prob > 0.0:
            rng_c = self._disturbance_rng(_SEED_OP_SLOWDOWN)
            mask = {}
            for kind in kinds:
                mask[kind] = rng_c.random(size=(n_rank_units, mbc)) < disturbance.op_slowdown_prob
            # Enforce global cap across (kind, rank_unit, mb) in row-major order:
            # stack kinds in declared order, flatten, keep first N True entries.
            if disturbance.op_slowdown_max_count is not None:
                stacked = np.stack([mask[k] for k in kinds], axis=0)
                flat = stacked.reshape(-1)
                kept = 0
                for i in range(flat.size):
                    if flat[i]:
                        if kept < disturbance.op_slowdown_max_count:
                            kept += 1
                        else:
                            flat[i] = False
                stacked = flat.reshape(stacked.shape)
                for i, k in enumerate(kinds):
                    mask[k] = stacked[i]
            self.op_slowdown_mask = mask
            # Record triggered events for logging.
            for k in kinds:
                idxs = np.argwhere(mask[k])
                for (idx, mb) in idxs:
                    self.op_slowdown_records.append(
                        {"kind": k, "rank_unit": int(idx), "mb": int(mb)}
                    )
            print(
                f"[SimuMax] Op-slowdown: p={disturbance.op_slowdown_prob}, "
                f"K={disturbance.op_slowdown_k}, "
                f"triggered={len(self.op_slowdown_records)}"
                + (f" (cap={disturbance.op_slowdown_max_count})"
                   if disturbance.op_slowdown_max_count is not None else "")
            )
        else:
            self.op_slowdown_mask = None

    def _sample_stage_disturbance(self):
        """Populate ``stage_slowdown_mult`` (Feature B) with the stage-wide
        slowdown semantics.

        With probability ``stage_slowdown_prob`` a single physical PP rank —
        i.e. an entire TP/EP group, typically a whole node — is picked
        uniformly at iteration start, and every task mapped onto it (all
        F / B / W across every microbatch, and every virtual stage mapped
        onto that rank for interleaved / zb_v schedules) gets multiplied
        by ``stage_slowdown_k``. At most one slowed stage per iteration.
        Schedule-independent: no dry-run required.
        """
        strategy = self.strategy
        disturbance = self.disturbance
        self._slowed_rank = None
        self.stage_slowdown_mult = None
        if disturbance.stage_slowdown_prob <= 0.0:
            return

        pp = strategy.pp_size
        mbc = strategy.micro_batch_num
        n_rank_units, kinds = self._disturbance_shape()
        K = disturbance.stage_slowdown_k

        rng_b = self._disturbance_rng(_SEED_STAGE_SLOWDOWN)
        if float(rng_b.random()) >= disturbance.stage_slowdown_prob:
            print(
                f"[SimuMax] Stage-slowdown: prob={disturbance.stage_slowdown_prob}, "
                f"K={K} -> no stage slowed"
            )
            return

        slowed_rank = int(rng_b.integers(0, pp))
        self._slowed_rank = slowed_rank

        # Rows of the multiplier table that correspond to the slowed physical
        # rank. For physical-rank schedules that's just [r]; for virtual-stage
        # schedules it's every vs owned by r.
        _V, _vs_to_rank, owned_vs = self._vs_to_rank_owned()
        affected_idx = owned_vs[slowed_rank]

        mult = {k: np.ones((n_rank_units, mbc), dtype=np.float64) for k in kinds}
        for kind in kinds:
            for idx in affected_idx:
                mult[kind][idx, :] = K
        self.stage_slowdown_mult = mult
        print(
            f"[SimuMax] Stage-slowdown: prob={disturbance.stage_slowdown_prob}, "
            f"K={K} -> slowed_rank={slowed_rank}"
        )

    def get_num_layers_to_build(self, config: StrategyConfig, model_conf: ModelConfig, parallel_stage="first") -> int:
        """
        Determine the number of transformer layers to build for the current pipeline stage.
        Args:
            config (TransformerConfig): Configuration object containing transformer model parameters.

        Returns:
            int: The number of layers to be built for the current pipeline stage.
        """
        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):

            assert not (
                config.account_for_embedding_in_pipeline_split
                or config.account_for_loss_in_pipeline_split
            ), " \
            Does not support standalone embedding stage and standalone loss stage with uneven pp"
            # Number of layers to distribute over rest of pipeline stages
            layers_to_distribute = model_conf.layer_num
            # Number of pipeline stages left for distributing transformer layers
            # pipeline_stages_left = parallel_state.get_pipeline_model_parallel_world_size()
            pipeline_stages_left = config.pp_size

            # If the uneven first (last) pipeline stage is enabled, remove the specified number
            # of layers to calculate the number of layers on each middle pipeline stage.
            if config.num_layers_in_first_pipeline_stage is not None:
                layers_to_distribute -= config.num_layers_in_first_pipeline_stage
                pipeline_stages_left -= 1

            if config.num_layers_in_last_pipeline_stage is not None:
                layers_to_distribute -= config.num_layers_in_last_pipeline_stage
                pipeline_stages_left -= 1

            assert (
                layers_to_distribute % pipeline_stages_left == 0
            ), f"With uneven pipelineing the left over layers must be divisible by left over stages, layers_to_distribute={layers_to_distribute}, pipeline_stages_left={pipeline_stages_left}"  
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left

            # If the uneven first (last) pipeline stage is enabled, return the specified number
            # of layers for all virtual pipeline parallel stages within the first (last) pipeline
            # parallel stage.
            if (
                parallel_stage == "first"
                and config.num_layers_in_first_pipeline_stage is not None
            ):
                num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

            if (
                parallel_stage == "last"
                and config.num_layers_in_last_pipeline_stage is not None
            ):
                num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
        else:
            # Include the embedding layer and loss layer into pipeline parallelism partition
            num_layers = model_conf.layer_num
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            assert (
                num_layers % config.pp_size == 0
            ), f"num_layers should be divisible by pipeline_model_parallel_size, but got {num_layers} and {config.pp_size}"
            num_layers_per_pipeline_rank = num_layers // config.pp_size

        # if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        #     # Interleaved pipeline parallelism:
        #     # Number of layers in each model chunk is the number of layers in the stage,
        #     # divided by the number of model chunks in a stage.
        #     # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        #     # layers to stages like (each list is a model chunk):
        #     # Stage 0: [0]  [2]  [4]  [6]
        #     # Stage 1: [1]  [3]  [5]  [7]
        #     # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        #     # layers to stages like (each list is a model chunk):
        #     # Stage 0: [0, 1]  [4, 5]
        #     # Stage 1: [2, 3]  [6, 7]
        #     vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        #     assert (
        #         num_layers_per_pipeline_rank % vp_size == 0
        #     ), "num_layers_per_pipeline_rank should be divisible by vp_size"
        #     num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        #     num_layers_to_build = num_layers_per_virtual_rank

        # else:
            # Non-interleaved pipeline parallelism:
            # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

        # The embedding (or loss) layer cannot function as a standalone transformer layer
        # Reduce the number of layers to construct by 1 on the first (or last) stage if the
        # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
        if parallel_stage == "first" and config.account_for_embedding_in_pipeline_split:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

        if parallel_stage == "last" and config.account_for_loss_in_pipeline_split:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"
        # if parallel_stage == "middle":
        #     num_layers_to_build += sum([config.account_for_embedding_in_pipeline_split, config.account_for_loss_in_pipeline_split])
        if SIMU_DEBUG:
            print(f"Building {num_layers_to_build} layers for {parallel_stage} stage")
        return num_layers_to_build

    def build(self):
        """
        build first stage model chunk and last stage model chunk
        """
        self.strategy.sanity_check()
        self.model_chunk_dict:Dict[str, LLMModel] = {}

        # Build First Stage Model Chunk
        # Only consider the even divide case fow now
        # layer_num = self.model_config.layer_num // self.strategy.pp_size
        remian_dense_layers=self.model_config.dense_layers
        dense_layers_i = max(0, remian_dense_layers)
        remian_dense_layers -= dense_layers_i

        layer_num_first = self.get_num_layers_to_build(self.strategy, self.model_config, "first")
        if self.strategy.pp_size > 1:
            self.model_chunk_dict["first_stage_chunk"] = LLMModel(
                layer_num=layer_num_first,
                preprocess=True,
                postprocess=False,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_first_pp_stage"
            )
        else:
            self.model_chunk_dict["first_stage_chunk"] = LLMModel(
                layer_num=layer_num_first,
                preprocess=True,
                postprocess=True,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
                dense_layers=dense_layers_i,
                # specific_name="llm_first_stage_chunk"
            )
        if self.strategy.pp_size > 2:
            layer_num_middle = self.get_num_layers_to_build(self.strategy, self.model_config, "middle")
            dense_layers_i = max(0, remian_dense_layers)
            remian_dense_layers -= dense_layers_i*(self.strategy.pp_size-2)
            self.model_chunk_dict["middle_stage_chunk"] = LLMModel(
                layer_num=layer_num_middle,
                preprocess=False,
                postprocess=False,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_middle_pp_stage"
            )

        # # Build Last Stage Model Chunk
        if self.strategy.pp_size > 1:
            layer_num_last = self.get_num_layers_to_build(self.strategy, self.model_config, "last")
            dense_layers_i = max(0, remian_dense_layers)
            self.model_chunk_dict["last_stage_chunk"] = LLMModel(
                layer_num=layer_num_last,
                preprocess=False,
                postprocess=True,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_last_pp_stage"
            )

    def _cross_sanity_check(self) -> bool:
        # assert (
        #     self.model_config.layer_num % self.strategy.pp_size == 0
        # ), "layer num should be divisible by pp_size"

        assert self.debug_points is None or isinstance(
            self.debug_points, list
        ), "debug_points should be a list"
        assert (
            self.model_config.expert_num % self.strategy.ep_size == 0
        ), f"expert num {self.model_config.expert_num} should be divisible by ep_size {self.strategy.ep_size}"  # pylint: disable=line-too-long

    @property
    def global_hidden_states_size(self):
        hidden_states_size = (
            self.strategy.global_batch_size
            * self.strategy.seq_len
            * self.model_config.hidden_size
        )
        return hidden_states_size

    @property
    def micro_hidden_states_size(self):
        hidden_states_size = (
            self.strategy.micro_batch_size
            * self.strategy.seq_len
            * self.model_config.hidden_size
        )
        return hidden_states_size

    def _compute_bubble_time(self, fwd_bwd_time):
        # TODO: support uneven divide && interleaving
        bubble_time = fwd_bwd_time * (self.strategy.pp_size - 1)
        return bubble_time
    def _compute_optim_time(self, model_name):
        # we use the chunk weight accessed time as the optim time
        result = {"optim_time": 0, "optim_exposed_time": 0}
        model_info = self.model_chunk_dict[model_name].get_model_info()
        state_weight_bytes = model_info.all_state_bytes

        use_megatron = True
        if use_megatron:
            # refer to megatron-lm, TODO(sherry): support fp8
            zero_grad_buffer_time = self.system.compute_mem_access_time('default', model_info.all_grad_bytes)

            l2_norm_before_reduce_time = self.system.compute_mem_access_time('default', model_info.all_grad_bytes) # read grads
            mul_before_reduce_time = self.system.compute_mem_access_time('default', 2 * model_info.all_grad_bytes) if self.strategy.dp_size > 1 else 0# read grads and write grads

            grads_chunk_after_reduce_time = state_weight_bytes / 6 if self.strategy.grad_reduce_in_bf16 else state_weight_bytes / 3
            weight_bytes = state_weight_bytes / 3 
            l2_norm_after_reduce_time = self.system.compute_mem_access_time('default', grads_chunk_after_reduce_time) # read grads chunk
            grads_clip_after_reduce_time = self.system.compute_mem_access_time('default', 2 * grads_chunk_after_reduce_time) # read and write grad_chunk, when l2 norm is scaler

            adam_time = self.system.compute_mem_access_time('default',
                grads_chunk_after_reduce_time + 3 * state_weight_bytes # read and write m/w/v, read grad_chunk
            )
            copy_main_params_to_model_params_time = self.system.compute_mem_access_time('default', weight_bytes + 0.5 * weight_bytes) # fp32 -> bf16

            result['zero_grad_buffer_time'] = zero_grad_buffer_time
            result['l2_norm_before_reduce_time'] = l2_norm_before_reduce_time
            result['mul_before_reduce_time'] = mul_before_reduce_time
            result['l2_norm_after_reduce_time'] = l2_norm_after_reduce_time
            result['grads_clip_after_reduce_time'] = grads_clip_after_reduce_time
            result['adam_time'] = adam_time
            result['copy_main_params_to_model_params_time'] = copy_main_params_to_model_params_time
            optim_time = sum(result.values())
            result['optim_time'] = optim_time
            result['optim_exposed_time'] = optim_time
            return result
        else:
            chunk_weight_accessed_time = 3 * state_weight_bytes # why 3x?
            optim_time = self.system.compute_mem_access_time(chunk_weight_accessed_time)
            optim_exposed_time = adam_time  # no overlap for now
            result["optim_time"] = adam_time
            result["optim_exposed_time"] = optim_exposed_time 
            return result

    def _compute_dp_time(self, model_name):
        # TODO: support overlap
        use_megatron = True
    
        def compute_dp_helper(rs_comm_size, gather_comm_size, dp_net, dp_size, dp_group):
            result = {"dp_comm_time": 0, "dp_comm_exposed_time": 0}
            dp_comm_time = 0
            bucket_size = (
                max(40000000, 1000000 * dp_size) * 4
            )  # consider bucket size

            num_reduce_bucket = (rs_comm_size - 1) // bucket_size + 1  
            num_gather_bucket = (gather_comm_size - 1) // bucket_size + 1
            if self.model_config.model_type == "moe" and use_megatron:
                num_gather_bucket *= 2 
            details = {}
            if self.strategy.zero_state >= 1:
                reduce_scatter_time = num_reduce_bucket * self.system.compute_net_op_time(
                    "reduce_scatter",
                    bucket_size,
                    comm_num=dp_size,
                    net=dp_net,
                    comm_stage=dp_group, 
                    strategy=self.strategy
                )
                all_gather_time = num_gather_bucket * self.system.compute_net_op_time(
                    "all_gather", 
                    bucket_size, 
                    comm_num=dp_size, 
                    net=dp_net,
                    comm_stage=dp_group,
                    strategy=self.strategy
                )
                dp_comm_time += all_gather_time + reduce_scatter_time
                details['reduce_scatter_time'] = reduce_scatter_time
                details['all_gather_time'] = all_gather_time
            else:
                dp_comm_time += num_reduce_bucket * self.system.compute_net_op_time(
                    "all_reduce", 
                    bucket_size, 
                    comm_num=dp_size, 
                    net=dp_net,
                    comm_stage=dp_group,
                    strategy=self.strategy
                )

            dp_comm_exposed_time = dp_comm_time  # no overlap for now
            result['dp_comm_rs_size'] = rs_comm_size if dp_size > 1 else 0
            result['dp_comm_ag_size'] = gather_comm_size if dp_size > 1 else 0
            result['dp_comm_num_gather'] = 2 if self.model_config.model_type == "moe" and use_megatron else 1
            result["dp_comm_time"] = dp_comm_time
            result["dp_comm_exposed_time"] = dp_comm_exposed_time
            if details:
                result['details'] = details
            return result
        
        model_info = self.model_chunk_dict[model_name].get_model_info()

        # dense
        rs_comm_size = model_info.dense_grad_bytes/2  if self.strategy.grad_reduce_in_bf16 else model_info.dense_grad_bytes 
        gather_comm_size = model_info.dense_grad_bytes / 4 * self.dtype_to_element_size[self.strategy.dtype] 
        
        # moe
        moe_rs_comm_size = model_info.moe_grad_bytes / 2 if self.strategy.grad_reduce_in_bf16 else model_info.moe_grad_bytes
        moe_gather_comm_size = model_info.moe_grad_bytes / 4 * self.dtype_to_element_size[self.strategy.dtype]

        dense_dp_result = compute_dp_helper(rs_comm_size, gather_comm_size, self.strategy.dp_net, self.strategy.dp_size, dp_group="dp")
        moe_dp_result = compute_dp_helper(moe_rs_comm_size, moe_gather_comm_size, self.strategy.edp_net, self.strategy.edp_size, dp_group="edp")
        all_result = {
            'dp_comm_exposed_time': dense_dp_result['dp_comm_exposed_time'] + moe_dp_result['dp_comm_exposed_time'],
            'dense': dense_dp_result,
            'moe': moe_dp_result,
        }
        return all_result

    def _analysis_mem_impl(
        self,
        micro_batch_num,
        model_name=FIRST_CHUNK,
    ):
        result = {}
        model_info = self.model_chunk_dict[model_name].get_model_info()

        #-------------------------- 0. set base info --------------------------
        result["micro_batch_num"] = self.strategy.micro_batch_num
        result["micro_batch_size"] = self.strategy.micro_batch_size
        result["cached_micro_batch_num"] = micro_batch_num -1
        result['parallel_config'] = {
            'parallelism': self.strategy.parallelism,
            'fp8': self.strategy.fp8,
            'recompute_status':{
                'layer_num': self.model_config.layer_num,
                'actual_layer_num': self.model_chunk_dict['first_stage_chunk'].layer_num,
                'recompute_layer': self.strategy.recompute_layer_num,
                'recompute_recompute_granularity': self.strategy.recompute_granularity,
            }
        }
        if self.strategy.grad_reduce_in_bf16:
                model_info.dense_grad_bytes = model_info.dense_grad_bytes/2 # TODO(sherry): this is a hack to make it work, need to fix
                model_info.moe_grad_bytes = model_info.moe_grad_bytes/2

        #-------------------------- 1. compute model mem --------------------------
        dense_model_mem = dict(
            all_mem = model_info.dense_weight_bytes + model_info.dense_grad_bytes + model_info.dense_state_bytes,
            detail = dict(
                weight_bytes = model_info.dense_weight_bytes,
                grad_bytes = model_info.dense_grad_bytes,
                state_bytes = model_info.dense_state_bytes
            )
        )
        moe_model_mem = dict(
            all_mem = model_info.moe_weight_bytes + model_info.moe_grad_bytes + model_info.moe_state_bytes,
            detail = dict(
                weight_bytes = model_info.moe_weight_bytes,
                grad_bytes = model_info.moe_grad_bytes,
                state_bytes = model_info.moe_state_bytes
            )
        )
        result["model_mem"] = dense_model_mem['all_mem'] + moe_model_mem['all_mem']
        result["model_mem_detail"] = dict(
            dense = dense_model_mem,
            moe = moe_model_mem
        )
        # result["with_recompute"] = self.strategy.enable_recompute
        
        #-------------------------- 2. compute peak activation in 1F1B--------------------------
        cur_act_info:PeakPoint = self.pp_state_peak_point[model_name]
        result["fwd_activation_cache_per_micro_batch"] = f"{cur_act_info.activation_mem_cache/1024/1024/1024:.4f} GB"
        result["peak_activation_mem_in_1F1B"] = cur_act_info.peak_mem
        model_mem = result["model_mem"]

        #-------------------------- 3. compute total peak peak mem --------------------------
        # result["fwd_peak_allocated_mem"] = cur_act_info.fwd_peak_mem
        # result["bwd_peak_allocated_mem"] = max(cur_act_info.bwd_peak_mem, cur_act_info.recomp_fwd_peak_mem, cur_act_info.recomp_bwd_peak_mem)
        result["peak_mem"] = (
            model_mem + 
            (micro_batch_num-1) * cur_act_info.activation_mem_cache +
            result["peak_activation_mem_in_1F1B"]
        )
        result["peak_mem_with_reserved"] = result["peak_mem"]/self.strategy.mem_factor
        
        result["memory_reserved_ratio"] = str(self.strategy.mem_factor)
        result["peak_path"] = f"{cur_act_info.peak_path}, stage=[{cur_act_info.peak_stage}]"
        # Convert to human format
        convert_final_result_to_human_format(result)
        return result

    def analysis_mem(self):
        """Based the simulation result, analyze the memory usage"""
        if self.strategy.pp_size == 1:
            result = self._analysis_mem_impl(
                micro_batch_num=1, model_name=FIRST_CHUNK
            )
        elif self.strategy.pp_size == 2:
            # add more condition here to ensure the correctness the order of pp stage in result
            result = {"first_stage": {}, "last_stage": {}}
            result["first_stage"] = self._analysis_mem_impl(
                micro_batch_num=self.strategy.pp_size, model_name=FIRST_CHUNK
            ) # The 0th stage, here should be the corresponding 1F1B, the ac of stage1 needs to hold pp_size mbs (micro batch size)
            result["last_stage"] = self._analysis_mem_impl(
                micro_batch_num=1, model_name=LAST_CHUNK
            )
        elif self.strategy.pp_size>2: 
            result = {"first_stage": {}, "middle_stage": {},"last_stage": {}}
            result["first_stage"] = self._analysis_mem_impl(
                micro_batch_num=self.strategy.pp_size, model_name=FIRST_CHUNK
            ) # The 0th stage, here should be the corresponding 1F1B, the ac of stage1 needs to hold pp_size mbs (micro batch size)
            result["middle_stage"] = self._analysis_mem_impl(
                micro_batch_num=self.strategy.pp_size-1, model_name=MIDDLE_CHUNK
            ) # The first stage, here should be the corresponding 1F1B, the ac of stage2 needs to hold pp_size-1 mbs (micro batch size)
            result["last_stage"] = self._analysis_mem_impl(
                micro_batch_num=1, model_name=LAST_CHUNK
            )
        return Result(result)

    def _analysis_single_batch_cost_impl(  # pylint: disable=invalid-name
        self, enable_recompute=True, model_name="first_stage_chunk"
    ):
        # compute time = module fwd time + module bwd time + update time
        # comm time = tp time + pp_time + dp_time
        result = {"compute_info": None, "cost_info": None}
        cost_batch_stat = {
            "fwd_compute_time": 0,
            "bwd_compute_time": 0,
            "recompute_compute_time": 0,
            "fwd_net_time": 0,
            "bwd_net_time": 0,
            "recompute_net_time": 0,
            "fwd_net_exposed_time": 0,
            "bwd_net_exposed_time": 0,
            "recompute_net_exposed_time": 0,
            "fwd_time": 0,
            "bwd_time": 0,
            "recompute_time": 0,
        }
        compute_batch_stat = {
            "fwd_flops": 0,
            "recompute_flops": 0,
            "bwd_flops": 0,
            "fwd_accessed_mem": 0,
            "recompute_accessed_mem": 0,
            "bwd_accessed_mem": 0,
        }

        compute_info = self.model_chunk_dict[model_name].get_compute_info()
        cost_info = self.model_chunk_dict[model_name].get_cost_info()

        cost_batch_stat["fwd_time"] = cost_info.fwd_time
        cost_batch_stat["bwd_time"] = cost_info.bwd_time
        cost_batch_stat["recompute_time"] = (
            cost_info.recompute_time if enable_recompute else 0
        )
        cost_batch_stat["fwd_compute_time"] = cost_info.fwd_compute_time
        cost_batch_stat["bwd_compute_time"] = cost_info.bwd_compute_time
        cost_batch_stat["recompute_compute_time"] = cost_info.recompute_compute_time

        cost_batch_stat["fwd_net_time"] = cost_info.fwd_net_time
        cost_batch_stat["bwd_net_time"] = cost_info.bwd_net_time
        cost_batch_stat["recompute_net_time"] = cost_info.recompute_net_time

        cost_batch_stat["fwd_net_exposed_time"] = cost_info.fwd_net_exposed_time
        cost_batch_stat["bwd_net_exposed_time"] = cost_info.bwd_net_exposed_time
        cost_batch_stat["recompute_net_exposed_time"] = (
            cost_info.recompute_net_exposed_time
        )
        result["cost_info"] = cost_batch_stat

        compute_batch_stat["fwd_flops"] = compute_info.fwd_flops
        compute_batch_stat["recompute_flops"] = (
            compute_info.recompute_flops if enable_recompute else 0
        )
        compute_batch_stat["bwd_flops"] = compute_info.bwd_flops
        compute_batch_stat["fwd_accessed_mem"] = compute_info.fwd_accessed_mem
        compute_batch_stat["recompute_accessed_mem"] = (
            compute_info.recompute_accessed_mem if enable_recompute else 0
        )
        compute_batch_stat["bwd_accessed_mem"] = compute_info.bwd_accessed_mem
        result["compute_info"] = compute_batch_stat
        return result

    def _analysis_gbs_compute_time(self, batch_stat, model_name):
        result = {}
        micro_batch_num = self.strategy.micro_batch_num
        # skip_ckpt_micro_batch_num = self.strategy.skip_ckpt_micro_batch_num
        result["batch_compute_stat"] = batch_stat

        result["fwd_compute_time"] = (
            batch_stat["cost_info"]["fwd_compute_time"] * micro_batch_num
        )
        result["recompute_time"] = (
            batch_stat["cost_info"]["recompute_compute_time"] * micro_batch_num
        )
        result["bwd_compute_time"] = (
            batch_stat["cost_info"]["bwd_compute_time"] * micro_batch_num
        )
        optim_result = self._compute_optim_time(model_name)
        result["optim_time"] = optim_result
        result["fwd_flops"] = batch_stat["compute_info"]["fwd_flops"] * micro_batch_num
        result["recompute_flops"] = (
            batch_stat["compute_info"]["recompute_flops"] * micro_batch_num
        )
        result["bwd_flops"] = batch_stat["compute_info"]["bwd_flops"] * micro_batch_num
        result["model_flops"] = result["fwd_flops"] + result["bwd_flops"]
        return result

    def _analysis_gbs_comm_time(self, batch_stat, model_name):
        result = {}
        micro_batch_num = self.strategy.micro_batch_num
        dp_comm_result = self._compute_dp_time(model_name)
        # TODO: add ckpt bubble and add strategy extra comm time, # e.g sp grad reduce
        intra_exposed_time = sum(  # pylint: disable=invalid-name
            batch_stat["cost_info"][k]
            for k in ["fwd_net_time", "bwd_net_time", "recompute_net_time"]
        )
        if self.strategy.pp_size > 1:
            pp_comm_size = (
                self.micro_hidden_states_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            pp_comm_size = (
                pp_comm_size / self.strategy.tp_size
                if self.strategy.enable_sequence_parallel
                else pp_comm_size
            )
            inter_exposed_time_per_batch = 2 * 2 * self.system.compute_net_op_time(
                "p2p", pp_comm_size, 2, net=self.strategy.pp_net, comm_stage="pp"
            )  # 2 p2p, 2 to fwd and bwd
        else:
            inter_exposed_time_per_batch = 0

        inter_exposed_time = inter_exposed_time_per_batch * micro_batch_num
        result["dp_comm_time"] = dp_comm_result
        # Now we don't consider the mix of recompute and non-recompute
        intra_exposed_time_per_batch = intra_exposed_time
        intra_exposed_time = intra_exposed_time_per_batch * micro_batch_num
        
        result["intra_comm_time"] = {
            "intra_exposed_time_per_batch": intra_exposed_time_per_batch,
            "intra_exposed_time": intra_exposed_time,
        }
        result["inter_comm_time"] = {
            "inter_exposed_time_per_batch": inter_exposed_time_per_batch,
            "inter_exposed_time": inter_exposed_time,
        }
        return result
    
    def calculate_1f1b_bubble(self, pp, mbc, forward_times, backward_times, draw=False, output_path=None):
        # forward_times / backward_times are [pp][mbc] 2-D lists.
        schedules = [[] for _ in range(pp)]
        fwd_ready = [[0]  for _ in range(pp)]
        bwd_ready = [[0]  for _ in range(pp)]

        for step in range(mbc):
            for rank in range(pp):
                warmup_step = pp-1-rank
                if step<warmup_step:
                    "F"
                    current_time = schedules[rank][-1][4] if schedules[rank] else 0
                    prev_fwd = fwd_ready[rank - 1][-1] if rank > 0 else 0
                    start_time = max(current_time, prev_fwd)
                    fwd_mb = len(fwd_ready[rank]) - 1
                    duration = forward_times[rank][fwd_mb]
                    schedules[rank].append(('F', fwd_mb, start_time, duration, start_time+duration))
                    fwd_ready[rank].append(start_time + duration)
                else:
                    "F-B"
                    current_time = schedules[rank][-1][4] if schedules[rank] else 0
                    prev_fwd = fwd_ready[rank - 1][-1] if rank > 0 else 0
                    start_time = max(current_time, prev_fwd)
                    fwd_mb = len(fwd_ready[rank]) - 1
                    duration = forward_times[rank][fwd_mb]
                    schedules[rank].append(('F', fwd_mb, start_time, duration, start_time+duration))
                    fwd_ready[rank].append(start_time + duration)

                    current_time = schedules[rank][-1][4]
                    next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                    start_time = max(current_time, next_bwd)
                    bwd_mb = len(bwd_ready[rank]) - 1
                    duration = backward_times[rank][bwd_mb]
                    schedules[rank].append(('B', bwd_mb, start_time, duration, start_time+duration))
                    bwd_ready[rank].append(start_time + duration)


        for step in range(pp-1,-1,-1):
            for rank in range(step):
                "B"
                current_time = schedules[rank][-1][4]
                next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                start_time = max(current_time, next_bwd)
                bwd_mb = len(bwd_ready[rank]) - 1
                duration = backward_times[rank][bwd_mb]
                schedules[rank].append(('B', bwd_mb, start_time, duration, start_time+duration))
                bwd_ready[rank].append(start_time + duration)

        max_time = max([s[-1][4] for s in schedules])
        self._last_schedules = schedules

        # f_b_time = [x+y for x,y in zip(forward_times, backward_times)]

        # idx = np.argmax(f_b_time)
        # bubble = sum(f_b_time)+sum(forward_times[idx+1:])+sum(backward_times[idx+1:]) - (pp-idx)*(f_b_time[idx])
        # all_time = bubble + mbc*f_b_time[idx]
        if draw:
            # Visualize the schedule
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = {'F': '#6b8ec9', 'B': '#6db5b5'}

            for rank, tasks in enumerate(schedules):
                for task_type, mb, start, duration, end in tasks:
                    ax.barh(y=pp - 1 - rank, width=duration, left=start,
                            height=0.6, color=colors[task_type], edgecolor='black')
                    ax.text(start + duration / 2, pp - 1 - rank, f'{task_type}{mb + 1}',
                            va='center', ha='center', fontsize=9, color='black')

            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Stage {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(f"Corrected 1F1B Pipeline Execution Timeline (pp={pp}, mbc={mbc})")
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename("1f1b"))
            plt.close(fig)

        return max_time

    def _compute_single_batch_fwd_bwd_time(self, model_name, chunk = False):
            if self.strategy.pp_size > 1:
                pp_comm_size = (
                    self.micro_hidden_states_size
                    * self.dtype_to_element_size[self.strategy.dtype]
                )
                pp_comm_size = (
                    pp_comm_size / self.strategy.tp_size
                    if self.strategy.enable_sequence_parallel
                    else pp_comm_size
                )
                pp_time = 2 * self.system.compute_net_op_time(
                    "p2p", pp_comm_size, 2, net=self.strategy.pp_net
                )  # 2 p2p, fwd/bwd each
            else:
                pp_time = 0

            cost_info = self.model_chunk_dict[model_name].get_cost_info()

            fwd_chunk_time = (cost_info.fwd_compute_time +
                                cost_info.fwd_net_time +
                                pp_time)
            bwd_chunk_time = (cost_info.bwd_compute_time +
                                cost_info.bwd_net_time +
                                cost_info.recompute_compute_time +
                                cost_info.recompute_net_time +
                                pp_time)
            return (fwd_chunk_time, bwd_chunk_time) if not chunk else fwd_chunk_time + bwd_chunk_time

    def _compute_single_batch_fwd_b_w_time(self, model_name):
        """Return per-chunk (F, B-for-input, B-for-weight) times for ZB-style
        schedules. B covers the activation-gradient path (on the critical path,
        drives the next rank's B) plus recompute; W covers the weight-gradient
        path (deferrable). F+B+W equals the fused (F, B) sum."""
        if self.strategy.pp_size > 1:
            pp_comm_size = (
                self.micro_hidden_states_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            pp_comm_size = (
                pp_comm_size / self.strategy.tp_size
                if self.strategy.enable_sequence_parallel
                else pp_comm_size
            )
            pp_time = 2 * self.system.compute_net_op_time(
                "p2p", pp_comm_size, 2, net=self.strategy.pp_net
            )
        else:
            pp_time = 0

        cost_info = self.model_chunk_dict[model_name].get_cost_info()

        fwd_chunk_time = (cost_info.fwd_compute_time +
                          cost_info.fwd_net_time +
                          pp_time)
        b_input_chunk_time = (cost_info.bwd_grad_act_time +
                              cost_info.bwd_grad_act_net_time +
                              cost_info.recompute_compute_time +
                              cost_info.recompute_net_time +
                              pp_time)
        w_chunk_time = (cost_info.bwd_grad_w_time +
                        cost_info.bwd_grad_w_net_time)
        return fwd_chunk_time, b_input_chunk_time, w_chunk_time

    # ------------------------------------------------------------------
    # Per-(chunk, seq_len) timing cache (variable sequence-length support)
    # ------------------------------------------------------------------

    def _pp_time_for(self, s: int) -> float:
        """PP p2p send/recv time (fwd+bwd, 2 hops) for a microbatch of seq_len ``s``."""
        if self.strategy.pp_size <= 1:
            return 0.0
        pp_comm_size = (
            self.strategy.micro_batch_size
            * int(s)
            * self.model_config.hidden_size
            * self.dtype_to_element_size[self.strategy.dtype]
        )
        if self.strategy.enable_sequence_parallel:
            pp_comm_size = pp_comm_size / self.strategy.tp_size
        return 2 * self.system.compute_net_op_time(
            "p2p", pp_comm_size, 2, net=self.strategy.pp_net
        )

    def _populate_chunk_cost_cache(self, chunks, seq_lens) -> None:
        """Populate ``self._chunk_cost_cache`` with one entry per
        ``(chunk_name, s_int)`` in ``chunks x set(seq_lens)``.

        Each entry is a dict of compute/net time components pulled from the
        chunk's ``_cost_info`` after invoking it with the given seq_len. After
        populating all entries for a chunk, the chunk is re-invoked once at
        ``_nominal_seq_len_for_mem()`` so subsequent analysis paths that read
        the chunk state directly (not via the cache) see the nominal state.
        """
        unique_s = sorted({int(s) for s in seq_lens})
        nominal = self._nominal_seq_len_for_mem()
        for chunk_name in chunks:
            ctx = PathDebugContext(
                point_datas={}, point_datas_with_recomp={}, path_list=[]
            )
            for s in unique_s:
                key = (chunk_name, s)
                if key in self._chunk_cost_cache:
                    continue
                if s != nominal:
                    self._invoke_chunk(chunk_name, s, ctx)
                # else: chunk state is already at nominal (set by _run).
                ci = self.model_chunk_dict[chunk_name].get_cost_info()
                self._chunk_cost_cache[key] = {
                    "fwd_compute_time": ci.fwd_compute_time,
                    "fwd_net_time": ci.fwd_net_time,
                    "bwd_compute_time": ci.bwd_compute_time,
                    "bwd_net_time": ci.bwd_net_time,
                    "bwd_grad_act_time": ci.bwd_grad_act_time,
                    "bwd_grad_act_net_time": ci.bwd_grad_act_net_time,
                    "bwd_grad_w_time": ci.bwd_grad_w_time,
                    "bwd_grad_w_net_time": ci.bwd_grad_w_net_time,
                    "recompute_compute_time": ci.recompute_compute_time,
                    "recompute_net_time": ci.recompute_net_time,
                }
            # Restore the chunk to nominal seq_len so later analysis reads
            # the expected state.
            if unique_s and unique_s[-1] != nominal or (len(unique_s) > 1):
                self._invoke_chunk(chunk_name, nominal, ctx)

    def _chunk_fwd_bwd_at(self, chunk_name: str, s: int):
        """Return (fwd_time, bwd_time) for a microbatch of seq_len ``s`` on ``chunk_name``."""
        entry = self._chunk_cost_cache[(chunk_name, int(s))]
        pp_time = self._pp_time_for(s)
        fwd = entry["fwd_compute_time"] + entry["fwd_net_time"] + pp_time
        bwd = (
            entry["bwd_compute_time"]
            + entry["bwd_net_time"]
            + entry["recompute_compute_time"]
            + entry["recompute_net_time"]
            + pp_time
        )
        return fwd, bwd

    def _chunk_fwd_b_w_at(self, chunk_name: str, s: int):
        """Return (fwd, B, W) for a microbatch of seq_len ``s`` on ``chunk_name``.

        B covers the activation-gradient critical path + recompute; W covers
        the deferrable weight-gradient path. Sum equals fwd+bwd from
        :meth:`_chunk_fwd_bwd_at`.
        """
        entry = self._chunk_cost_cache[(chunk_name, int(s))]
        pp_time = self._pp_time_for(s)
        fwd = entry["fwd_compute_time"] + entry["fwd_net_time"] + pp_time
        b = (
            entry["bwd_grad_act_time"]
            + entry["bwd_grad_act_net_time"]
            + entry["recompute_compute_time"]
            + entry["recompute_net_time"]
            + pp_time
        )
        w = entry["bwd_grad_w_time"] + entry["bwd_grad_w_net_time"]
        return fwd, b, w

    def _chunk_names_for_pp(self):
        """The set of chunk names that actually exist for the current PP size."""
        pp = self.strategy.pp_size
        chunks = [FIRST_CHUNK]
        if pp > 2:
            chunks.append(MIDDLE_CHUNK)
        if pp > 1:
            chunks.append(LAST_CHUNK)
        return chunks

    def _calculate_zb_bubble(self, pp, mbc, forward_times, b_times, w_times,
                             n_warmup_fn, mem_limit, schedule_name,
                             draw=False, output_path=None):
        """Shared core of the ZB-family schedulers (ZB-H1 / ZB-H2).

        Backward is split into B (activation gradient) and W (weight
        gradient). Under the F-W memory model (an activation is held
        until both its B and W complete), each rank's schedule is:

        1. Warmup: ``n_warmup_fn(g)`` forward passes.
        2. Accumulation: ``(B, F)`` pairs to fill live activations up to
           ``mem_limit``. Downstream ranks (small warmup) grow via this
           phase so Bs fire early and unblock the backward chain.
        3. Steady state: ``(B, W, F)`` triplets while Fs remain — W fires
           as late as memory allows, but must precede the next F.
        4. Drain: ``(B, W)`` pairs while Bs remain, then any remaining W.

        ZB-H1 uses ``mem_limit = p``, matching 1F1B's peak live count.
        ZB-H2 uses ``mem_limit = 2*p - 1`` per the paper.
        """
        primary_tracks = []
        fill_tracks = []
        f_total_per_rank = [mbc] * pp

        for g in range(pp):
            n_warmup = max(1, min(n_warmup_fn(g), mbc))
            order = []
            n_f = 0
            n_b = 0
            n_w = 0
            live = 0
            for i in range(n_warmup):
                order.append(("F", i)); n_f += 1; live += 1
            while live < mem_limit and n_f < mbc and n_b < n_f:
                order.append(("B", n_b)); n_b += 1
                order.append(("F", n_f)); n_f += 1; live += 1
            while n_f < mbc:
                order.append(("B", n_b)); n_b += 1
                order.append(("W", n_w)); n_w += 1; live -= 1
                order.append(("F", n_f)); n_f += 1; live += 1
            while n_b < mbc:
                order.append(("B", n_b)); n_b += 1
                order.append(("W", n_w)); n_w += 1; live -= 1
            while n_w < mbc:
                order.append(("W", n_w)); n_w += 1; live -= 1

            primary_tracks.append(order)
            fill_tracks.append([])

        # 2) Resolve timings with iterative event-pass scheduler.
        schedules = [[] for _ in range(pp)]
        fwd_done = [dict() for _ in range(pp)]
        bwd_done = [dict() for _ in range(pp)]
        primary_cursor = [0] * pp
        fill_cursor = [0] * pp
        f_done_count = [0] * pp
        rank_t = [0.0] * pp

        def try_schedule(r, kind, mb):
            if kind == "F":
                if r > 0 and mb not in fwd_done[r - 1]:
                    return None
                dep = fwd_done[r - 1][mb] if r > 0 else 0.0
                dur = forward_times[r][mb]
                start = max(rank_t[r], dep)
                return start, dur, start + dur
            if kind == "B":
                if mb not in fwd_done[r]:
                    return None
                if r < pp - 1 and mb not in bwd_done[r + 1]:
                    return None
                dep_f = fwd_done[r][mb]
                dep_b = bwd_done[r + 1][mb] if r < pp - 1 else 0.0
                dur = b_times[r][mb]
                start = max(rank_t[r], dep_f, dep_b)
                return start, dur, start + dur
            # "W"
            if mb not in bwd_done[r]:
                return None
            dep_b = bwd_done[r][mb]
            dur = w_times[r][mb]
            start = max(rank_t[r], dep_b)
            return start, dur, start + dur

        def commit(r, kind, mb, start, dur, end):
            schedules[r].append((kind, mb, start, dur, end))
            if kind == "F":
                fwd_done[r][mb] = end
            elif kind == "B":
                bwd_done[r][mb] = end
            rank_t[r] = end

        while any(
            primary_cursor[r] < len(primary_tracks[r])
            or fill_cursor[r] < len(fill_tracks[r])
            for r in range(pp)
        ):
            progressed = False
            for r in range(pp):
                primary_remaining = primary_cursor[r] < len(primary_tracks[r])
                fill_remaining = fill_cursor[r] < len(fill_tracks[r])
                if not primary_remaining and not fill_remaining:
                    continue

                all_f_done = f_done_count[r] >= f_total_per_rank[r]
                w_threshold = 2 * r + 1

                if primary_remaining:
                    kind, mb = primary_tracks[r][primary_cursor[r]]
                    result = try_schedule(r, kind, mb)
                    if result is not None:
                        # After all F ops done, prefer fill W over next B
                        # when b_mb - w_mb >= 2*g + 1.
                        if all_f_done and fill_remaining and kind == "B":
                            fkind, fmb = fill_tracks[r][fill_cursor[r]]
                            if fkind == "W" and mb - fmb >= w_threshold:
                                fresult = try_schedule(r, fkind, fmb)
                                if fresult is not None:
                                    fstart, fdur, fend = fresult
                                    commit(r, fkind, fmb, fstart, fdur, fend)
                                    fill_cursor[r] += 1
                                    progressed = True
                                    continue
                        start, dur, end = result
                        commit(r, kind, mb, start, dur, end)
                        primary_cursor[r] += 1
                        if kind == "F":
                            f_done_count[r] += 1
                        progressed = True
                        continue

                # Primary blocked or exhausted — fall back to fill W.
                if fill_remaining:
                    fkind, fmb = fill_tracks[r][fill_cursor[r]]
                    fresult = try_schedule(r, fkind, fmb)
                    if fresult is not None:
                        fstart, fdur, fend = fresult
                        commit(r, fkind, fmb, fstart, fdur, fend)
                        fill_cursor[r] += 1
                        progressed = True

            if not progressed:
                raise RuntimeError(
                    f"{schedule_name} scheduler deadlock; check event ordering / deps."
                )

        max_time = max(s[-1][4] for s in schedules)
        self._last_schedules = schedules

        if draw:
            fig, ax = plt.subplots(figsize=(14, 5))
            colors = {"F": "#6b8ec9", "B": "#6db5b5", "W": "#5fa75f"}
            for rank, tasks in enumerate(schedules):
                for task_type, mb, start, duration, end in tasks:
                    ax.barh(y=pp - 1 - rank, width=duration, left=start,
                            height=0.6, color=colors[task_type], edgecolor="black")
                    ax.text(start + duration / 2, pp - 1 - rank, f"{task_type}{mb + 1}",
                            va="center", ha="center", fontsize=8, color="black")
            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Stage {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(
                f"{schedule_name} Pipeline Execution Timeline (pp={pp}, mbc={mbc})"
            )
            plt.grid(True, axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename(schedule_name))
            plt.close(fig)

        return max_time

    def calculate_zb_h1_bubble(self, pp, mbc, forward_times, b_times, w_times,
                               draw=False, output_path=None):
        """ZB-H1 schedule (Qi et al., ICLR 2024, §3.1).

        Same memory bound as 1F1B with the B/W split deferring W. Bubble
        shrinks vs 1F1B because deferred W fills idle time during the
        drain phase.
        """
        return self._calculate_zb_bubble(
            pp, mbc, forward_times, b_times, w_times,
            n_warmup_fn=lambda g: pp - g,
            mem_limit=pp,
            schedule_name="ZB-H1",
            draw=draw, output_path=output_path,
        )

    def calculate_zb_h2_bubble(self, pp, mbc, forward_times, b_times, w_times,
                               draw=False, output_path=None):
        """ZB-H2 schedule (Qi et al., ICLR 2024, §3.2).

        Deeper warmup (``2*(pp-g)-1``) and peak activation ``2*(p-g)-1``
        at rank g, pushing bubble toward zero at the cost of ~2x memory.
        """
        return self._calculate_zb_bubble(
            pp, mbc, forward_times, b_times, w_times,
            n_warmup_fn=lambda g: 2 * (pp - g) - 1,
            mem_limit=2 * pp - 1,
            schedule_name="ZB-H2",
            draw=draw, output_path=output_path,
        )

    def calculate_gpipe_bubble(self, pp, mbc, forward_times, backward_times,
                               draw=False, output_path=None):
        """GPipe schedule (Huang et al., 2018).

        Fully synchronous: every rank processes all ``mbc`` forward passes
        before starting any backward. Bubble is ``(pp-1)*(F+B)``. Included
        as an upper-bound baseline for comparison.
        """
        events_per_rank = []
        for r in range(pp):
            events = [("F", mb) for mb in range(mbc)]
            events.extend(("B", mb) for mb in range(mbc - 1, -1, -1))
            events_per_rank.append(events)

        schedules = [[] for _ in range(pp)]
        fwd_done = [dict() for _ in range(pp)]
        bwd_done = [dict() for _ in range(pp)]
        rank_ptr = [0] * pp
        rank_t = [0.0] * pp

        while any(rank_ptr[r] < len(events_per_rank[r]) for r in range(pp)):
            progressed = False
            for r in range(pp):
                if rank_ptr[r] >= len(events_per_rank[r]):
                    continue
                kind, mb = events_per_rank[r][rank_ptr[r]]
                if kind == "F":
                    if r > 0 and mb not in fwd_done[r - 1]:
                        continue
                    dep = fwd_done[r - 1][mb] if r > 0 else 0.0
                    dur = forward_times[r][mb]
                    start = max(rank_t[r], dep)
                    end = start + dur
                    schedules[r].append(("F", mb, start, dur, end))
                    fwd_done[r][mb] = end
                    rank_t[r] = end
                else:  # "B"
                    if mb not in fwd_done[r]:
                        continue
                    if r < pp - 1 and mb not in bwd_done[r + 1]:
                        continue
                    dep_f = fwd_done[r][mb]
                    dep_b = bwd_done[r + 1][mb] if r < pp - 1 else 0.0
                    dur = backward_times[r][mb]
                    start = max(rank_t[r], dep_f, dep_b)
                    end = start + dur
                    schedules[r].append(("B", mb, start, dur, end))
                    bwd_done[r][mb] = end
                    rank_t[r] = end
                rank_ptr[r] += 1
                progressed = True
            if not progressed:
                raise RuntimeError(
                    "GPipe scheduler deadlock; check event ordering / deps."
                )

        max_time = max(s[-1][4] for s in schedules)
        self._last_schedules = schedules

        if draw:
            fig, ax = plt.subplots(figsize=(14, 5))
            colors = {"F": "#6b8ec9", "B": "#6db5b5"}
            for rank, tasks in enumerate(schedules):
                for task_type, mb, start, duration, end in tasks:
                    ax.barh(y=pp - 1 - rank, width=duration, left=start,
                            height=0.6, color=colors[task_type], edgecolor="black")
                    ax.text(start + duration / 2, pp - 1 - rank,
                            f"{task_type}{mb + 1}",
                            va="center", ha="center", fontsize=8, color="black")
            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Stage {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(f"GPipe Pipeline Execution Timeline (pp={pp}, mbc={mbc})")
            plt.grid(True, axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename("GPipe"))
            plt.close(fig)

        return max_time

    def _run_virtual_stage_scheduler(self, pp, V, events_per_rank, vs_to_rank,
                                     fwd_times_vs, bwd_times_vs=None,
                                     b_times_vs=None, w_times_vs=None,
                                     schedule_name="",
                                     draw=False, output_path=None):
        """Event-pass scheduler for virtual-stage schedules.

        Each event is a tuple ``(kind, mb, vs_local)`` where ``kind`` is
        one of ``"F"``, ``"B"``, ``"W"`` and the global virtual-stage
        index is recovered via ``vs_to_rank``. Dependencies:
        - F(mb, gvs): F(mb, gvs-1) on its owning rank, if gvs > 0.
        - B(mb, gvs): F(mb, gvs) on this rank AND B(mb, gvs+1) on its
          owning rank if gvs < V*pp - 1.
        - W(mb, gvs): B(mb, gvs) on this rank.

        Pass ``bwd_times_vs`` for fused-B schedules (Interleaved 1F1B) or
        ``(b_times_vs, w_times_vs)`` for the split-B/W ZB-V schedule.
        """
        total_vs = V * pp
        split = b_times_vs is not None

        # Map (rank, vs_local) ↔ global vs.
        vs_local_map = [{} for _ in range(pp)]  # rank -> {gvs: vs_local}
        owned_vs = [[] for _ in range(pp)]
        for gvs in range(total_vs):
            r = vs_to_rank(gvs)
            vs_local = len(owned_vs[r])
            owned_vs[r].append(gvs)
            vs_local_map[r][gvs] = vs_local

        def gvs_of(r, vs_local):
            return owned_vs[r][vs_local]

        schedules = [[] for _ in range(pp)]
        fwd_done = [{} for _ in range(pp)]  # (mb, vs_local) -> end
        b_done = [{} for _ in range(pp)]
        rank_t = [0.0] * pp
        # Each rank dispatches the first not-yet-done event whose deps
        # are ready, scanning forward through its event list rather
        # than stalling on a blocked head (matches the rlpp ZB-V
        # reference — necessary to break fold-rank deadlocks on
        # cross-chunk cross-rank cycles).
        dispatched = [set() for _ in range(pp)]

        def try_schedule(r, kind, mb, vs_local):
            gvs = gvs_of(r, vs_local)
            if kind == "F":
                if gvs > 0:
                    prev_gvs = gvs - 1
                    pr = vs_to_rank(prev_gvs)
                    pv = vs_local_map[pr][prev_gvs]
                    if (mb, pv) not in fwd_done[pr]:
                        return None
                    dep = fwd_done[pr][(mb, pv)]
                else:
                    dep = 0.0
                dur = fwd_times_vs[gvs][mb]
                start = max(rank_t[r], dep)
                return start, dur, start + dur
            if kind == "B":
                if (mb, vs_local) not in fwd_done[r]:
                    return None
                dep_f = fwd_done[r][(mb, vs_local)]
                if gvs < total_vs - 1:
                    next_gvs = gvs + 1
                    nr = vs_to_rank(next_gvs)
                    nv = vs_local_map[nr][next_gvs]
                    if (mb, nv) not in b_done[nr]:
                        return None
                    dep_b = b_done[nr][(mb, nv)]
                else:
                    dep_b = 0.0
                dur = (b_times_vs[gvs][mb] if split else bwd_times_vs[gvs][mb])
                start = max(rank_t[r], dep_f, dep_b)
                return start, dur, start + dur
            # W
            if (mb, vs_local) not in b_done[r]:
                return None
            dep = b_done[r][(mb, vs_local)]
            dur = w_times_vs[gvs][mb]
            start = max(rank_t[r], dep)
            return start, dur, start + dur

        def commit(r, kind, mb, vs_local, start, dur, end):
            schedules[r].append((kind, mb, vs_local, start, dur, end))
            if kind == "F":
                fwd_done[r][(mb, vs_local)] = end
            elif kind == "B":
                b_done[r][(mb, vs_local)] = end
            rank_t[r] = end

        def all_done():
            return all(
                len(dispatched[r]) == len(events_per_rank[r])
                for r in range(pp)
            )

        while not all_done():
            progressed = False
            for r in range(pp):
                if len(dispatched[r]) == len(events_per_rank[r]):
                    continue
                for idx, ev in enumerate(events_per_rank[r]):
                    if idx in dispatched[r]:
                        continue
                    kind, mb, vs_local = ev
                    result = try_schedule(r, kind, mb, vs_local)
                    if result is not None:
                        start, dur, end = result
                        commit(r, kind, mb, vs_local, start, dur, end)
                        dispatched[r].add(idx)
                        progressed = True
                        break
            if not progressed:
                blocked = []
                for r in range(pp):
                    heads = [
                        (i, events_per_rank[r][i])
                        for i in range(len(events_per_rank[r]))
                        if i not in dispatched[r]
                    ][:3]
                    blocked.append(f"r{r} next: {heads}")
                raise RuntimeError(
                    f"{schedule_name} scheduler deadlock; blocked:\n"
                    + "\n".join(blocked)
                )

        max_time = max(s[-1][5] for s in schedules)
        self._last_schedules = schedules

        if draw:
            fig, ax = plt.subplots(figsize=(16, 5))
            colors = {"F": "#6b8ec9", "B": "#6db5b5", "W": "#5fa75f"}
            for r, tasks in enumerate(schedules):
                for kind, mb, vs_local, start, dur, end in tasks:
                    ax.barh(y=pp - 1 - r, width=dur, left=start,
                            height=0.6, color=colors[kind], edgecolor="black")
                    txt_color = "black" if vs_local == 0 else "white"
                    label = f"{kind}{mb + 1}"
                    ax.text(start + dur / 2, pp - 1 - r, label,
                            va="center", ha="center", fontsize=7,
                            color=txt_color)
            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Rank {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(
                f"{schedule_name} Pipeline Timeline "
                f"(pp={pp}, V={V}, total_vs={total_vs})"
            )
            plt.grid(True, axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename(schedule_name))
            plt.close(fig)

        return max_time

    def calculate_interleaved_1f1b_bubble(self, pp, mbc, V, fwd_times_vs,
                                          bwd_times_vs, draw=False,
                                          output_path=None):
        """Interleaved 1F1B (Narayanan et al. 2021, Megatron).

        Each rank owns ``V`` non-contiguous virtual stages via round-robin
        mapping: rank ``g`` ↔ global vs ``{g, g+p, ..., g+(V-1)*p}``.
        Warmup on rank ``g``: ``(V-1)*p + (p - g - 1)`` Fs. Steady state
        alternates (B, F) at the virtual-stage level; drain is remaining
        Bs.
        """
        def vs_to_rank(gvs):
            return gvs % pp

        # Build per-rank F / B sequences in microbatch-group-major order.
        # Forward: for each group of pp mbs, cycle vs_local=0..V-1, firing F
        # for each mb in the group.
        # Backward: same grouping, but vs_local descends (V-1..0) since
        # gradients flow back through the highest virtual stage first.
        def build_mb_groups(n):
            groups = []
            n_full = n // pp
            for gi in range(n_full):
                groups.append(list(range(gi * pp, (gi + 1) * pp)))
            if n_full * pp < n:
                groups.append(list(range(n_full * pp, n)))
            return groups

        groups = build_mb_groups(mbc)
        per_rank_f_seq = []
        per_rank_b_seq = []
        for _g in range(pp):
            f_seq = []
            b_seq = []
            for grp in groups:
                for vs_local in range(V):
                    for mb in grp:
                        f_seq.append((mb, vs_local))
                for vs_local in range(V - 1, -1, -1):
                    for mb in grp:
                        b_seq.append((mb, vs_local))
            per_rank_f_seq.append(f_seq)
            per_rank_b_seq.append(b_seq)

        # Megatron-LM formula: warmup = 2*(p - g - 1) + (V - 1)*p.
        # Steady state is 1F1B at the virtual-stage level: (F, B) pairs.
        # Cooldown drains remaining B's.
        events_per_rank = []
        for g in range(pp):
            f_seq = per_rank_f_seq[g]
            b_seq = per_rank_b_seq[g]
            warmup = min(2 * (pp - g - 1) + (V - 1) * pp, len(f_seq))
            warmup = max(1, warmup)
            events = []
            f_idx = 0
            b_idx = 0
            for _ in range(warmup):
                mb, vs_local = f_seq[f_idx]
                events.append(("F", mb, vs_local)); f_idx += 1
            while f_idx < len(f_seq):
                mb_f, vs_f = f_seq[f_idx]
                events.append(("F", mb_f, vs_f)); f_idx += 1
                mb_b, vs_b = b_seq[b_idx]
                events.append(("B", mb_b, vs_b)); b_idx += 1
            while b_idx < len(b_seq):
                mb_b, vs_b = b_seq[b_idx]
                events.append(("B", mb_b, vs_b)); b_idx += 1
            events_per_rank.append(events)

        return self._run_virtual_stage_scheduler(
            pp, V, events_per_rank, vs_to_rank,
            fwd_times_vs=fwd_times_vs, bwd_times_vs=bwd_times_vs,
            schedule_name="Interleaved 1F1B",
            draw=draw, output_path=output_path,
        )

    def _run_vs_greedy_scheduler_split(self, pp, V, f_queues_per_vs,
                                         b_queues_per_vs, w_queues_per_vs,
                                         vs_to_rank, fwd_times_vs, b_times_vs,
                                         w_times_vs, mem_limit,
                                         schedule_name="",
                                         draw=False, output_path=None):
        """Greedy scheduler with per-(rank, vs_local) queues.

        On each rank, F/B/W queues are indexed by vs_local so that the
        scheduler can prefer firing the action that unblocks the chain.
        Priority within a rank: B(vs=V-1..0) > F(vs=V-1..0) > W(vs=V-1..0)
        — B first to advance the critical path, F next to feed future
        B's / keep the pipeline full, W last (deferred until needed to
        free memory). Each queue is processed head-to-tail per mb.
        """
        total_vs = V * pp
        vs_local_map = [{} for _ in range(pp)]
        owned_vs = [[] for _ in range(pp)]
        for gvs in range(total_vs):
            r = vs_to_rank(gvs)
            owned_vs[r].append(gvs)
            vs_local_map[r][gvs] = len(owned_vs[r]) - 1

        def gvs_of(r, vs_local):
            return owned_vs[r][vs_local]

        schedules = [[] for _ in range(pp)]
        fwd_done = [{} for _ in range(pp)]
        b_done = [{} for _ in range(pp)]
        rank_t = [0.0] * pp
        # f_ptr[r][vs] / b_ptr[r][vs] / w_ptr[r][vs]
        f_ptr = [[0] * V for _ in range(pp)]
        b_ptr = [[0] * V for _ in range(pp)]
        w_ptr = [[0] * V for _ in range(pp)]
        live = [0] * pp

        def try_F(r, mb, vs_local):
            gvs = gvs_of(r, vs_local)
            if gvs > 0:
                prev_gvs = gvs - 1
                pr = vs_to_rank(prev_gvs)
                pv = vs_local_map[pr][prev_gvs]
                if (mb, pv) not in fwd_done[pr]:
                    return None
                dep = fwd_done[pr][(mb, pv)]
            else:
                dep = 0.0
            dur = fwd_times_vs[gvs][mb]
            start = max(rank_t[r], dep)
            return start, dur, start + dur

        def try_B(r, mb, vs_local):
            gvs = gvs_of(r, vs_local)
            if (mb, vs_local) not in fwd_done[r]:
                return None
            dep_f = fwd_done[r][(mb, vs_local)]
            if gvs < total_vs - 1:
                next_gvs = gvs + 1
                nr = vs_to_rank(next_gvs)
                nv = vs_local_map[nr][next_gvs]
                if (mb, nv) not in b_done[nr]:
                    return None
                dep_b = b_done[nr][(mb, nv)]
            else:
                dep_b = 0.0
            dur = b_times_vs[gvs][mb]
            start = max(rank_t[r], dep_f, dep_b)
            return start, dur, start + dur

        def try_W(r, mb, vs_local):
            if (mb, vs_local) not in b_done[r]:
                return None
            gvs = gvs_of(r, vs_local)
            dep = b_done[r][(mb, vs_local)]
            dur = w_times_vs[gvs][mb]
            start = max(rank_t[r], dep)
            return start, dur, start + dur

        def commit(r, kind, mb, vs_local, start, dur, end):
            schedules[r].append((kind, mb, vs_local, start, dur, end))
            if kind == "F":
                fwd_done[r][(mb, vs_local)] = end
                live[r] += 1
            elif kind == "B":
                b_done[r][(mb, vs_local)] = end
            else:
                live[r] -= 1
            rank_t[r] = end

        def remaining():
            for r in range(pp):
                for vs in range(V):
                    if f_ptr[r][vs] < len(f_queues_per_vs[r][vs]):
                        return True
                    if b_ptr[r][vs] < len(b_queues_per_vs[r][vs]):
                        return True
                    if w_ptr[r][vs] < len(w_queues_per_vs[r][vs]):
                        return True
            return False

        while remaining():
            progressed = False
            order = sorted(range(pp), key=lambda r: rank_t[r])
            for r in order:
                fired = False
                # Priority: B (vs=V-1..0) > F (vs=V-1..0) > W (vs=V-1..0).
                for vs in range(V - 1, -1, -1):
                    if b_ptr[r][vs] < len(b_queues_per_vs[r][vs]):
                        mb = b_queues_per_vs[r][vs][b_ptr[r][vs]]
                        res = try_B(r, mb, vs)
                        if res is not None:
                            commit(r, "B", mb, vs, *res)
                            b_ptr[r][vs] += 1
                            fired = True; progressed = True; break
                if fired:
                    continue
                for vs in range(V - 1, -1, -1):
                    if (f_ptr[r][vs] < len(f_queues_per_vs[r][vs])
                            and live[r] < mem_limit):
                        mb = f_queues_per_vs[r][vs][f_ptr[r][vs]]
                        res = try_F(r, mb, vs)
                        if res is not None:
                            commit(r, "F", mb, vs, *res)
                            f_ptr[r][vs] += 1
                            fired = True; progressed = True; break
                if fired:
                    continue
                for vs in range(V - 1, -1, -1):
                    if w_ptr[r][vs] < len(w_queues_per_vs[r][vs]):
                        mb = w_queues_per_vs[r][vs][w_ptr[r][vs]]
                        res = try_W(r, mb, vs)
                        if res is not None:
                            commit(r, "W", mb, vs, *res)
                            w_ptr[r][vs] += 1
                            fired = True; progressed = True; break
            if not progressed:
                blocked = []
                for r in range(pp):
                    heads = []
                    for vs in range(V):
                        if f_ptr[r][vs] < len(f_queues_per_vs[r][vs]):
                            heads.append(f"F(mb={f_queues_per_vs[r][vs][f_ptr[r][vs]]}, vs={vs})")
                        if b_ptr[r][vs] < len(b_queues_per_vs[r][vs]):
                            heads.append(f"B(mb={b_queues_per_vs[r][vs][b_ptr[r][vs]]}, vs={vs})")
                        if w_ptr[r][vs] < len(w_queues_per_vs[r][vs]):
                            heads.append(f"W(mb={w_queues_per_vs[r][vs][w_ptr[r][vs]]}, vs={vs})")
                    blocked.append(f"r{r} live={live[r]} {heads}")
                raise RuntimeError(
                    f"{schedule_name} split greedy deadlock:\n"
                    + "\n".join(blocked)
                )

        max_time = max(s[-1][5] for s in schedules)
        self._last_schedules = schedules

        if draw:
            fig, ax = plt.subplots(figsize=(16, 5))
            colors = {"F": "#6b8ec9", "B": "#6db5b5", "W": "#5fa75f"}
            for r, tasks in enumerate(schedules):
                for kind, mb, vs_local, start, dur, end in tasks:
                    ax.barh(y=pp - 1 - r, width=dur, left=start,
                            height=0.6, color=colors[kind], edgecolor="black")
                    txt_color = "black" if vs_local == 0 else "white"
                    label = f"{kind}{mb + 1}"
                    ax.text(start + dur / 2, pp - 1 - r, label,
                            va="center", ha="center", fontsize=7,
                            color=txt_color)
            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Rank {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(
                f"{schedule_name} Pipeline Timeline "
                f"(pp={pp}, V={V}, total_vs={total_vs})"
            )
            plt.grid(True, axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename(schedule_name))
            plt.close(fig)

        return max_time

    def _run_vs_greedy_scheduler(self, pp, V, f_queues, b_queues, w_queues,
                                  vs_to_rank, fwd_times_vs, b_times_vs,
                                  w_times_vs, mem_limit, schedule_name="",
                                  draw=False, output_path=None):
        """Greedy event-driven scheduler for virtual-stage schedules.

        Each rank maintains independent F/B/W queues processed in-order
        per queue, but the queue to fire from each step is chosen
        greedily: prefer W (free memory), then B (critical path), then F
        (only when ``live < mem_limit``). ``live[r] = fires[F] - fires[W]``
        is the per-rank activation count bounded by ``mem_limit``.

        Dependencies (same as the strict scheduler):
        - F(mb, gvs): F(mb, gvs-1) on its owning rank.
        - B(mb, gvs): F(mb, gvs) locally AND B(mb, gvs+1) on its owning rank.
        - W(mb, gvs): B(mb, gvs) locally.
        """
        total_vs = V * pp

        vs_local_map = [{} for _ in range(pp)]
        owned_vs = [[] for _ in range(pp)]
        for gvs in range(total_vs):
            r = vs_to_rank(gvs)
            owned_vs[r].append(gvs)
            vs_local_map[r][gvs] = len(owned_vs[r]) - 1

        def gvs_of(r, vs_local):
            return owned_vs[r][vs_local]

        schedules = [[] for _ in range(pp)]
        fwd_done = [{} for _ in range(pp)]
        b_done = [{} for _ in range(pp)]
        rank_t = [0.0] * pp
        f_ptr = [0] * pp
        b_ptr = [0] * pp
        w_ptr = [0] * pp
        live = [0] * pp

        def try_F(r, mb, vs_local):
            gvs = gvs_of(r, vs_local)
            if gvs > 0:
                prev_gvs = gvs - 1
                pr = vs_to_rank(prev_gvs)
                pv = vs_local_map[pr][prev_gvs]
                if (mb, pv) not in fwd_done[pr]:
                    return None
                dep = fwd_done[pr][(mb, pv)]
            else:
                dep = 0.0
            dur = fwd_times_vs[gvs][mb]
            start = max(rank_t[r], dep)
            return start, dur, start + dur

        def try_B(r, mb, vs_local):
            gvs = gvs_of(r, vs_local)
            if (mb, vs_local) not in fwd_done[r]:
                return None
            dep_f = fwd_done[r][(mb, vs_local)]
            if gvs < total_vs - 1:
                next_gvs = gvs + 1
                nr = vs_to_rank(next_gvs)
                nv = vs_local_map[nr][next_gvs]
                if (mb, nv) not in b_done[nr]:
                    return None
                dep_b = b_done[nr][(mb, nv)]
            else:
                dep_b = 0.0
            dur = b_times_vs[gvs][mb]
            start = max(rank_t[r], dep_f, dep_b)
            return start, dur, start + dur

        def try_W(r, mb, vs_local):
            if (mb, vs_local) not in b_done[r]:
                return None
            gvs = gvs_of(r, vs_local)
            dep = b_done[r][(mb, vs_local)]
            dur = w_times_vs[gvs][mb]
            start = max(rank_t[r], dep)
            return start, dur, start + dur

        def commit(r, kind, mb, vs_local, start, dur, end):
            schedules[r].append((kind, mb, vs_local, start, dur, end))
            if kind == "F":
                fwd_done[r][(mb, vs_local)] = end
                live[r] += 1
            elif kind == "B":
                b_done[r][(mb, vs_local)] = end
            else:  # W
                live[r] -= 1
            rank_t[r] = end

        def remaining():
            return any(f_ptr[r] < len(f_queues[r]) or
                       b_ptr[r] < len(b_queues[r]) or
                       w_ptr[r] < len(w_queues[r]) for r in range(pp))

        while remaining():
            progressed = False
            # Fire on the rank with the lowest rank_t first to keep
            # dependencies unblocked as early as possible.
            order = sorted(range(pp), key=lambda r: rank_t[r])
            for r in order:
                # Priority: B (critical path) -> W (free memory) -> F.
                if b_ptr[r] < len(b_queues[r]):
                    mb, vs = b_queues[r][b_ptr[r]]
                    res = try_B(r, mb, vs)
                    if res is not None:
                        commit(r, "B", mb, vs, *res)
                        b_ptr[r] += 1
                        progressed = True
                        continue
                if w_ptr[r] < len(w_queues[r]):
                    mb, vs = w_queues[r][w_ptr[r]]
                    res = try_W(r, mb, vs)
                    if res is not None:
                        commit(r, "W", mb, vs, *res)
                        w_ptr[r] += 1
                        progressed = True
                        continue
                if f_ptr[r] < len(f_queues[r]) and live[r] < mem_limit:
                    mb, vs = f_queues[r][f_ptr[r]]
                    res = try_F(r, mb, vs)
                    if res is not None:
                        commit(r, "F", mb, vs, *res)
                        f_ptr[r] += 1
                        progressed = True
                        continue
            if not progressed:
                blocked = []
                for r in range(pp):
                    heads = []
                    if f_ptr[r] < len(f_queues[r]):
                        heads.append(f"F{f_queues[r][f_ptr[r]]}")
                    if b_ptr[r] < len(b_queues[r]):
                        heads.append(f"B{b_queues[r][b_ptr[r]]}")
                    if w_ptr[r] < len(w_queues[r]):
                        heads.append(f"W{w_queues[r][w_ptr[r]]}")
                    blocked.append(
                        f"r{r} live={live[r]} heads={heads}"
                    )
                raise RuntimeError(
                    f"{schedule_name} greedy deadlock:\n" + "\n".join(blocked)
                )

        max_time = max(s[-1][5] for s in schedules)
        self._last_schedules = schedules

        if draw:
            fig, ax = plt.subplots(figsize=(16, 5))
            colors = {"F": "#6b8ec9", "B": "#6db5b5", "W": "#5fa75f"}
            for r, tasks in enumerate(schedules):
                for kind, mb, vs_local, start, dur, end in tasks:
                    gvs = gvs_of(r, vs_local)
                    ax.barh(y=pp - 1 - r, width=dur, left=start,
                            height=0.6, color=colors[kind], edgecolor="black")
                    label = f"{kind}{mb}.{gvs}"
                    ax.text(start + dur / 2, pp - 1 - r, label,
                            va="center", ha="center", fontsize=7, color="black")
            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Rank {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(
                f"{schedule_name} Pipeline Timeline "
                f"(pp={pp}, V={V}, total_vs={total_vs})"
            )
            plt.grid(True, axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_path or self.default_gantt_filename(schedule_name))
            plt.close(fig)

        return max_time

    def calculate_zb_v_bubble(self, pp, mbc, fwd_times_vs, b_times_vs,
                              w_times_vs, draw=False, output_path=None):
        """ZB-V schedule (Qi et al. 2024, §6; rlpp reference).

        V=2 virtual stages per rank with V-shape mapping: rank ``g``
        owns ``chunk0 = g`` (early) and ``chunk1 = 2p-1-g`` (late).
        Pyramid assignment balances activation memory: rank 0 pairs
        embed-heavy first stage with LM-head-heavy last stage.

        Per-rank event sequence (four phases, from rlpp/zb_v.py):

        1. **Warmup 1a**: ``2(p-g)-1`` F(chunk0) (mb=0..k-1). For g=0
           this is 7 F's — rank 0 loads the first chunk of the first 7
           microbatches; then chunk1 of mb=0 returns (pipeline wraps
           through all ranks and folds back).
        2. **Warmup 1b**: ``g`` interleaved ``[F(chunk1), F(chunk0)]``
           pairs, filling the fold-side bubble.
        3. **Warmup 1c**: ``p-g`` cycles of
           ``[F(chunk1), B(chunk1), W(chunk1)]`` — immediately frees
           late-stage memory, keeping peak at ``2p`` live half-chunks.
        4. **Steady state**: alternating
           ``F(chunk0), B(chunk0), W(chunk0), F(chunk1), B(chunk1),
           W(chunk1)`` per mb until F's exhaust.
        5. **Cooldown**: ``g`` pairs of B(chunk0)+B(chunk1), then
           ``p-g`` pairs of B(chunk0)+W(chunk0).
        6. **Tail**: remaining W's.

        Dispatch uses scan-and-skip (not strict head-of-queue) to
        avoid deadlock on fold-rank cross-chunk cycles.
        """
        V = 2

        def vs_to_rank(gvs):
            return gvs if gvs < pp else (2 * pp - 1 - gvs)

        # vs_local for a given rank g: chunk0 → vs_local=0, chunk1 →
        # vs_local=1 (matches V-shape ownership order in
        # _run_virtual_stage_scheduler: gvs=g first, gvs=2p-1-g second).
        CHUNK0 = 0
        CHUNK1 = 1

        events_per_rank = []
        for g in range(pp):
            f0 = f1 = b0 = b1 = w0 = w1 = 0
            ops = []

            warmup_n1 = 2 * (pp - g) - 1
            for _ in range(warmup_n1):
                if f0 < mbc:
                    ops.append(("F", f0, CHUNK0)); f0 += 1

            warmup_n2 = g
            for _ in range(warmup_n2):
                if f1 < mbc:
                    ops.append(("F", f1, CHUNK1)); f1 += 1
                if f0 < mbc:
                    ops.append(("F", f0, CHUNK0)); f0 += 1

            warmup_n3 = pp - g
            for _ in range(warmup_n3):
                if f1 < mbc:
                    ops.append(("F", f1, CHUNK1)); f1 += 1
                if b1 < f1 and b1 < mbc:
                    ops.append(("B", b1, CHUNK1)); b1 += 1
                if w1 < b1 and w1 < mbc:
                    ops.append(("W", w1, CHUNK1)); w1 += 1

            while f0 < mbc or f1 < mbc:
                if f0 < mbc:
                    ops.append(("F", f0, CHUNK0)); f0 += 1
                if b0 < f0 and b0 < mbc:
                    ops.append(("B", b0, CHUNK0)); b0 += 1
                if w0 < b0 and w0 < mbc:
                    ops.append(("W", w0, CHUNK0)); w0 += 1
                if f1 < mbc:
                    ops.append(("F", f1, CHUNK1)); f1 += 1
                if b1 < f1 and b1 < mbc:
                    ops.append(("B", b1, CHUNK1)); b1 += 1
                if w1 < b1 and w1 < mbc:
                    ops.append(("W", w1, CHUNK1)); w1 += 1

            cooldown_n1 = g
            for _ in range(cooldown_n1):
                if b0 < mbc:
                    ops.append(("B", b0, CHUNK0)); b0 += 1
                if b1 < mbc:
                    ops.append(("B", b1, CHUNK1)); b1 += 1

            cooldown_n2 = pp - g
            for _ in range(cooldown_n2):
                if b0 < mbc:
                    ops.append(("B", b0, CHUNK0)); b0 += 1
                if w0 < b0 and w0 < mbc:
                    ops.append(("W", w0, CHUNK0)); w0 += 1

            while w1 < mbc:
                ops.append(("W", w1, CHUNK1)); w1 += 1
            while w0 < mbc:
                ops.append(("W", w0, CHUNK0)); w0 += 1

            events_per_rank.append(ops)

        return self._run_virtual_stage_scheduler(
            pp, V, events_per_rank, vs_to_rank,
            fwd_times_vs=fwd_times_vs,
            b_times_vs=b_times_vs, w_times_vs=w_times_vs,
            schedule_name="ZB-V",
            draw=draw, output_path=output_path,
        )

    @staticmethod
    def default_gantt_filename(schedule_name):
        """Canonical Gantt PNG filename per schedule.

        Accepts either the config-level key (e.g. ``"zb_h1"``) or the
        human-readable name used in chart titles (``"ZB-H1"``).
        """
        key = schedule_name.lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "1f1b": "corrected_1F1B_pipeline.png",
            "zb_h1": "zb_h1_pipeline.png",
            "zb_h2": "zb_h2_pipeline.png",
            "gpipe": "gpipe_pipeline.png",
            "interleaved_1f1b": "interleaved_1f1b_pipeline.png",
            "zb_v": "zb_v_pipeline.png",
        }
        return mapping.get(key, "pp_pipeline.png")

    def _rank_to_chunk_name(self):
        """Map pipeline rank -> chunk name (first / middle / last).

        Used to look up per-(chunk, seq_len) cached timings for a given rank.
        """
        pp = self.strategy.pp_size
        if pp == 1:
            return [FIRST_CHUNK]
        result = [FIRST_CHUNK]
        if pp > 2:
            result.extend([MIDDLE_CHUNK] * (pp - 2))
        result.append(LAST_CHUNK)
        return result

    def _ensure_seq_lens(self):
        """Guard: populate self.seq_lens with a constant array if unset."""
        if self.seq_lens is None:
            self.seq_lens = np.full(
                self.strategy.micro_batch_num,
                int(self.strategy.seq_len),
                dtype=np.int64,
            )

    def _per_rank_fwd_bwd_times(self, apply_disturbance=True):
        """Return ``(forward_times, backward_times)``, each a ``pp x mbc``
        2-D list indexed as ``arr[rank][microbatch]``. When
        ``seq_len_std == 0`` all rows are constant, matching prior behaviour.

        When ``apply_disturbance`` is True (the default) and the active
        pp_schedule is a physical-rank schedule, the composed A/B/C
        multipliers are applied here. ``_per_virtual_stage_times`` sets
        this to False so multipliers are applied at the gvs level instead.
        """
        self._ensure_seq_lens()
        pp = self.strategy.pp_size
        mbc = self.strategy.micro_batch_num
        self._populate_chunk_cost_cache(self._chunk_names_for_pp(), self.seq_lens)
        rank_chunk = self._rank_to_chunk_name()

        forward_times = [[0.0] * mbc for _ in range(pp)]
        backward_times = [[0.0] * mbc for _ in range(pp)]
        for r in range(pp):
            chunk_name = rank_chunk[r]
            for m in range(mbc):
                s = int(self.seq_lens[m])
                fwd, bwd = self._chunk_fwd_bwd_at(chunk_name, s)
                if apply_disturbance:
                    fwd *= self._compose_mult("F", r, m)
                    bwd *= self._compose_mult("B", r, m)
                forward_times[r][m] = fwd
                backward_times[r][m] = bwd
        return forward_times, backward_times

    def _per_rank_fwd_b_w_times(self, apply_disturbance=True):
        """Return ``(forward_times, b_times, w_times)``, each a ``pp x mbc``
        2-D list indexed as ``arr[rank][microbatch]`` (ZB-style B/W split).

        ``apply_disturbance`` as in :meth:`_per_rank_fwd_bwd_times`.
        """
        self._ensure_seq_lens()
        pp = self.strategy.pp_size
        mbc = self.strategy.micro_batch_num
        self._populate_chunk_cost_cache(self._chunk_names_for_pp(), self.seq_lens)
        rank_chunk = self._rank_to_chunk_name()

        forward_times = [[0.0] * mbc for _ in range(pp)]
        b_times = [[0.0] * mbc for _ in range(pp)]
        w_times = [[0.0] * mbc for _ in range(pp)]
        for r in range(pp):
            chunk_name = rank_chunk[r]
            for m in range(mbc):
                s = int(self.seq_lens[m])
                fwd, b, w = self._chunk_fwd_b_w_at(chunk_name, s)
                if apply_disturbance:
                    fwd *= self._compose_mult("F", r, m)
                    b *= self._compose_mult("B", r, m)
                    w *= self._compose_mult("W", r, m)
                forward_times[r][m] = fwd
                b_times[r][m] = b
                w_times[r][m] = w
        return forward_times, b_times, w_times

    def _per_virtual_stage_times(self, V, vs_to_rank, split_bw=False):
        """Per-virtual-stage timing arrays, each a ``(V*pp) x mbc`` 2-D list.

        ``vs_to_rank(gvs)`` maps a global virtual-stage index to the rank
        that owns it. Each virtual stage inherits its rank's compute time
        (for the relevant microbatch) divided by ``V`` (uniform layer split).
        Disturbance multipliers are applied per-gvs here (A, B and C all
        use virtual-stage indexing for these schedules).
        """
        self._ensure_seq_lens()
        pp = self.strategy.pp_size
        mbc = self.strategy.micro_batch_num
        # Fetch clean per-rank times; we'll compose the multipliers at the
        # gvs level below so each virtual stage gets its own sampled noise.
        if split_bw:
            fwd_r, b_r, w_r = self._per_rank_fwd_b_w_times(apply_disturbance=False)
        else:
            fwd_r, bwd_r = self._per_rank_fwd_bwd_times(apply_disturbance=False)
        total_vs = V * pp
        fwd_vs = [[0.0] * mbc for _ in range(total_vs)]
        if split_bw:
            b_vs = [[0.0] * mbc for _ in range(total_vs)]
            w_vs = [[0.0] * mbc for _ in range(total_vs)]
        else:
            bwd_vs = [[0.0] * mbc for _ in range(total_vs)]
        for gvs in range(total_vs):
            r = vs_to_rank(gvs)
            for m in range(mbc):
                fwd_vs[gvs][m] = (fwd_r[r][m] / V) * self._compose_mult("F", gvs, m)
                if split_bw:
                    b_vs[gvs][m] = (b_r[r][m] / V) * self._compose_mult("B", gvs, m)
                    w_vs[gvs][m] = (w_r[r][m] / V) * self._compose_mult("W", gvs, m)
                else:
                    bwd_vs[gvs][m] = (bwd_r[r][m] / V) * self._compose_mult("B", gvs, m)
        if split_bw:
            return fwd_vs, b_vs, w_vs
        return fwd_vs, bwd_vs

    def _compute_pp_total_time(self, draw=False, output_path=None):
        pp = self.strategy.pp_size
        mbc = self.strategy.micro_batch_num
        schedule = self.strategy.pp_schedule

        if schedule in ("zb_h1", "zb_h2"):
            fwd_times, b_times, w_times = self._per_rank_fwd_b_w_times()
            scheduler = (self.calculate_zb_h1_bubble if schedule == "zb_h1"
                         else self.calculate_zb_h2_bubble)
            return scheduler(
                pp, mbc, fwd_times, b_times, w_times,
                draw=draw, output_path=output_path,
            )

        if schedule == "interleaved_1f1b":
            V = self.strategy.interleaving_size
            fwd_vs, bwd_vs = self._per_virtual_stage_times(
                V, vs_to_rank=lambda gvs: gvs % pp, split_bw=False
            )
            return self.calculate_interleaved_1f1b_bubble(
                pp, mbc, V, fwd_vs, bwd_vs,
                draw=draw, output_path=output_path,
            )

        if schedule == "zb_v":
            V = 2
            fwd_vs, b_vs, w_vs = self._per_virtual_stage_times(
                V, vs_to_rank=lambda gvs: gvs if gvs < pp else (2 * pp - 1 - gvs),
                split_bw=True,
            )
            return self.calculate_zb_v_bubble(
                pp, mbc, fwd_vs, b_vs, w_vs,
                draw=draw, output_path=output_path,
            )

        forward_times, backward_times = self._per_rank_fwd_bwd_times()
        if schedule == "gpipe":
            return self.calculate_gpipe_bubble(
                pp, mbc, forward_times, backward_times,
                draw=draw, output_path=output_path,
            )
        return self.calculate_1f1b_bubble(
            pp, mbc, forward_times, backward_times,
            draw=draw, output_path=output_path,
        )

    def draw_pp_gantt(self, output_path=None):
        """Render the Gantt chart for the configured PP schedule.

        Dispatches on ``strategy.pp_schedule`` and returns the simulated
        iteration time. Honours the chosen schedule's per-rank timing
        decomposition (fused B for 1F1B, split B/W for ZB-H2).
        """
        return self._compute_pp_total_time(draw=True, output_path=output_path)
    
    def _analysis_single_iter_cost_impl(self):
        # we construct the result in the following hierarchy:
        # first level: useful FlopS、mfu、all FlopS、throughput、duration_per_iter
        # second level: time break down = compute time + comm_time + bubble_time
        # third level-0:  compute time = fwd time + recom_time + bwd_time + optim update time
        # third level-1:  comm_time_: tp_time(tp_time、tp_time_can_overlap) + pp_time
        all_result = {}
        single_batch_cost = self._analysis_single_batch_cost_impl(
            enable_recompute=self.strategy.enable_recompute, model_name = FIRST_CHUNK
        )
        # 1.comm_result： dp_time + fwd/bwd/recompute net time + pp_time
        gbs_comm_in_first_stage = self._analysis_gbs_comm_time(single_batch_cost, model_name = FIRST_CHUNK)
        # 2.compute result: 
        gbs_compute_cost_in_first_stage = self._analysis_gbs_compute_time(
            single_batch_cost, model_name = FIRST_CHUNK
        )
        # 3. all time
        # can't be overlap for now
        chunk_time = self._compute_single_batch_fwd_bwd_time(FIRST_CHUNK, chunk=True)
        if self.strategy.pp_size > 1:
            single_batch_cost = self._analysis_single_batch_cost_impl(
                    enable_recompute=self.strategy.enable_recompute, model_name=LAST_CHUNK
                )
            gbs_comm_result_in_last_stage = self._analysis_gbs_comm_time(
                single_batch_cost, model_name=LAST_CHUNK
            )
            gbs_compute_result_in_last_stage = self._analysis_gbs_compute_time(
                single_batch_cost, model_name=LAST_CHUNK
            )
            chunk_time_lstage = self._compute_single_batch_fwd_bwd_time(LAST_CHUNK, chunk=True)
        
        breakdown_result = {}
        breakdown_result["fwd_compute_time"] = gbs_compute_cost_in_first_stage["fwd_compute_time"]
        breakdown_result["recompute_time"] = gbs_compute_cost_in_first_stage["recompute_time"]
        breakdown_result["bwd_compute_time"] = gbs_compute_cost_in_first_stage["bwd_compute_time"]
        breakdown_result["optim_time"] = gbs_compute_cost_in_first_stage["optim_time"][
            "optim_exposed_time"
        ]
        breakdown_result["intra_exposed_time"] = gbs_comm_in_first_stage["intra_comm_time"][
            "intra_exposed_time"
        ]
        breakdown_result["inter_exposed_time"] = gbs_comm_in_first_stage["inter_comm_time"][
            "inter_exposed_time"
        ]
        breakdown_result["dp_exposed_time"] = gbs_comm_in_first_stage["dp_comm_time"][
            "dp_comm_exposed_time"
        ]

        if self.strategy.pp_size > 1:
            breakdown_result_last_stage = {}
            breakdown_result_last_stage["fwd_compute_time"] = gbs_compute_result_in_last_stage["fwd_compute_time"]
            breakdown_result_last_stage["recompute_time"] = gbs_compute_result_in_last_stage["recompute_time"]
            breakdown_result_last_stage["bwd_compute_time"] = gbs_compute_result_in_last_stage["bwd_compute_time"]
            breakdown_result_last_stage["optim_time"] = gbs_compute_result_in_last_stage["optim_time"][
                "optim_exposed_time"
            ]
            breakdown_result_last_stage["intra_exposed_time"] = gbs_comm_result_in_last_stage["intra_comm_time"][
                "intra_exposed_time"
            ]
            breakdown_result_last_stage["inter_exposed_time"] = gbs_comm_result_in_last_stage["inter_comm_time"][
                "inter_exposed_time"
            ]
            breakdown_result_last_stage["dp_exposed_time"] = gbs_comm_result_in_last_stage["dp_comm_time"][
                "dp_comm_exposed_time"
            ]
            
            if self.strategy.pp_size > 2:
                chunk_time_middle_stage = self._compute_single_batch_fwd_bwd_time(MIDDLE_CHUNK, chunk=True)
            else:
                chunk_time_middle_stage = 0
            
            all_result["breakdown_result_last_stage"] = breakdown_result_last_stage

        # 4.compute first level
        model_flops = gbs_compute_cost_in_first_stage["model_flops"]

        # ------------------------- SUMMRY -------------------------
        pp_size = self.strategy.pp_size
        dense_param_numel = self.model_chunk_dict[FIRST_CHUNK]._model_info.weight_numel + (
                            self.model_chunk_dict[MIDDLE_CHUNK]._model_info.weight_numel if pp_size > 2 else 0
                        ) * (pp_size - 2) + (
                            self.model_chunk_dict[LAST_CHUNK]._model_info.weight_numel if pp_size > 1 else 0
                        )
        moe_param_numel = self.model_chunk_dict[FIRST_CHUNK]._model_info.moe_weight_numel + (
                            self.model_chunk_dict[MIDDLE_CHUNK]._model_info.moe_weight_numel if pp_size > 2 else 0
                        ) * (pp_size - 2) + (
                            self.model_chunk_dict[LAST_CHUNK]._model_info.moe_weight_numel if pp_size > 1 else 0
                        )
        
        def get_dp_and_optim(model_chunk):
            t = self._compute_dp_time(model_chunk)['dp_comm_exposed_time']
            t += self._compute_optim_time(model_chunk)['optim_exposed_time']
            return t
        single_iter_time_no_dp_opim = self._compute_pp_total_time()
        duration_times = [single_iter_time_no_dp_opim + get_dp_and_optim(FIRST_CHUNK)]
        duration_times.append(single_iter_time_no_dp_opim + get_dp_and_optim(MIDDLE_CHUNK)) if self.strategy.pp_size > 2 else 0
        duration_times.append(single_iter_time_no_dp_opim + get_dp_and_optim(LAST_CHUNK)) if self.strategy.pp_size > 1 else 0

        final_duration_time_per_iter = max(duration_times)
        # When seq_len_std > 0, total tokens per iter is the sum across
        # sampled microbatches rather than nominal seq_len * global_batch.
        if self.seq_lens is not None and self.disturbance.seq_len_std > 0.0:
            tokens_per_mb_slot = int(self.seq_lens.sum()) * self.strategy.micro_batch_size
            all_tokens_per_iter = tokens_per_mb_slot * self.strategy.dp_size
        else:
            all_tokens_per_iter = self.strategy.seq_len * self.strategy.global_batch_size

        theory_flops_per_token = self.model_config.flops_per_token(context_seq_len=self.strategy.seq_len, with_attn=True)
        theory_flops = self.model_config.flops_per_token(context_seq_len=self.strategy.seq_len, with_attn=True) * all_tokens_per_iter //  self.strategy.world_size
        TGS = all_tokens_per_iter/(final_duration_time_per_iter/1000)/self.strategy.world_size
        TFLOPS = theory_flops / (final_duration_time_per_iter/1000)/1e12
        TFLOPS_PER_TOKEN = theory_flops_per_token / (final_duration_time_per_iter/1000)/1e12
        new_mfu_6nd_with_attn = TFLOPS / self.system.accelerator.op["default"].tflops
        
        mbc = self.strategy.micro_batch_num
        all_result["comm_details"] = gbs_comm_in_first_stage
        all_result["compute_details"] = gbs_compute_cost_in_first_stage
        all_result["breakdown_result"] = breakdown_result
        all_result["all_tokens_per_iter"] = all_tokens_per_iter

        def chunk_total_work(model_chunk):
            t = 0.0
            for m in range(mbc):
                s = int(self.seq_lens[m])
                fwd_m, bwd_m = self._chunk_fwd_bwd_at(model_chunk, s)
                t += fwd_m + bwd_m
            return t

        # Cache per-chunk totals so we can both populate format_chunk_time and
        # aggregate across ranks for pp_utilization without recomputing.
        chunk_work = {FIRST_CHUNK: chunk_total_work(FIRST_CHUNK)}
        if pp_size > 2:
            chunk_work[MIDDLE_CHUNK] = chunk_total_work(MIDDLE_CHUNK)
        if pp_size > 1:
            chunk_work[LAST_CHUNK] = chunk_total_work(LAST_CHUNK)

        def format_chunk_time(model_chunk, max_chunk_time, duration_time):
            # max_chunk_time: F+B at the nominal (== max(seq_lens) when seq_len_std > 0)
            # seq_len — an upper bound per microbatch used for memory sizing.
            # avg_chunk_time: sum_m (F+B at actual seq_lens[m]) / mbc. Equal to
            # max_chunk_time under constant seq_lens; strictly smaller when seq_lens vary.
            # bubble_time uses the sum of per-microbatch work so it reflects the true idle
            # on that rank (max_iter_time − actual useful work), avoiding the over-estimate
            # that mbc × max_chunk_time would produce under variable seq_lens.
            total_work = chunk_work[model_chunk]
            avg_chunk_time = total_work / mbc if mbc > 0 else 0.0
            return {
                model_chunk:{
                    'duration_time(avg_chunk_timexmbc+dp_optim+bubble)': duration_time,
                    'avg_chunk_time(fwd+bwd)': avg_chunk_time,
                    'max_chunk_time(fwd+bwd)': max_chunk_time,
                    'dp_and_optim_time': get_dp_and_optim(model_chunk),
                    'bubble_time': single_iter_time_no_dp_opim - total_work
                }
            }
        all_result['all_chunk_times'] = format_chunk_time(FIRST_CHUNK, chunk_time, duration_times[0])
        all_result['all_chunk_times'].update(format_chunk_time(MIDDLE_CHUNK, chunk_time_middle_stage, duration_times[1]) if pp_size > 2 else {})
        all_result['all_chunk_times'].update(format_chunk_time(LAST_CHUNK, chunk_time_lstage, duration_times[-1]) if pp_size > 1 else {})

        # pp_utilization = Σ_r work_r / (pp × max_iter_time). 1.0 means every rank is
        # busy for the whole iteration (zero bubble); lower values mean more idle time
        # distributed across ranks. Complement is the bubble fraction.
        total_work_all_ranks = chunk_work[FIRST_CHUNK]
        if pp_size > 2:
            total_work_all_ranks += (pp_size - 2) * chunk_work[MIDDLE_CHUNK]
        if pp_size > 1:
            total_work_all_ranks += chunk_work[LAST_CHUNK]
        pp_utilization = (total_work_all_ranks / (pp_size * single_iter_time_no_dp_opim)
                          if single_iter_time_no_dp_opim > 0 else 1.0)

        all_result.update({
            'duration_time_per_iter': final_duration_time_per_iter,
            'throughput_per_accelerator': TGS,
            'throughput per GPU (TFLOP/s/GPU)': TFLOPS,
            'throughput per GPU per token (TFLOP/s/GPU/token)': TFLOPS_PER_TOKEN,
            'mfu_6nd_with_attn': new_mfu_6nd_with_attn,
            'mfu':new_mfu_6nd_with_attn,
            'pp_utilization': pp_utilization,
            'moe_param_numel': f'{moe_param_numel/1e9:.2f}B',
        })
        all_result['flops_info'] = {
            'theory_flops': theory_flops,
            # 'theory_flops_per_token': theory_flops_per_token,
            'model_flops': model_flops,
        }

        all_result['param_numel_info'] = {
            "dense" : f'{dense_param_numel/1e9:.2f}B',
            "moe"    : f'{moe_param_numel/1e9:.2f}B',
            "all"    : f'{(dense_param_numel+moe_param_numel)/1e9:.2f}B',
        }

        if self.model_config.model_type == 'moe':
            activaton_params_numel = dense_param_numel + moe_param_numel * (self.model_config.topk / self.model_config.expert_num)
            activaton_ratio = activaton_params_numel/(dense_param_numel+moe_param_numel)
            all_result['param_numel_info'].update({
                    "activations" : f'{activaton_params_numel/1e9:.2f}B',
                    "activations_ratio" : f'{activaton_ratio*100:.2f}%',
                }
            )
        
        # convert to format
        convert_final_result_to_human_format(all_result)
        return all_result

    def analysis_cost(self):
        result = self._analysis_single_iter_cost_impl()
        return Result(result)

    def analysis_gemm_costs(self):
        def merge_gemm_costs(gemm_costs1, gemm_costs2):
            for key in gemm_costs1:
                gemm_costs1[key].extend(gemm_costs2[key])
            return gemm_costs1

        gemm_costs = self.model_chunk_dict['first_stage_chunk'].get_all_gemm_cost_info()
        if self.strategy.pp_size > 1:
            last_gemm_costs = self.model_chunk_dict['last_stage_chunk'].get_all_gemm_cost_info()
            gemm_costs = merge_gemm_costs(gemm_costs, last_gemm_costs)
        if self.strategy.pp_size > 2:
            middle_gemm_costs = self.model_chunk_dict['middle_stage_chunk'].get_all_gemm_cost_info() 
            for _ in range(self.strategy.pp_size -2):
                gemm_costs = merge_gemm_costs(gemm_costs, middle_gemm_costs)
        return gemm_costs
    
    def analysis_op_info(self):
        """
        """
        op_infos = {}
        for key in self.model_chunk_dict:
            op_infos[key] = self.model_chunk_dict[key].get_all_gemm_cost_info()
        return op_infos
    
    def _build_input_info(self, chunk_name: str, seq_len: int) -> InputOutputInfo:
        """Build the ``InputOutputInfo`` fed to a model chunk for a given seq_len.

        The first chunk receives a 2-D ``(mbs, seq_len)`` token-id tensor; the
        middle and last chunks receive ``(mbs, seq_len_after_sp, hidden)``
        activation tensors, where ``seq_len_after_sp = seq_len // tp_size`` when
        sequence parallelism is enabled.
        """
        mbs = self.strategy.micro_batch_size
        if chunk_name == FIRST_CHUNK:
            return InputOutputInfo(tensors=[TensorSize(shape=(mbs, seq_len))])
        sp_seq_len = (
            seq_len // self.strategy.tp_size
            if self.strategy.enable_sequence_parallel
            else seq_len
        )
        return InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(mbs, sp_seq_len, self.model_config.hidden_size)
                )
            ]
        )

    def _invoke_chunk(self, chunk_name: str, seq_len: int, path_ctx: PathDebugContext):
        """Invoke a chunk with a seq_len-specific input; returns nothing.

        Side effect: refreshes ``self.model_chunk_dict[chunk_name]._cost_info``
        and activation info to reflect the given seq_len.
        """
        input_info = self._build_input_info(chunk_name, seq_len)
        _ = self.model_chunk_dict[chunk_name](input_info, path_ctx)

    def _run(self):
        # Use the nominal seq_len for the primary pass (memory/activation info).
        # For variable seq_len simulations the per-microbatch *timing* is
        # recomputed on demand via _chunk_fwd_b_w_at(); memory uses the
        # conservative max(seq_lens) substitution (see _nominal_seq_len_for_mem).
        nominal_seq_len = self._nominal_seq_len_for_mem()

        self.path_debug_context = PathDebugContext(
            point_datas={},
            point_datas_with_recomp={},
            target_point=self.debug_points,
            path_list=[],
        )
        self._invoke_chunk(FIRST_CHUNK, nominal_seq_len, self.path_debug_context)
        self.pp_state_peak_point[FIRST_CHUNK] = self.model_chunk_dict[FIRST_CHUNK].compute_activations()

        if self.strategy.pp_size > 2:
            self.path_debug_context_last_stage = PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                path_list=[],
            )
            self._invoke_chunk(MIDDLE_CHUNK, nominal_seq_len,
                               self.path_debug_context_last_stage)
            self.pp_state_peak_point[MIDDLE_CHUNK] = self.model_chunk_dict[MIDDLE_CHUNK].compute_activations()

        if self.strategy.pp_size > 1:
            self.path_debug_context_last_stage = PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                target_point=self.debug_points_last_stage,
                path_list=[],
            )
            self._invoke_chunk(LAST_CHUNK, nominal_seq_len,
                               self.path_debug_context_last_stage)
            self.pp_state_peak_point[LAST_CHUNK] = self.model_chunk_dict[LAST_CHUNK].compute_activations()

    def _nominal_seq_len_for_mem(self) -> int:
        """Seq_len used for the memory/activation pass.

        When seq_len_std == 0 this is just strategy.seq_len. When std > 0 we
        use ``max(self.seq_lens)`` so the memory peak is conservative w.r.t.
        the worst microbatch in the simulation.
        """
        if self.seq_lens is not None and self.disturbance.seq_len_std > 0.0:
            return int(self.seq_lens.max())
        return self.strategy.seq_len

    def get_pp_stage_peak_mem(self, mem_result, peak_mem_key, toG:bool = False):
        assert peak_mem_key in ["peak_mem_with_reserved", "peak_mem"], f"peak_mem_key should be in ['peak_mem_with_reserved', 'peak_mem'] but got {peak_mem_key}"
        pp_size = self.strategy.pp_size
        if pp_size == 1:
            peak_mem =  HumanReadableSize.from_string(
                mem_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
            return dict(
                first_stage = peak_mem/ 1024 / 1024 / 1024 if toG else peak_mem, 
            )
        
        peak_mem_list = dict()
        if pp_size> 1:
            first_stage_mem_result = mem_result.get("first_stage")
            first_stage_peak_cached_mem = HumanReadableSize.from_string(
                first_stage_mem_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
            peak_mem_list['first_stage'] = first_stage_peak_cached_mem

            last_stage_mem_result = mem_result.get("last_stage")
            last_stage_peak_cached_mem = HumanReadableSize.from_string(
                last_stage_mem_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
            peak_mem_list['last_stage'] = last_stage_peak_cached_mem
        if pp_size > 2:
            middle_stage_mem_result = mem_result.get("middle_stage")
            middle_stage_peak_cached_mem = HumanReadableSize.from_string(
                middle_stage_mem_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
            peak_mem_list['middle_stage'] = middle_stage_peak_cached_mem
        if toG:
            for key in peak_mem_list:
                peak_mem_list[key] = peak_mem_list[key] / 1024**3
        return peak_mem_list
    
    def search_max_micro_batch_size(self, micro_batch_num = None):
        """
        Fixes `micro_batch_count` and searches for the largest possible `micro_batch_size` under the current parallel strategy.
        """
        left = 1
        right = 2**16
        accelerator_mem_bytes = self.system.accelerator.mem_gbs * 1024**3
        origin_micro_batch_size = self.strategy.micro_batch_size
        origin_micro_batch_num = self.strategy.micro_batch_num

        self.strategy.micro_batch_num = self.strategy.pp_size * 16 if micro_batch_num is None else micro_batch_num # TODO(sherry): batch_num is not the same as pp_size, change 1000 to 16
        while left < right:
            micro_batch_size = left + ((right - left) >> 1)
            self.strategy.micro_batch_size = micro_batch_size
            # run
            self.run_estimate()
            # mem analysis
            mem_result = self.analysis_mem()
            if mem_result.get("first_stage") is None:
                peak_cached_mem_bytes = HumanReadableSize.from_string(
                    mem_result.get("peak_mem"),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
            else:
                first_stage_mem_result = mem_result.get("first_stage")
                first_stage_peak_cached_mem = HumanReadableSize.from_string(
                    first_stage_mem_result.get("peak_mem"),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
                last_stage_mem_result = mem_result.get("last_stage")
                last_stage_peak_cached_mem = HumanReadableSize.from_string(
                    last_stage_mem_result.get("peak_mem"),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
                peak_cached_mem_bytes = max(
                    first_stage_peak_cached_mem, last_stage_peak_cached_mem
                )
            
            if peak_cached_mem_bytes > accelerator_mem_bytes:
                right = micro_batch_size
            else:
                left = micro_batch_size + 1
        max_micro_batch_size = left - 1
        print(f"cur micro_batch_size: {micro_batch_size}, micro_batch_num: {self.strategy.micro_batch_num}")
        print(f"Peak cached memory: {peak_cached_mem_bytes/1024**3: .2f} GB")
        self.strategy.micro_batch_size = origin_micro_batch_size
        self.strategy.micro_batch_num = origin_micro_batch_num
        return max_micro_batch_size, peak_cached_mem_bytes

    def search_max_micro_batch_size_fixed_gbs(self, pp_size, dp_size, global_batch_size, memory_utils = 1.0, gmi_error=6, use_reserved_memory=True, save_all=True): 
        """
        Fixes `global_batch_size` and searches for the maximum possible `micro_batch_size` under the current parallel strategy.
        """
        gmi_error = gmi_error * 1024**3
        PEAK_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list = [], [], [], []
        accelerator_mem_bytes = self.system.accelerator.mem_gbs * 1024**3 * memory_utils
        origin_micro_batch_size = self.strategy.micro_batch_size
        origin_micro_batch_num = self.strategy.micro_batch_num
        
        for micro_batch_size in range(global_batch_size-1, 0, -1):
            micro_batch_num  = global_batch_size // (micro_batch_size * dp_size)
            if global_batch_size %  (micro_batch_size * dp_size) != 0:
                continue    

            if global_batch_size % micro_batch_size != 0 or micro_batch_num < pp_size:
                continue
           
            self.strategy.micro_batch_num = micro_batch_num 
            self.strategy.micro_batch_size = micro_batch_size

            # run
            rm_tmp()
            self.run_estimate()
            # mem analysis
            mem_result = self.analysis_mem()
            if mem_result.get("first_stage") is None:
                peak_cached_mem_bytes = HumanReadableSize.from_string(
                    mem_result.get(PEAK_KEY),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
            else:
                first_stage_mem_result = mem_result.get("first_stage")
                first_stage_peak_cached_mem = HumanReadableSize.from_string(
                    first_stage_mem_result.get(PEAK_KEY),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
                last_stage_mem_result = mem_result.get("last_stage")
                last_stage_peak_cached_mem = HumanReadableSize.from_string(
                    last_stage_mem_result.get(PEAK_KEY),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
                peak_cached_mem_bytes = max(
                    first_stage_peak_cached_mem, last_stage_peak_cached_mem
                )
            if peak_cached_mem_bytes + gmi_error <= accelerator_mem_bytes:
                search_micro_batch_size = micro_batch_size
                search_micro_batch_num = micro_batch_num
                cost_result = self.analysis_cost()
                peak_mem_list = self.get_pp_stage_peak_mem(self.analysis_mem(), PEAK_KEY, toG=True)

                if save_all:
                    all_search_micro_batch_size.append(search_micro_batch_size)
                    all_search_micro_batch_num.append(search_micro_batch_num)
                    all_peak_cached_mem_list.append(peak_mem_list)
                    all_cost_list.append(cost_result)
                else:
                    return [search_micro_batch_size], [search_micro_batch_num], [peak_mem_list], [cost_result]

        self.strategy.micro_batch_size = origin_micro_batch_size
        self.strategy.micro_batch_num = origin_micro_batch_num
        
        return all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list


    def log_available_strategy(self, mfu, peak_mem):
        print(f"Find result  parallelism={self.strategy.parallelism}, pp_num_layers={self.get_pp_num_layers()}, recompute={self.strategy.recompute_status},mfu={mfu} gbs={self.strategy.global_batch_size} peak_cached_mem_bytes={peak_mem}GB", flush=True)

    def get_pp_num_layers(self):
        num_layers_per_pp = math.ceil(self.model_config.layer_num/self.strategy.pp_size)
        pp_num_layers = f'[{num_layers_per_pp}]x{self.strategy.pp_size-1} + [{self.strategy.num_layers_in_last_pipeline_stage}]' if self.strategy.pp_size > 1 else [self.model_config.layer_num]
        return pp_num_layers

    def dump_paralism_and_recompute_perf(self, mem_result, cost_result):
        # from pprint import pprint
        # pprint(mem_result.data)
        dtype = 'fp8' if self.strategy.fp8 else 'bf16'
        perf = {
            'model_name': self.model_config.model_name,
            'system': self.system.sys_name,
            'parallelism': f'{dtype}.dense{self.model_config.dense_layers}.{self.strategy.parallelism}',
            'recompute_status': self.strategy.recompute_status,
            'mfu': cost_result.data["mfu_6nd_with_attn"],
            'TFLOPS': cost_result.data['throughput per GPU (TFLOP/s/GPU)'],
            'TGS_per_gpu' : cost_result.data['throughput_per_accelerator'],
            'iter_time':  cost_result.data["duration_time_per_iter"],
            'peak_mem':  mem_result.data["peak_mem"] if "peak_mem" in mem_result.data else {s:v['peak_mem'] for s,v in mem_result.data.items()},
            'peak_mem_with_reserved':  mem_result.data["peak_mem_with_reserved"] if "peak_mem_with_reserved" in mem_result.data else {s:v['peak_mem_with_reserved'] for s,v in mem_result.data.items()}
        }
        return perf
    
    def dump_paralism_and_recompute_bw_perf(self, mem_result, cost_result):
        perf = self.dump_paralism_and_recompute_perf(mem_result, cost_result)
        perf['comm_bw_info'] = str(deepcopy(self.system.real_comm_bw))
        # perf['estimate_details'] = {
        #                 'mem_result': str(mem_result),
        #                 'compute_result': str(cost_result),
        #                 'model_arch':str(self.model_chunk_dict),
        #                 'strategy_config': str(self.strategy),
        #                 'system_config': str(self.system),
        #                 'model_config': str(self.model_config)
        #             }
        return perf
    
    def search_best_selective_recompute(self, use_reserved_memory, gmi_error, best_mfu=None, all_search_result = None, save_path = None):
        self.strategy.recompute_granularity = "selective_recompute"
        accelerator_mem_gbytes = self.system.accelerator.mem_gbs  - gmi_error # gmi has 6 GB error

        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        from itertools import product
        best_strategy = {}
        params = ['attn_recompute', 'mla_rms_recompute', 'mlp_recompute', 'mlp_rms_recompute']
        combinations = [dict(zip(params, combo)) for combo in product([False, True], repeat=4)]
        combinations = [
            {
                'mla_rms_recompute': True,
                'attn_recompute': True,
                'mlp_rms_recompute': True,
                'mlp_recompute': True,
            },
            {
                'mla_rms_recompute': True,
                'attn_recompute': True,
                'mlp_rms_recompute': False,
                'mlp_recompute': False,
            },
            {
                'mla_rms_recompute': False,
                'attn_recompute': False,
                'mlp_rms_recompute': True,
                'mlp_recompute': True,
            },
        ]
        for recompute_params in combinations:
            self.strategy.attn_recompute = recompute_params['attn_recompute']
            self.strategy.mla_rms_recompute = recompute_params['mla_rms_recompute']
            self.strategy.mlp_recompute = recompute_params['mlp_recompute']
            self.strategy.mlp_rms_recompute = recompute_params['mlp_rms_recompute']

            self.run_estimate()
            mem_result = self.analysis_mem()
            cost_result = self.analysis_cost()
            peak_mem_list = self.get_pp_stage_peak_mem(mem_result, PEAK_MEM_KEY, toG=True)
            peak_cached_mem_gbytes = max(peak_mem_list.values())
            if peak_cached_mem_gbytes <= accelerator_mem_gbytes:
                cur_perf = self.dump_paralism_and_recompute_bw_perf(mem_result, cost_result)
                if cur_perf['mfu'] > best_mfu:
                    best_mfu = cur_perf['mfu']
                    best_strategy = cur_perf
                    self.log_available_strategy(cost_result.data['mfu'], peak_cached_mem_gbytes)
                    if save_path is not None:
                        self._dump_memory_and_cost(mem_result, cost_result, save_path)
                if all_search_result is not None:
                    merge_dict(cur_perf, all_search_result)
        return best_strategy

    def search_best_recompute_layer_num(self, 
                                        layer_num, 
                                        use_reserved_memory: bool, 
                                        gmi_error:int,
                                        best_mfu,
                                        all_search_result:dict,
                                        save_path = None):
        """
         Searches for the number of full recompute layers of the highest MFU that can be placed in memory under the current micro_batch_size, micro_batch_count, and parallel policies. 

        Args:
            layer_num (int): layer number
            use_reserved_memory (bool): whether to use reserved memory
            gmi_error (int): The error between gmi and the actual allocated storage
            best_mfu (float): best mfu
            all_search_result (dict): all search result
        Returns:
            dict: search result
        """
        accelerator_mem_gbytes = self.system.accelerator.mem_gbs  - gmi_error # gmi has 6 GB error
        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        best_strategy = dict()
        left, right = 0, math.ceil(layer_num/self.strategy.pp_size)
        # right = min(right, layer_num-1)
        ori_recompute_layer_num = self.strategy.recompute_layer_num 

        while left <= right:
            recompute_layer_num = (left + right) // 2  

            max_recompute_layer_num = math.ceil(layer_num / self.strategy.pp_size)
            assert recompute_layer_num <= max_recompute_layer_num, f'recompute_layer_num: {recompute_layer_num}, max_recompute_layer_num={max_recompute_layer_num}, layer_num: {layer_num}, pp_size: {self.strategy.pp_size}'
            self.strategy.recompute_layer_num = recompute_layer_num

            rm_tmp()
            self.run_estimate()
            mem_result = self.analysis_mem()
            cost_result = self.analysis_cost()
            peak_mem_list = self.get_pp_stage_peak_mem(mem_result, PEAK_MEM_KEY, toG=True)
            peak_cached_mem_gbytes = max(peak_mem_list.values())
            if peak_cached_mem_gbytes > accelerator_mem_gbytes:
                left = recompute_layer_num + 1
            else:
                right = recompute_layer_num - 1
                # Save best search results
                if cost_result.data['mfu'] >= best_mfu:
                    best_mfu = cost_result.data['mfu']
                    best_strategy = self.dump_paralism_and_recompute_bw_perf(mem_result, cost_result)
                    self.log_available_strategy(cost_result.data['mfu'], peak_cached_mem_gbytes)
                    if save_path is not None:
                        self._dump_memory_and_cost(mem_result, cost_result, save_path)

                if all_search_result is not None:
                    perf = self.dump_paralism_and_recompute_bw_perf(mem_result, cost_result)
                    merge_dict(perf, all_search_result)

        self.strategy.recompute_layer_num = ori_recompute_layer_num # recompute_layer_num

        return best_strategy
    
    def search_best_strategy_no_recompute(self, gmi_error, use_reserved_memory, best_mfu, all_search_result, save_path = None):
        self.strategy.recompute_granularity = None
        self.strategy.recompute_layer_num = 0
        accelerator_mem_gbytes = self.system.accelerator.mem_gbs  - gmi_error
          # gmi has 6 GB error
        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        best_strategy = dict()
        self.run_estimate()
        mem_result = self.analysis_mem()
        cost_result = self.analysis_cost()
        peak_mem_list = self.get_pp_stage_peak_mem(mem_result, PEAK_MEM_KEY, toG=True)
        peak_cached_mem_gbytes = max(peak_mem_list.values())
        if peak_cached_mem_gbytes <= accelerator_mem_gbytes:
            cur_strategy = self.dump_paralism_and_recompute_bw_perf(mem_result, cost_result)
            merge_dict(cur_strategy, all_search_result)

            if cost_result.data['mfu'] >  best_mfu:
                best_mfu = cost_result.data['mfu']
                best_strategy = cur_strategy
                self.log_available_strategy(cost_result.data['mfu'], peak_cached_mem_gbytes)
                if save_path is not None:
                    self._dump_memory_and_cost(mem_result, cost_result, save_path)

        return best_strategy

    def search_best_parallel_strategy_with_recompute(self,
                                                 world_size:int,  
                                                 gmi_error:int,
                                                 micro_batch_size:int,
                                                 global_batch_size:int, 
                                                 all_search_result:dict,
                                                 tp_search_list:List = None,
                                                 ep_search_list:List = None,
                                                 pp_search_list:List = None,
                                                 use_etp:bool = False,
                                                 recompute_search_type:str = ['no_recompute', 'full_block', 'selective_recompute'],
                                                 use_reserved_memory: bool = True,
                                                 dump_path:str=None):
        """
        Searches for the optimal combination of parallel strategies (tp/ep/pp) and full recompute layer configuration that maximizes performance under fixed global batch size constraints.

        Args:
            world_size (int): world size
            gmi_error (int): The error between gmi and the actual allocated storage
            micro_batch_size (int):  fixed micro batch size
            global_batch_size (int): fixed global batch size
            all_search_result (dict): all search result of this model, include (tp, ep, pp, recompute_layer_num) combination.
        Returns:
            best_strategy (dict): the best strategy of this model, include (tp, ep, pp, full_recompute_layer_num) combination.
        """
        
        # world_size = 256 GPUs
        # tp: 1 2 4 8
        # ep: 1 2 4 8
        # and layer_num must be divisible
        if not isinstance(recompute_search_type, list):
            recompute_search_type = [recompute_search_type]

        layer_num = self.model_config.layer_num
        if tp_search_list is None:
            tp_search_list = [1, 2, 4, 8]  if self.model_config.model_type == "dense" else [1]
        if  ep_search_list is None:
            ep_search_list = [1, 2, 4, 8] if self.model_config.model_type == "moe" else [1]
        if pp_search_list is None:
            pp_search_list = list(range(1, layer_num+1))    

        global_best_strategy = {}
        best_strategy_cost_path = f"{dump_path}/best_strategy_costs"

        print(f"Start search strategy for world_size={world_size}, model_type={self.model_config.model_type}, model_name={self.model_config.model_name}, system={self.system.sys_name}")
        print(f"- tp_search_list={tp_search_list}, ep_search_list={ep_search_list}, pp_search_list={pp_search_list}")
        print(f"- layer_num={layer_num}")
        print(f"- moe_pad_expert_input_to_capacity={self.model_config.moe_pad_expert_input_to_capacity}")
        print(f"- capacity={self.model_config.capacity}")

        global_best_mfu = -1
        for tp_size in tp_search_list:
            for ep_size in ep_search_list:
                for pp_size in pp_search_list:
                    is_tp_valid = self.model_config.head_num % tp_size == 0 and self.model_config.kv_head_num % tp_size == 0
                    is_dp_valid =  world_size % (pp_size * tp_size) == 0
                    dp_size = world_size // (pp_size * tp_size)
                    is_ep_valid = dp_size % ep_size == 0
                    etp_size = tp_size if use_etp else 1
                    is_etp_valid = (world_size %(ep_size* etp_size) == 0) and (etp_size*ep_size < self.system.num_per_node) 
                    is_etp_valid = (world_size %(ep_size* etp_size) == 0) # TODO(sherry): temporarily limit etp_size*ep_size < self.system.num_per_node

                    if pp_size > 1:
                        num_layers_per_pp = math.ceil(layer_num/pp_size)
                        is_pp_valid = num_layers_per_pp > 0
                        num_layers_in_last_pipeline_stage = layer_num - (num_layers_per_pp * (pp_size - 1))
                        is_pp_valid = is_pp_valid and num_layers_in_last_pipeline_stage > 0
                    else:
                        num_layers_in_last_pipeline_stage = None
                        is_pp_valid = True
                    
                    if is_dp_valid and is_tp_valid and is_ep_valid and is_etp_valid and is_pp_valid:
                        # set strategy  
                        self.strategy.world_size = world_size
                        self.strategy.tp_size = tp_size
                        self.strategy.ep_size = ep_size
                        self.strategy.pp_size = pp_size
                        self.strategy.etp_size = etp_size
                        self.strategy.num_layers_in_first_pipeline_stage = None
                        self.strategy.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage

                
                        search_micro_batch_num = global_batch_size // (self.strategy.dp_size * micro_batch_size)
                        
                        if global_batch_size % (self.strategy.dp_size * micro_batch_size) != 0:
                            continue
                        self.strategy.micro_batch_num = search_micro_batch_num
                        self.strategy.micro_batch_size = micro_batch_size

                        if micro_batch_size != 0 and search_micro_batch_num != 0:
                            for recompute_type in recompute_search_type:
                                if recompute_type == 'no_recompute':
                                    self.strategy.recompute_granularity = None
                                    self.strategy.recompute_layer_num = 0
                                    self.strategy.recompute_variance = True 
                                    search_best_strategy = self.search_best_strategy_no_recompute(gmi_error=gmi_error,
                                                                                                       use_reserved_memory=use_reserved_memory,
                                                                                                       best_mfu=global_best_mfu,
                                                                                                       all_search_result=all_search_result)
                                elif recompute_type == 'full_block':
                                    self.strategy.recompute_granularity = "full_block"
                                    self.strategy.recompute_variance = False # megatron-LM's full recompute does not support variance
                                    search_best_strategy = self.search_best_recompute_layer_num(
                                                                            layer_num=self.model_config.layer_num, 
                                                                            use_reserved_memory = use_reserved_memory,
                                                                            gmi_error=gmi_error,
                                                                            best_mfu=global_best_mfu,
                                                                            all_search_result=all_search_result,
                                                                            save_path=best_strategy_cost_path)
                                elif recompute_type == 'layer_only':
                                    # recompute_granularity is defined by user, we only search the best layer num
                                    search_best_strategy = self.search_best_recompute_layer_num(
                                                                            layer_num=self.model_config.layer_num, 
                                                                            use_reserved_memory = use_reserved_memory,
                                                                            gmi_error=gmi_error,
                                                                            best_mfu=global_best_mfu,
                                                                            all_search_result=all_search_result,
                                                                            save_path=best_strategy_cost_path)
                                elif recompute_type == 'selective_recompute':
                                    self.strategy.recompute_granularity = "selective_recompute"
                                    self.strategy.recompute_layer_num = math.ceil(layer_num/pp_size)
                                    search_best_strategy = self.search_best_selective_recompute(
                                        use_reserved_memory=use_reserved_memory,
                                        gmi_error=gmi_error,
                                        best_mfu=global_best_mfu,
                                        all_search_result=all_search_result,
                                        save_path=best_strategy_cost_path
                                    )
                                else:
                                    raise NotImplementedError(f'recompute strategy {recompute_search_type} not implemented')
                                
                                if search_best_strategy and 'mfu' in search_best_strategy:
                                    global_best_strategy = search_best_strategy
                                    global_best_mfu = search_best_strategy['mfu']

        if dump_path is not None and len(global_best_strategy) > 0:
            model_name = self.model_config.model_name
            system_name = self.system.sys_name
            os.makedirs(dump_path, exist_ok=True)


            if 'peak_mem' in global_best_strategy and isinstance(global_best_strategy['peak_mem'], dict):
                global_best_strategy['peak_mem'] = str(global_best_strategy['peak_mem']) # serialize dict to string to avoid csv dump error
            best_strategy_df = pd.DataFrame(global_best_strategy, index=[0])
            best_strategy_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seqlen{self.strategy.seq_len}_worldsize{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_best_strategy.csv") 
            print(best_strategy_df)

            if all_search_result is not None:
                all_search_result_df = pd.DataFrame(all_search_result)
                all_search_result_df = all_search_result_df.sort_values(by ='mfu',  ascending=False)
                all_search_result_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seqlen{self.strategy.seq_len}_worldsize{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_all_search_strategies.csv")
            
        return global_best_strategy                        
    
    def _dump_memory_and_cost(self, mem_result:dict, compute_result:dict, save_path:str):
        print(f"Saving analysis results to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        base_info = {}
        base_info["arch"] = str(self.model_chunk_dict)
        base_info["all_param"] = self.model_config.param_numel
        base_info["act_param"] = self.model_config.activated_param_numel
        with open(f"{save_path}/model_arch", "w") as f:
            f.write(base_info["arch"])
        with open(f"{save_path}/base_info.json", "w") as f:
            f.write(json.dumps(base_info, indent=2, sort_keys=False, ensure_ascii=False))

        with open(f"{save_path}/mem_result.json", "w") as f:
            f.write(str(mem_result))

        with open(f"{save_path}/compute_result.json", "w") as f:
            f.write(str(compute_result))
        
        with open(f"{save_path}/strategy_config.json", "w") as f:
            f.write(str(self.strategy))

        if self.disturbance is not None:
            with open(f"{save_path}/disturbance_config.json", "w") as f:
                f.write(str(self.disturbance))

        with open(f"{save_path}/system_config.json", "w") as f:
            f.write(str(self.system))

        with open(f"{save_path}/model_config.json", "w") as f:
            f.write(str(self.model_config))

    def analysis(self, save_path=None, console_log = True):
        """Analyze the performance of the model. Return a dictionary containing the results."""
        mem_result = self.analysis_mem()
        compute_result = self.analysis_cost()
        
        if SIMU_CHECK:
            save_path = TMP_PATH
        if save_path is not None:
            print(f"Saving analysis results to {save_path}")
            os.makedirs(save_path, exist_ok=True)
            base_info = {}
            base_info["arch"] = str(self.model_chunk_dict)
            base_info["all_param"] = self.model_config.param_numel
            base_info["act_param"] = self.model_config.activated_param_numel
            with open(f"{save_path}/model_arch", "w") as f:
                f.write(base_info["arch"])
            with open(f"{save_path}/base_info.json", "w") as f:
                f.write(json.dumps(base_info, indent=2, sort_keys=False, ensure_ascii=False))

            with open(f"{save_path}/mem_result.json", "w") as f:
                f.write(str(mem_result))

            with open(f"{save_path}/compute_result.json", "w") as f:
                f.write(str(compute_result))
            
            with open(f"{save_path}/strategy_config.json", "w") as f:
                f.write(str(self.strategy))

            with open(f"{save_path}/system_config.json", "w") as f:
                f.write(str(self.system))

            with open(f"{save_path}/model_config.json", "w") as f:
                f.write(str(self.model_config))

            # disturbance_config dumped below alongside disturbance_log

            if self.seq_lens is not None:
                seq_lens_info = {
                    "values": self.seq_lens.tolist(),
                    "mean": float(self.seq_lens.mean()),
                    "std": float(self.seq_lens.std()),
                    "min": int(self.seq_lens.min()),
                    "max": int(self.seq_lens.max()),
                    "configured_mean": (
                        self.disturbance.seq_len_mean
                        if self.disturbance.seq_len_mean is not None
                        else self.strategy.seq_len
                    ),
                    "configured_std": self.disturbance.seq_len_std,
                    "seed": self.disturbance.seed,
                    "seq_len_min": self.disturbance.seq_len_min,
                    "seq_len_max": self.disturbance.seq_len_max,
                }
                with open(f"{save_path}/seq_lens.json", "w") as f:
                    json.dump(seq_lens_info, f, indent=2)

            if self.disturbance is not None:
                with open(f"{save_path}/disturbance_config.json", "w") as f:
                    f.write(str(self.disturbance))

            # Disturbance audit log: features A / B / C in one file.
            # Only emitted when at least one feature was active. The seed is
            # reported once at the top; each stream is derived via
            # SeedSequence(seed).spawn(4) at sampling time.
            disturbance = self.disturbance
            disturbance_log = {}
            if self.op_noise_mult is not None:
                mult_summary = {}
                all_vals = []
                for k, arr in self.op_noise_mult.items():
                    mult_summary[k] = arr.tolist()
                    all_vals.append(arr.ravel())
                flat = np.concatenate(all_vals)
                disturbance_log["op_duration_noise"] = {
                    "configured_std": disturbance.op_duration_std,
                    "min_factor": disturbance.op_duration_min_factor,
                    "max_factor": disturbance.op_duration_max_factor,
                    "summary": {
                        "mean": float(flat.mean()),
                        "std": float(flat.std()),
                        "min": float(flat.min()),
                        "max": float(flat.max()),
                    },
                    "multipliers_per_kind": mult_summary,
                }
            if self.op_slowdown_mask is not None:
                disturbance_log["op_slowdown"] = {
                    "configured_prob": disturbance.op_slowdown_prob,
                    "k": disturbance.op_slowdown_k,
                    "max_count": disturbance.op_slowdown_max_count,
                    "triggered_count": len(self.op_slowdown_records),
                    "triggered": self.op_slowdown_records,
                }
            if disturbance.stage_slowdown_prob > 0.0:
                disturbance_log["stage_slowdown"] = {
                    "configured_prob": disturbance.stage_slowdown_prob,
                    "k": disturbance.stage_slowdown_k,
                    "slowed_rank": self._slowed_rank,
                }
            if disturbance_log:
                disturbance_log["seed"] = disturbance.seed
                with open(f"{save_path}/disturbance_log.json", "w") as f:
                    json.dump(disturbance_log, f, indent=2)

        # print mfu/tflops/peak_mem
        peak_mem = mem_result.data["peak_mem"] if 'peak_mem' in mem_result.data else (({s:r['peak_mem'] for s, r in mem_result.data.items()}))
        peak_mem_with_reserved = mem_result.data["peak_mem_with_reserved"] if 'peak_mem_with_reserved' in mem_result.data else (({s:r['peak_mem_with_reserved'] for s, r in mem_result.data.items()}))
        if console_log:
            tp = self.strategy.tp_size
            ep = self.strategy.ep_size
            pp = self.strategy.pp_size
            act_info = f", act={compute_result.data['param_numel_info']['activations']}" if self.model_config.model_type == 'moe' else ''
            print(f"-------------SIMUMAX SUMMARY  \033[33m{self.model_config.model_name}({compute_result.data['param_numel_info']['all']}{act_info}) TP={tp},EP={ep},PP={pp}\033[0m -------------")
            print(f'- parallelism = layer{self.model_config.layer_num}.dense{self.model_config.dense_layers}.{self.strategy.parallelism}')
            print(f'- recompute = {self.strategy.recompute_status}')
            print(f"- \033[31mdtype = {'fp8' if self.strategy.fp8 else 'bf16'}, grad_reduce = {'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}\033[0m")
            print(f"- system = {self.system.sys_name}")
            print(f"- model_type = {self.model_config.model_type}")
            print(f"· \033[32mmfu = {compute_result.data['mfu_6nd_with_attn']:.2f}\033[0m")
            print(f"· \033[32mpp_utilization = {compute_result.data['pp_utilization']:.4f}\033[0m")
            print(f"· \033[32mTFLOPS = {compute_result.data['throughput per GPU (TFLOP/s/GPU)']:.2f}T (tflops={compute_result.data['flops_info']['theory_flops']}, duration={compute_result.data['duration_time_per_iter']})\033[0m")
            print(f"· \033[32mTFLOPS_PER_TOKEN = {compute_result.data['throughput per GPU per token (TFLOP/s/GPU/token)']:.2f}T, duration={compute_result.data['duration_time_per_iter']})\033[0m")
            print(f"· \033[31mpeak_alloc_mem = {peak_mem}\033[0m")
            print(f"- peak_alloc_mem_with_reserved = {peak_mem_with_reserved}")
            print(f"- TGS_per_gpu = {compute_result.data['throughput_per_accelerator']}")
            print(f'- net = {self.strategy.net} ')
            print(f"------------------------------------------")

            
        # capture graph
        if ENABLE_SIMU_GRAPH:
            self.capture(save_path)
            visualize_with_graphviz(os.path.join(save_path, 'model_graph.json'), output_path=os.path.join(save_path, 'computational_graph'))
        return {
            'model': self.model_config.model_name,
            'model_type': self.model_config.model_type,
            'params': compute_result.data['param_numel_info']['all'],
            'system': self.system.sys_name,
            'peak_mem': peak_mem,
            'peak_mem_with_reserved': peak_mem_with_reserved,
            'duration_time_per_iter': compute_result.data['duration_time_per_iter'],
            'TFLOPS': compute_result.data['throughput per GPU (TFLOP/s/GPU)'],
            'TGS_per_gpu': compute_result.data['throughput_per_accelerator'],
            'mfu': compute_result.data['mfu_6nd_with_attn'],
            'parallelism': self.strategy.parallelism,
            'recompute': self.strategy.recompute_status,
            'dtype': f"{'fp8' if self.strategy.fp8 else 'bf16'},grad_reduce in {'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}",
            'net': self.strategy.net,
        }
 