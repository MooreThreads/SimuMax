"""performance model for LLM"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import math
import json
import heapq
from copy import deepcopy
from typing import List, Union, Dict, Tuple
from sympy import divisors
from pprint import pprint
import pandas as pd
from simumax.core.base_struct import PathDebugContext, RecomputeStatus
from simumax.core.config import StrategyConfig, SystemConfig, ModelConfig, set_capture_graph_only, TMP_PATH, SIMU_CHECK, SIMU_DEBUG, ENABLE_SIMU_GRAPH
from simumax.core.base_struct import InputOutputInfo, TensorSize, Result
from simumax.core.model_struct import ActivationInfo
from simumax.core.transformer.language_model import LLMModel, PeakPoint
from simumax.core.graph import SimuONNXGraphBuilder, visualize_with_graphviz
from simumax.core.simu_runner import run_simulation
from simumax.core.trace_export import (
    export_pipeline_schedule_trace,
    normalize_schedule_records,
)
from simumax.core.utils import (
    HumanReadableSize,
    human_readable_bytes,
    convert_final_result_to_human_format,
    get_pp_stage_representative_rank,
    get_pp_p2p_comm_size,
    merge_dict,
    rm_tmp
)

FIRST_CHUNK = "first_stage_chunk"
MIDDLE_CHUNK = "middle_stage_chunk"
LAST_CHUNK = "last_stage_chunk"
STRAGGLER_BASE_FACTOR = 0.09


# Search-only caches should be keyed by every strategy field that can change a
# unit's own local single-batch behavior. Candidate assembly knobs such as
# `pp_size` / `micro_batch_num` / PP networking stay out so they can be
# recomputed per layout.
_SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS = {
    "world_size",
    "pp_size",
    "micro_batch_num",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "account_for_embedding_in_pipeline_split",
    "account_for_loss_in_pipeline_split",
    "interleaving_size",
    "microbatch_group_size_per_vp_stage",
    "pp_comm_async",
    "enable_straggler_model",
    "pp_net",
    "dp_net",
    "edp_net",
    # Derived assembly/report fields; keeping them would only fragment hits.
    "global_batch_size",
    "parallelism",
    "recompute_status",
    "shard_size",
    "net",
}


class CachedChunkProfile:
    # Keep only summary objects here. Reusing a full LLMModel across layout tasks
    # leaks the old strategy/system references and can corrupt later assembly.
    def __init__(
        self,
        *,
        layer_num: int,
        main_grad_element_size: int,
        model_info,
        compute_info,
        cost_info,
        all_gemm_cost_info,
        miss_efficiency=None,
    ) -> None:
        self.layer_num = layer_num
        self.main_grad_element_size = main_grad_element_size
        self._model_info = model_info
        self._compute_info = compute_info
        self._cost_info = cost_info
        self._all_gemm_cost_info = deepcopy(all_gemm_cost_info)
        self._miss_efficiency = deepcopy(miss_efficiency or {})

    @classmethod
    def from_model_chunk(cls, model_chunk: LLMModel, miss_efficiency=None):
        return cls(
            layer_num=model_chunk.layer_num,
            main_grad_element_size=model_chunk.main_grad_element_size,
            model_info=model_chunk.get_model_info(),
            compute_info=model_chunk.get_compute_info(),
            cost_info=model_chunk.get_cost_info(),
            all_gemm_cost_info=model_chunk.get_all_gemm_cost_info(),
            miss_efficiency=miss_efficiency,
        )

    def get_model_info(self):
        return self._model_info

    def get_compute_info(self):
        return self._compute_info

    def get_cost_info(self):
        return self._cost_info

    def get_all_gemm_cost_info(self):
        return deepcopy(self._all_gemm_cost_info)

    @property
    def miss_efficiency(self):
        return self._miss_efficiency


@dataclass(frozen=True)
class CachedLeafActivationProfile:
    full_name: str
    current_full_module_path: str
    enable_recompute: bool
    recompute_status: str
    offload_inputs: bool
    input_mem_bytes: float
    activation_mem_cache: float
    fwd_peak_mem_no_cache: float
    bwd_peak_mem_no_cache: float


class CachedUnitRuntimeProfile:
    def __init__(
        self,
        *,
        compute_info,
        cost_info,
        all_gemm_cost_info,
        leaf_profiles,
        miss_efficiency,
    ) -> None:
        self._compute_info = compute_info
        self._cost_info = cost_info
        self._all_gemm_cost_info = deepcopy(all_gemm_cost_info)
        self._leaf_profiles = tuple(leaf_profiles)
        self._miss_efficiency = deepcopy(miss_efficiency)

    @classmethod
    def from_model_chunk(cls, model_chunk: LLMModel):
        leaf_profiles = []
        for leaf in model_chunk.get_all_leaf_modules():
            act_info = leaf.get_act_info()
            leaf_profiles.append(
                CachedLeafActivationProfile(
                    full_name=leaf.full_name,
                    current_full_module_path=leaf.current_full_module_path,
                    enable_recompute=leaf.enable_recompute,
                    recompute_status=leaf.recompute_status,
                    offload_inputs=leaf.offload_inputs,
                    input_mem_bytes=leaf.all_input_element_num(),
                    activation_mem_cache=act_info.activation_mem_cache,
                    fwd_peak_mem_no_cache=act_info.fwd_peak_mem_no_cache,
                    bwd_peak_mem_no_cache=act_info.bwd_peak_mem_no_cache,
                )
            )
        return cls(
            compute_info=model_chunk.get_compute_info(),
            cost_info=model_chunk.get_cost_info(),
            all_gemm_cost_info=model_chunk.get_all_gemm_cost_info(),
            leaf_profiles=leaf_profiles,
            miss_efficiency=deepcopy(getattr(model_chunk.system, "miss_efficiency", {})),
        )

    @property
    def compute_info(self):
        return self._compute_info

    @property
    def cost_info(self):
        return self._cost_info

    @property
    def all_gemm_cost_info(self):
        return deepcopy(self._all_gemm_cost_info)

    @property
    def leaf_profiles(self):
        return self._leaf_profiles

    @property
    def miss_efficiency(self):
        return self._miss_efficiency


class CachedUnitModelProfile:
    def __init__(self, *, main_grad_element_size: int, model_info) -> None:
        self.main_grad_element_size = main_grad_element_size
        self._model_info = model_info

    @classmethod
    def from_model_chunk(cls, model_chunk: LLMModel):
        return cls(
            main_grad_element_size=model_chunk.main_grad_element_size,
            model_info=model_chunk.get_model_info(),
        )

    @property
    def model_info(self):
        return self._model_info


class _SearchLeafState:
    def __init__(
        self,
        leaf_profile: CachedLeafActivationProfile,
        *,
        layer_idx: int | None = None,
        unit_tag: str = "",
    ) -> None:
        display_prefix = unit_tag
        if layer_idx is not None:
            display_prefix = f"{display_prefix}layer_{layer_idx}"
        if display_prefix:
            display_prefix = f"[{display_prefix}] "

        self.full_name = f"{display_prefix}{leaf_profile.full_name}"
        self.current_full_module_path = f"{display_prefix}{leaf_profile.current_full_module_path}"
        self.enable_recompute = leaf_profile.enable_recompute
        self.recompute_status = leaf_profile.recompute_status
        self.offload_inputs = leaf_profile.offload_inputs
        self._input_mem_bytes = leaf_profile.input_mem_bytes
        self.is_recompute_forward_finished = False
        self.is_leaf_module = True

        act_info = ActivationInfo(
            activation_mem_cache=leaf_profile.activation_mem_cache,
            fwd_peak_mem_no_cache=leaf_profile.fwd_peak_mem_no_cache,
        )
        act_info.bwd_peak_mem_no_cache = leaf_profile.bwd_peak_mem_no_cache
        self._act_info = act_info

    def get_act_info(self):
        return self._act_info

    def all_input_element_num(self):
        return self._input_mem_bytes


_LLM_CHUNK_PROFILE_CACHE: Dict[Tuple, Tuple[CachedChunkProfile, PeakPoint]] = {}
_LLM_UNIT_RUNTIME_PROFILE_CACHE: Dict[Tuple, CachedUnitRuntimeProfile] = {}
_LLM_UNIT_MODEL_PROFILE_CACHE: Dict[Tuple, CachedUnitModelProfile] = {}


def get_effective_straggler_sample_count(
    world_size: int,
    num_per_node: int,
    dp_size: int,
    edp_size: int,
) -> int:
    """Estimate the number of independent machine-level straggler samples.

    SimuMax assumes GPUs on the same node are performance-stable, while node-to-
    node runtime can fluctuate. Under that assumption, the effective sample
    count should be limited by:

    - how many nodes are present
    - how many dense-DP replicas are active
    - how many expert-DP replicas are active

    Using min(node_count, dp_size, edp_size) keeps single-node and small-scale
    runs from exaggerating straggler inflation.
    """

    safe_num_per_node = max(1, int(num_per_node))
    node_count = max(1, math.ceil(int(world_size) / safe_num_per_node))
    return max(1, min(node_count, int(dp_size), int(edp_size)))


def estimate_straggler_increase_ratio(worker_count: int) -> float:
    """Empirical machine-level straggler inflation ratio.

    The formula preserves the expected sqrt(log n) growth of the maximum over
    many machines, while damping small-n behavior to match local simulations.
    """

    n = max(1, int(worker_count))
    if n <= 1:
        return 1.0
    n_straggler = math.log2(n)
    return 1.0 + n_straggler / (n_straggler + 1.0) * STRAGGLER_BASE_FACTOR * math.sqrt(n_straggler)

class PerfBase(ABC):
    """
    Abstract class for performance model
    """

    dtype_to_element_size = {"fp32": 4, "fp16": 2, "bf16": 2}

    def __init__(self) -> None:
        self.is_configured = False
        self.strategy = None
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

    def _set_model_config(self, model_config: ModelConfig):
        model_config.sanity_check()
        self.model_config = model_config
        # self.model_config.maybe_pad_vocab_size(self.strategy.tp_size, True)

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
        debug_points: List[str] = None,
        debug_points_last_stage=None,
    ):
        """
        Configure the performance model, including strategy, model and system config.
        And check the sanity of the configuration.
        """
        if not isinstance(strategy_config, StrategyConfig):
            strategy_config = StrategyConfig.init_from_config_file(strategy_config)
        self._set_strategy_config(strategy_config)

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
        cp_size = self.strategy.cp_size
        etp_size = self.strategy.etp_size
        edp_size = self.strategy.edp_size
        ep_size = self.strategy.ep_size
        pp_size = self.strategy.pp_size
        dp_size = self.strategy.dp_size
        num_gpu_per_nodes = self.system.num_per_node    
        # dense tp-cp-dp-pp
        # moe etp-ep-edp-pp
        # 1. analysis pp_net
        if self.strategy.pp_net == "auto" or re_analysis:
            self.strategy.pp_net = pcie_decision_helper(tp_size*dp_size*pp_size*cp_size)
        
        # 2. analysis ep_net 
        if self.strategy.ep_net == "auto" or re_analysis:
            self.strategy.ep_net = pcie_decision_helper(ep_size * etp_size)

        # 3. analysis tp_net
        if self.strategy.tp_net == "auto" or re_analysis:
            self.strategy.tp_net = pcie_decision_helper(tp_size) 
            
        # 3. analysis cp_net
        if self.strategy.cp_net == "auto" or re_analysis:
            self.strategy.cp_net = pcie_decision_helper(tp_size*cp_size) 
            
        # 4. analysis etp_net
        if self.strategy.etp_net == 'auto' or re_analysis:
            self.strategy.etp_net = pcie_decision_helper(etp_size)

        # 5. analysis dp_net
        if self.strategy.dp_net == "auto" or re_analysis:
            self.strategy.dp_net = pcie_decision_helper(tp_size*cp_size*dp_size)

        # 6. analysis edp_net
        if self.strategy.edp_net == "auto" or re_analysis:
            self.strategy.edp_net = pcie_decision_helper(etp_size * ep_size * edp_size)

    def analysis_high_link_net(self, re_analysis):
        world_size = self.strategy.world_size
        tp_size = self.strategy.tp_size
        cp_size = self.strategy.cp_size
        etp_size = self.strategy.etp_size
        edp_size = self.strategy.edp_size
        ep_size = self.strategy.ep_size
        pp_size = self.strategy.pp_size
        dp_size = self.strategy.dp_size
        num_gpu_per_nodes = self.system.num_per_node    
        # dense tp-cp-dp-pp
        # moe etp-ep-edp-pp
        
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
        
        # 3. analysis cp_net
        if self.strategy.cp_net == "auto" or re_analysis:
            condition = (tp_size * cp_size <= num_gpu_per_nodes)
            self.strategy.cp_net = "high_intra_node" if condition else "inter_node"
            
        # 4. analysis etp_net
        if self.strategy.etp_net == 'auto' or re_analysis:
            condition = etp_size <= num_gpu_per_nodes
            self.strategy.etp_net = "high_intra_node" if condition else "inter_node"

        # 5. analysis dp_net
        if self.strategy.dp_net == "auto" or re_analysis:
            condition = (tp_size * cp_size * dp_size <= num_gpu_per_nodes)
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
        self.model_config.maybe_pad_vocab_size(
            self.strategy.tp_size, log=getattr(self, "_search_verbose", True)
        )
        self.analysis_net(re_analysis = True)
        self.build()
        if capture_graph:
            self.graph = self.capture(save_path)

        self._run()
class PerfLLM(PerfBase):

    """Performance model for LLM"""

    def __init__(self) -> None:
        super().__init__()
        self.model_chunk_dict = {}
        self.vpp_chunk_dict = {}
        self.vpp_stage_chunk_names = {}
        self.path_debug_context = PathDebugContext()
        self.path_debug_context_last_stage = PathDebugContext()
        self.pp_state_peak_point = dict(
            first_stage_chunk=dict(),
            middle_stage_chunk=dict(),
            last_stage_chunk=dict()
        )
        self.enable_chunk_profile_cache = False
        self.enable_search_unit_profile_cache = False
        self._prepared_chunk_names = set()
        self._chunk_profile_model_key = None
        self._chunk_profile_system_key = None
        # os.makedirs(TMP_PATH, exist_ok=True)

    def _vp_size(self) -> int:
        return max(1, int(self.strategy.interleaving_size))

    def _vpp_chunk_name(self, stage_name: str, virtual_rank: int) -> str:
        return f"{stage_name}_v{virtual_rank}"

    def __del__(self):
        # try:
        #     import shutil
        #     if not SIMU_CHECK:
        #         if os.path.exists(TMP_PATH):
        #             shutil.rmtree(TMP_PATH)
        # except Exception as e:
        #     print(f"删除文件时出错: {e}")
        pass
    
    def get_num_layers_to_build(
        self,
        config: StrategyConfig,
        model_conf: ModelConfig,
        parallel_stage="first",
        virtual_pp_rank: int = None,
    ) -> int:
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

            # Megatron treats pp_size <= 2 as a no-middle-stage uneven-PP case.
            # In that case both first/last can be specified and there is no
            # remaining stage count to divide by.
            if pipeline_stages_left > 0:
                assert (
                    layers_to_distribute % pipeline_stages_left == 0
                ), f"With uneven pipelineing the left over layers must be divisible by left over stages, layers_to_distribute={layers_to_distribute}, pipeline_stages_left={pipeline_stages_left}"
                num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
            else:
                num_layers_per_pipeline_rank = 0

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

        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

        # Preserve legacy behavior when querying whole physical stage.
        if virtual_pp_rank is None:
            if parallel_stage == "first" and config.account_for_embedding_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the first pipeline stage"

            if parallel_stage == "last" and config.account_for_loss_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the last pipeline stage"
            if SIMU_DEBUG:
                print(f"Building {num_layers_to_build} layers for {parallel_stage} stage")
            return num_layers_to_build

        # Interleaved case: split a physical stage into virtual pipeline chunks.
        vp_size = max(1, int(config.interleaving_size))
        assert 0 <= virtual_pp_rank < vp_size, (
            f"virtual_pp_rank should be in [0, {vp_size - 1}], got {virtual_pp_rank}"
        )
        assert (
            num_layers_per_pipeline_rank % vp_size == 0
        ), (
            f"num_layers_per_pipeline_rank={num_layers_per_pipeline_rank} must be divisible "
            f"by interleaving_size={vp_size}"
        )
        num_layers_to_build = num_layers_per_pipeline_rank // vp_size

        # Keep Megatron-like placement for embedding/loss partition adjustments:
        # embedding subtraction at first virtual chunk; loss subtraction at last virtual chunk.
        if (
            parallel_stage == "first"
            and config.account_for_embedding_in_pipeline_split
            and virtual_pp_rank == 0
        ):
            num_layers_to_build -= 1
        if (
            parallel_stage == "last"
            and config.account_for_loss_in_pipeline_split
            and virtual_pp_rank == vp_size - 1
        ):
            num_layers_to_build -= 1
        assert num_layers_to_build >= 0, (
            f"Not enough layers in virtual pipeline stage: stage={parallel_stage}, "
            f"virtual_pp_rank={virtual_pp_rank}"
        )
        # if parallel_stage == "middle":
        #     num_layers_to_build += sum([config.account_for_embedding_in_pipeline_split, config.account_for_loss_in_pipeline_split])
        if SIMU_DEBUG:
            print(
                f"Building {num_layers_to_build} layers for {parallel_stage} stage "
                f"(virtual_pp_rank={virtual_pp_rank})"
            )
        return num_layers_to_build

    def build(self):
        """
        build first stage model chunk and last stage model chunk
        """
        self.strategy.sanity_check()
        self.model_chunk_dict:Dict[str, LLMModel] = {}
        self.vpp_chunk_dict = {}
        self._prepared_chunk_names = set()
        self.vpp_stage_chunk_names = {
            FIRST_CHUNK: [],
            MIDDLE_CHUNK: [],
            LAST_CHUNK: [],
        }

        def _register_chunk(
            *,
            chunk_name: str,
            layer_num: int,
            dense_layers: int,
            preprocess: bool,
            postprocess: bool,
            specific_name: str,
        ):
            if self._chunk_profile_cache_enabled():
                chunk_profile, peak_point = self._get_cached_chunk_profile(
                    layer_num=layer_num,
                    dense_layers=dense_layers,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    specific_name=specific_name,
                )
                self.model_chunk_dict[chunk_name] = chunk_profile
                self.pp_state_peak_point[chunk_name] = peak_point
                self.system.miss_efficiency = self._merge_missing_efficiency(
                    deepcopy(getattr(self.system, "miss_efficiency", {})),
                    deepcopy(chunk_profile.miss_efficiency),
                )
                self._prepared_chunk_names.add(chunk_name)
                return

            self.model_chunk_dict[chunk_name] = LLMModel(
                layer_num=layer_num,
                preprocess=preprocess,
                postprocess=postprocess,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
                dense_layers=dense_layers,
                specific_name=specific_name,
            )

        # Build First Stage Model Chunk
        # Only consider the even divide case fow now
        # layer_num = self.model_config.layer_num // self.strategy.pp_size
        remian_dense_layers=self.model_config.dense_layers
        dense_layers_i = max(0, remian_dense_layers)
        remian_dense_layers -= dense_layers_i

        layer_num_first = self.get_num_layers_to_build(self.strategy, self.model_config, "first")
        if self.strategy.pp_size > 1:
            _register_chunk(
                chunk_name=FIRST_CHUNK,
                layer_num=layer_num_first,
                preprocess=True,
                postprocess=False,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_first_pp_stage"
            )
        else:
            _register_chunk(
                chunk_name=FIRST_CHUNK,
                layer_num=layer_num_first,
                preprocess=True,
                postprocess=True,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_first_pp_stage",
            )
        first_stage_dense_layers = dense_layers_i
        if self.strategy.pp_size > 2:
            layer_num_middle = self.get_num_layers_to_build(self.strategy, self.model_config, "middle")
            dense_layers_i = max(0, remian_dense_layers)
            remian_dense_layers -= dense_layers_i*(self.strategy.pp_size-2)
            _register_chunk(
                chunk_name=MIDDLE_CHUNK,
                layer_num=layer_num_middle,
                preprocess=False,
                postprocess=False,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_middle_pp_stage"
            )
            middle_stage_dense_layers = dense_layers_i
        else:
            middle_stage_dense_layers = 0

        # # Build Last Stage Model Chunk
        if self.strategy.pp_size > 1:
            layer_num_last = self.get_num_layers_to_build(self.strategy, self.model_config, "last")
            dense_layers_i = max(0, remian_dense_layers)
            _register_chunk(
                chunk_name=LAST_CHUNK,
                layer_num=layer_num_last,
                preprocess=False,
                postprocess=True,
                dense_layers=dense_layers_i,
                specific_name="GPTModel_last_pp_stage"
            )
            last_stage_dense_layers = dense_layers_i
        else:
            last_stage_dense_layers = 0

        # Build virtual chunks for interleaving-aware timing (kept alongside physical chunks).
        vp_size = self._vp_size()
        if vp_size > 1:
            def _build_stage_vpp(stage_key: str, stage_dense_layers: int, preprocess: bool, postprocess: bool):
                if stage_key not in self.model_chunk_dict:
                    return
                stage_name = "first" if stage_key == FIRST_CHUNK else ("middle" if stage_key == MIDDLE_CHUNK else "last")
                for virtual_rank in range(vp_size):
                    layer_num_virtual = self.get_num_layers_to_build(
                        self.strategy,
                        self.model_config,
                        stage_name,
                        virtual_pp_rank=virtual_rank,
                    )
                    vpp_chunk_name = self._vpp_chunk_name(stage_key, virtual_rank)
                    # Keep legacy dense-layer semantics: attach stage dense layers to the first virtual chunk.
                    dense_layers_virtual = stage_dense_layers if virtual_rank == 0 else 0
                    vpp_chunk = LLMModel(
                        layer_num=layer_num_virtual,
                        preprocess=(preprocess and virtual_rank == 0),
                        postprocess=(postprocess and virtual_rank == vp_size - 1),
                        model_config=self.model_config,
                        strategy=self.strategy,
                        system=self.system,
                        dense_layers=dense_layers_virtual,
                        specific_name=f"{vpp_chunk_name}_model",
                    )
                    self.vpp_chunk_dict[vpp_chunk_name] = vpp_chunk
                    self.vpp_stage_chunk_names[stage_key].append(vpp_chunk_name)

            _build_stage_vpp(
                FIRST_CHUNK,
                first_stage_dense_layers,
                preprocess=True,
                postprocess=(self.strategy.pp_size == 1),
            )
            if self.strategy.pp_size > 2:
                _build_stage_vpp(
                    MIDDLE_CHUNK,
                    middle_stage_dense_layers,
                    preprocess=False,
                    postprocess=False,
                )
            if self.strategy.pp_size > 1:
                _build_stage_vpp(
                    LAST_CHUNK,
                    last_stage_dense_layers,
                    preprocess=False,
                    postprocess=True,
                )

    def _chunk_profile_cache_enabled(self) -> bool:
        return self.enable_chunk_profile_cache and self._vp_size() == 1

    def _search_unit_profile_cache_enabled(self) -> bool:
        return self.enable_search_unit_profile_cache and self._vp_size() == 1

    def _search_unit_fast_path_enabled_for_chunk(self, *, layer_num: int) -> bool:
        if not self._search_unit_profile_cache_enabled():
            return False
        if (
            layer_num > 0
            and self.strategy.enable_recompute
            and self.strategy.recompute_granularity == "full_block"
            and self.strategy.recompute_layer_num > 0
        ):
            return False
        return True

    def _search_cache_strategy_projection(self, strategy=None):
        strategy_dict = deepcopy((strategy or self.strategy).to_dict())
        for field in _SEARCH_CACHE_ASSEMBLY_ONLY_STRATEGY_FIELDS:
            strategy_dict.pop(field, None)
        return json.dumps(strategy_dict, sort_keys=True)

    def _search_unit_local_strategy(self, layer_idx: int | None):
        if layer_idx is None:
            return self.strategy

        local_strategy = deepcopy(self.strategy)
        local_recompute_enabled = (
            self.strategy.enable_recompute
            and self.strategy.recompute_granularity is not None
            and layer_idx < self.strategy.recompute_layer_num
        )
        local_strategy.recompute_layer_num = 1 if local_recompute_enabled else 0
        return local_strategy

    def _chunk_profile_cache_key(
        self,
        *,
        layer_num: int,
        dense_layers: int,
        preprocess: bool,
        postprocess: bool,
    ):
        local_key = (
            layer_num,
            dense_layers,
            preprocess,
            postprocess,
        )
        return (
            self._search_cache_strategy_projection(),
            self._chunk_profile_model_key,
            self._chunk_profile_system_key,
            local_key,
        )

    def _search_unit_runtime_cache_key(self, unit_kind: str, layer_idx: int | None = None):
        local_strategy = self._search_unit_local_strategy(layer_idx)
        return (
            self._search_cache_strategy_projection(local_strategy),
            self._chunk_profile_model_key,
            self._chunk_profile_system_key,
            unit_kind,
        )

    def _search_unit_model_cache_key(self, unit_kind: str, layer_idx: int | None = None):
        local_strategy = self._search_unit_local_strategy(layer_idx)
        return (
            self._search_cache_strategy_projection(local_strategy),
            self._chunk_profile_model_key,
            unit_kind,
        )

    def _build_chunk_input_info(self, preprocess: bool):
        if preprocess:
            return InputOutputInfo(
                tensors=[
                    TensorSize(
                        shape=(
                            self.strategy.micro_batch_size,
                            self.strategy.seq_len // self.strategy.cp_size,
                        )
                    )
                ]
            )
        seq_len = (
            self.strategy.seq_len // self.strategy.tp_size
            if self.strategy.enable_sequence_parallel
            else self.strategy.seq_len
        )
        return InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(
                        self.strategy.micro_batch_size,
                        seq_len // self.strategy.cp_size,
                        self.model_config.hidden_size,
                    )
                )
            ]
        )

    def _empty_gemm_cost_info(self):
        return {
            "Module": [],
            "type": [],
            "B": [],
            "M": [],
            "K": [],
            "N": [],
            "layout": [],
            "accumulate": [],
            "out_dtype": [],
            "compute_cost": [],
            "memory_cost": [],
            "cost": [],
            "bound": [],
        }

    def _merge_gemm_cost_info(self, left: dict, right: dict):
        if not left:
            return deepcopy(right)
        if not right:
            return left
        merged = deepcopy(left)
        for key, values in right.items():
            merged.setdefault(key, [])
            merged[key].extend(values)
        return merged

    def _merge_missing_efficiency(self, current: dict, incoming: dict):
        if not incoming:
            return current
        for key, value in incoming.items():
            if isinstance(value, dict):
                current[key] = self._merge_missing_efficiency(
                    deepcopy(current.get(key, {})),
                    value,
                )
            else:
                current[key] = value
        return current

    def _diff_missing_efficiency(self, current: dict, previous: dict):
        delta = {}
        for key, value in current.items():
            prev_value = previous.get(key)
            if isinstance(value, dict):
                if not isinstance(prev_value, dict):
                    delta[key] = deepcopy(value)
                    continue
                nested_delta = self._diff_missing_efficiency(value, prev_value)
                if nested_delta:
                    delta[key] = nested_delta
            elif key not in previous or prev_value != value:
                delta[key] = value
        return delta

    def _build_search_unit_model_chunk(self, unit_kind: str, strategy=None):
        strategy = strategy or self.strategy
        if unit_kind == "preprocess":
            return LLMModel(
                layer_num=0,
                preprocess=True,
                postprocess=False,
                model_config=self.model_config,
                strategy=strategy,
                system=self.system,
                dense_layers=0,
                specific_name="search_unit_preprocess",
            )
        if unit_kind == "postprocess":
            return LLMModel(
                layer_num=0,
                preprocess=False,
                postprocess=True,
                model_config=self.model_config,
                strategy=strategy,
                system=self.system,
                dense_layers=0,
                specific_name="search_unit_postprocess",
            )
        if unit_kind == "dense_block":
            return LLMModel(
                layer_num=1,
                preprocess=False,
                postprocess=False,
                model_config=self.model_config,
                strategy=strategy,
                system=self.system,
                dense_layers=1,
                specific_name="search_unit_dense_block",
            )
        if unit_kind == "moe_block":
            dense_layers = 1 if self.model_config.expert_num == 1 else 0
            return LLMModel(
                layer_num=1,
                preprocess=False,
                postprocess=False,
                model_config=self.model_config,
                strategy=strategy,
                system=self.system,
                dense_layers=dense_layers,
                specific_name="search_unit_moe_block",
            )
        raise ValueError(f"unsupported search unit kind: {unit_kind}")

    def _build_search_unit_profiles(self, unit_kind: str, layer_idx: int | None = None):
        local_strategy = self._search_unit_local_strategy(layer_idx)
        model_chunk = self._build_search_unit_model_chunk(unit_kind, strategy=local_strategy)
        input_info = self._build_chunk_input_info(unit_kind == "preprocess")
        _ = model_chunk(
            input_info,
            PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                target_point=[],
                path_list=[],
            ),
        )
        _ = model_chunk.compute_activations()
        return (
            CachedUnitRuntimeProfile.from_model_chunk(model_chunk),
            CachedUnitModelProfile.from_model_chunk(model_chunk),
        )

    def _get_cached_search_unit_profile(self, unit_kind: str, layer_idx: int | None = None):
        runtime_key = self._search_unit_runtime_cache_key(unit_kind, layer_idx)
        model_key = self._search_unit_model_cache_key(unit_kind, layer_idx)
        runtime_profile = _LLM_UNIT_RUNTIME_PROFILE_CACHE.get(runtime_key)
        model_profile = _LLM_UNIT_MODEL_PROFILE_CACHE.get(model_key)
        if runtime_profile is None or model_profile is None:
            runtime_profile, model_profile = self._build_search_unit_profiles(unit_kind, layer_idx)
            _LLM_UNIT_RUNTIME_PROFILE_CACHE[runtime_key] = runtime_profile
            _LLM_UNIT_MODEL_PROFILE_CACHE[model_key] = model_profile
        return runtime_profile, model_profile

    def _instantiate_search_leaf_states(
        self,
        *,
        unit_kind: str,
        leaf_profiles: Tuple[CachedLeafActivationProfile, ...],
        layer_idx: int | None,
    ):
        return [
            _SearchLeafState(
                leaf_profile,
                layer_idx=layer_idx,
                unit_tag=unit_kind,
            )
            for leaf_profile in leaf_profiles
        ]

    def _normalize_search_leaf_recompute_states(self, leaf_states):
        prev_recompute_leaf = None
        in_recompute_segment = False
        for leaf in leaf_states:
            if leaf.enable_recompute:
                leaf.recompute_status = RecomputeStatus.MIDDLE
                if not in_recompute_segment:
                    leaf.recompute_status = RecomputeStatus.FIRST
                    in_recompute_segment = True
                prev_recompute_leaf = leaf
                continue

            if in_recompute_segment and prev_recompute_leaf is not None:
                prev_recompute_leaf.recompute_status = RecomputeStatus.LAST
            in_recompute_segment = False
            prev_recompute_leaf = None
            leaf.recompute_status = RecomputeStatus.NO_RECOMPUTE

        if in_recompute_segment and prev_recompute_leaf is not None:
            prev_recompute_leaf.recompute_status = RecomputeStatus.LAST

    def _compute_peak_point_from_search_leaf_states(self, leaf_states):
        def _comp_fwd(enable_recompute, nodes, global_cache_mem, peak_point, stage="forward"):
            for node in nodes:
                act_info = node.get_act_info()
                cur_peak_mem = global_cache_mem + act_info.fwd_peak_mem_no_cache
                peak_point.update_peak(
                    f"{node.full_name}: {node.current_full_module_path}",
                    cur_peak_mem,
                    stage,
                )
                if enable_recompute and node.enable_recompute:
                    if stage == "recompute_forward" and node.recompute_status != "first":
                        act_info.cache_for_bwd_mem = act_info.activation_mem_cache
                        global_cache_mem += act_info.cache_for_bwd_mem
                    elif stage == "forward" and node.recompute_status == "first":
                        act_info.cache_for_bwd_mem = (
                            node.all_input_element_num() if not node.offload_inputs else 0
                        )
                        global_cache_mem += act_info.cache_for_bwd_mem
                else:
                    act_info.cache_for_bwd_mem = act_info.activation_mem_cache
                    global_cache_mem += act_info.cache_for_bwd_mem

            if nodes:
                last_node = nodes[-1]
                peak_point.update_peak(
                    f"{last_node.full_name}: {last_node.current_full_module_path}",
                    global_cache_mem,
                    stage,
                )
            if stage == "forward":
                peak_point.set_forward_mem_cache(global_cache_mem)
            return global_cache_mem

        def _comp_bwd_only(nodes, global_cache_mem, peak_point, stage="backward"):
            for node in nodes[::-1]:
                act_info = node.get_act_info()
                cur_peak_mem = global_cache_mem + act_info.bwd_peak_mem_no_cache
                peak_point.update_peak(
                    f"{node.full_name}: {node.current_full_module_path}",
                    cur_peak_mem,
                    stage,
                )
                global_cache_mem -= act_info.cache_for_bwd_mem
                act_info.cache_for_bwd_mem = 0
            return global_cache_mem

        peak_point = PeakPoint()
        enable_recompute = self.strategy.enable_recompute
        global_cache_mem = _comp_fwd(enable_recompute, leaf_states, 0, peak_point, "forward")

        wait_recompute_nodes = []
        i = len(leaf_states) - 1
        prepare_recompute_ready = False
        while i >= 0:
            node = leaf_states[i]
            if (
                enable_recompute
                and node.enable_recompute
                and not node.is_recompute_forward_finished
                and not prepare_recompute_ready
            ):
                wait_recompute_nodes.append(node)
                if node.recompute_status == "first":
                    prepare_recompute_ready = True
                i -= 1
            elif len(wait_recompute_nodes) > 0:
                wait_recompute_nodes = wait_recompute_nodes[::-1]
                global_cache_mem = _comp_fwd(
                    enable_recompute,
                    wait_recompute_nodes,
                    global_cache_mem,
                    peak_point,
                    "recompute_forward",
                )
                global_cache_mem = _comp_bwd_only(
                    wait_recompute_nodes,
                    global_cache_mem,
                    peak_point,
                    "recompute_backward",
                )
                for wait_node in wait_recompute_nodes:
                    wait_node.is_recompute_forward_finished = True
                wait_recompute_nodes = []
                prepare_recompute_ready = False
            else:
                act_info = node.get_act_info()
                cur_peak_mem = global_cache_mem + act_info.bwd_peak_mem_no_cache
                peak_point.update_peak(
                    f"{node.full_name}: {node.current_full_module_path}",
                    cur_peak_mem,
                    "backward",
                )
                global_cache_mem -= act_info.cache_for_bwd_mem
                act_info.cache_for_bwd_mem = 0
                i -= 1

        if len(wait_recompute_nodes) > 0:
            wait_recompute_nodes = wait_recompute_nodes[::-1]
            global_cache_mem = _comp_fwd(
                enable_recompute,
                wait_recompute_nodes,
                global_cache_mem,
                peak_point,
                "recompute_forward",
            )
            global_cache_mem = _comp_bwd_only(
                wait_recompute_nodes,
                global_cache_mem,
                peak_point,
                "recompute_backward",
            )

        assert global_cache_mem == 0, (
            f"search unit-profile activation assembly should end with zero cache, got {global_cache_mem}"
        )
        return peak_point

    def _prepare_chunk_profile_from_search_units(
        self,
        *,
        layer_num: int,
        dense_layers: int,
        preprocess: bool,
        postprocess: bool,
    ):
        model_info = None
        compute_info = None
        cost_info = None
        all_gemm_cost_info = self._empty_gemm_cost_info()
        leaf_states = []
        main_grad_element_size = None

        def _accumulate_unit(unit_kind: str, layer_idx: int | None = None):
            nonlocal model_info, compute_info, cost_info, all_gemm_cost_info, leaf_states, main_grad_element_size
            runtime_profile, model_profile = self._get_cached_search_unit_profile(unit_kind)
            model_info = (
                deepcopy(model_profile.model_info)
                if model_info is None
                else model_info + model_profile.model_info
            )
            compute_info = (
                deepcopy(runtime_profile.compute_info)
                if compute_info is None
                else compute_info + runtime_profile.compute_info
            )
            cost_info = (
                deepcopy(runtime_profile.cost_info)
                if cost_info is None
                else cost_info + runtime_profile.cost_info
            )
            all_gemm_cost_info = self._merge_gemm_cost_info(
                all_gemm_cost_info,
                runtime_profile.all_gemm_cost_info,
            )
            if main_grad_element_size is None:
                main_grad_element_size = model_profile.main_grad_element_size
            self.system.miss_efficiency = self._merge_missing_efficiency(
                deepcopy(getattr(self.system, "miss_efficiency", {})),
                deepcopy(runtime_profile.miss_efficiency),
            )
            leaf_states.extend(
                self._instantiate_search_leaf_states(
                    unit_kind=unit_kind,
                    leaf_profiles=runtime_profile.leaf_profiles,
                    layer_idx=layer_idx,
                )
            )

        if preprocess:
            _accumulate_unit("preprocess")
        for layer_idx in range(layer_num):
            unit_kind = "dense_block" if layer_idx < dense_layers else "moe_block"
            _accumulate_unit(unit_kind, layer_idx=layer_idx)
        if postprocess:
            _accumulate_unit("postprocess")

        self._normalize_search_leaf_recompute_states(leaf_states)
        peak_point = self._compute_peak_point_from_search_leaf_states(leaf_states)
        return (
            CachedChunkProfile(
                layer_num=layer_num,
                main_grad_element_size=main_grad_element_size or self.dtype_to_element_size[self.strategy.dtype],
                model_info=model_info,
                compute_info=compute_info,
                cost_info=cost_info,
                all_gemm_cost_info=all_gemm_cost_info,
            ),
            peak_point,
        )

    def _prepare_chunk_profile(
        self,
        *,
        layer_num: int,
        dense_layers: int,
        preprocess: bool,
        postprocess: bool,
        specific_name: str,
    ):
        previous_missing_efficiency = deepcopy(getattr(self.system, "miss_efficiency", {}))
        if self._search_unit_fast_path_enabled_for_chunk(layer_num=layer_num):
            chunk_profile, peak_point = self._prepare_chunk_profile_from_search_units(
                layer_num=layer_num,
                dense_layers=dense_layers,
                preprocess=preprocess,
                postprocess=postprocess,
            )
            chunk_profile._miss_efficiency = self._diff_missing_efficiency(
                getattr(self.system, "miss_efficiency", {}),
                previous_missing_efficiency,
            )
            return chunk_profile, peak_point

        model_chunk = LLMModel(
            layer_num=layer_num,
            preprocess=preprocess,
            postprocess=postprocess,
            model_config=self.model_config,
            strategy=self.strategy,
            system=self.system,
            dense_layers=dense_layers,
            specific_name=specific_name,
        )
        input_info = self._build_chunk_input_info(preprocess)
        _ = model_chunk(
            input_info,
            PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                target_point=[],
                path_list=[],
            ),
        )
        peak_point = model_chunk.compute_activations()
        miss_efficiency = self._diff_missing_efficiency(
            getattr(self.system, "miss_efficiency", {}),
            previous_missing_efficiency,
        )
        return CachedChunkProfile.from_model_chunk(model_chunk, miss_efficiency=miss_efficiency), peak_point

    def _get_cached_chunk_profile(
        self,
        *,
        layer_num: int,
        dense_layers: int,
        preprocess: bool,
        postprocess: bool,
        specific_name: str,
    ):
        cache_key = self._chunk_profile_cache_key(
            layer_num=layer_num,
            dense_layers=dense_layers,
            preprocess=preprocess,
            postprocess=postprocess,
        )
        cached = _LLM_CHUNK_PROFILE_CACHE.get(cache_key)
        if cached is None:
            cached = self._prepare_chunk_profile(
                layer_num=layer_num,
                dense_layers=dense_layers,
                preprocess=preprocess,
                postprocess=postprocess,
                specific_name=specific_name,
            )
            _LLM_CHUNK_PROFILE_CACHE[cache_key] = cached
        return cached

    def _cross_sanity_check(self) -> bool:
        # assert (
        #     self.model_config.layer_num % self.strategy.pp_size == 0
        # ), "layer num should be divisible by pp_size"

        assert self.debug_points is None or isinstance(
            self.debug_points, list
        ), "debug_points should be a list"
        if self.strategy.megatron_recompute:
            modules = self.strategy.megatron_recompute_module_set
            if "mla_up_proj" in modules:
                assert getattr(self.model_config, "attention_type", None) == "mla", (
                    "megatron_recompute mla_up_proj is only supported with MLA attention"
                )
            if "moe_act" in modules:
                assert self.model_config.expert_num > 1, (
                    "megatron_recompute moe_act is only supported on MoE models"
                )
                assert self.model_config.group_linear_mode == "parallel", (
                    "megatron_recompute moe_act is only supported with grouped-gemm MoE"
                )
            if self.strategy.fp8:
                unsupported_fp8_modules = modules & {"layernorm", "moe_act"}
                assert not unsupported_fp8_modules, (
                    "megatron_recompute layernorm and moe_act cannot work with fp8"
                )
        assert (
            self.model_config.head_num % self.strategy.tp_size == 0
        ), f"head_num {self.model_config.head_num} should be divisible by tp_size {self.strategy.tp_size}"
        if self.model_config.kv_head_num is not None:
            assert (
                self.model_config.kv_head_num % self.strategy.tp_size == 0
            ), f"kv_head_num {self.model_config.kv_head_num} should be divisible by tp_size {self.strategy.tp_size}"
        assert (
            self.model_config.expert_num % self.strategy.ep_size == 0
        ), f"expert num {self.model_config.expert_num} should be divisible by ep_size {self.strategy.ep_size}"  # pylint: disable=line-too-long
        if self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "a2a":
            assert self.model_config.head_num % self.strategy.cp_size == 0, (
                f"head_num {self.model_config.head_num} must be divisible by cp_size {self.strategy.cp_size} when cp_comm_type='a2a'"
            )
            if self.model_config.kv_head_num is not None:
                assert self.model_config.kv_head_num % self.strategy.cp_size == 0, (
                    f"kv_head_num {self.model_config.kv_head_num} must be divisible by cp_size {self.strategy.cp_size} when cp_comm_type='a2a'"
                )

    def configure(
        self,
        strategy_config: Union[StrategyConfig, str] = None,
        model_config: Union[ModelConfig, str] = None,
        system_config: Union[SystemConfig, str] = None,
        debug_points: List[str] = None,
        debug_points_last_stage=None,
    ):
        super().configure(
            strategy_config=strategy_config,
            model_config=model_config,
            system_config=system_config,
            debug_points=debug_points,
            debug_points_last_stage=debug_points_last_stage,
        )
        self._chunk_profile_model_key = json.dumps(
            self.model_config.to_dict(), sort_keys=True
        )
        self._chunk_profile_system_key = json.dumps(
            self.system.to_dict(), sort_keys=True
        )

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
            mul_before_reduce_time = self.system.compute_mem_access_time('default', 2 * model_info.all_grad_bytes) if self.strategy.dp_size * self.strategy.cp_size > 1 else 0# read grads and write grads

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
            chunk_weight_accessed_time = 3 * state_weight_bytes # why 3倍?
            optim_time = self.system.compute_mem_access_time(chunk_weight_accessed_time)
            optim_exposed_time = adam_time  # no overlap for now
            result["optim_time"] = adam_time
            result["optim_exposed_time"] = optim_exposed_time 
            return result

    def _compute_dp_time(self, model_name):
        # TODO: support overlap
        use_megatron = True

        def grad_bytes_to_param_bytes(grad_bytes):
            grad_numel = grad_bytes / self.model_chunk_dict[model_name].main_grad_element_size
            return grad_numel * self.dtype_to_element_size[self.strategy.dtype]
    
        def compute_dp_helper(
            rs_comm_size,
            gather_comm_size,
            dp_net,
            group_size,
            dp_group,
        ):
            result = {"dp_comm_time": 0, "dp_comm_exposed_time": 0}
            dp_comm_time = 0
            # `group_size` is the actual collective size for this communication
            # family. Keep bucket sizing local to that family instead of mixing
            # in global `cp_size` again.
            bucket_size = max(40000000, 1000000 * group_size) * 4

            num_reduce_bucket = (rs_comm_size - 1) // bucket_size + 1  
            num_gather_bucket = (gather_comm_size - 1) // bucket_size + 1
            if self.model_config.model_type == "moe" and use_megatron:
                num_gather_bucket *= 2 
            details = {}
            if self.strategy.zero_state >= 1:
                reduce_scatter_time = num_reduce_bucket * self.system.compute_net_op_time(
                    "reduce_scatter",
                    bucket_size,
                    comm_num=group_size,
                    net=dp_net,
                    comm_stage=dp_group, 
                    strategy=self.strategy
                )
                all_gather_time = num_gather_bucket * self.system.compute_net_op_time(
                    "all_gather", 
                    bucket_size, 
                    comm_num=group_size, 
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
                    comm_num=group_size, 
                    net=dp_net,
                    comm_stage=dp_group,
                    strategy=self.strategy
                )

            dp_comm_exposed_time = dp_comm_time  # no overlap for now
            result['dp_comm_rs_size'] = rs_comm_size if group_size > 1 else 0
            result['dp_comm_ag_size'] = gather_comm_size if group_size > 1 else 0
            result['dp_comm_num_gather'] = 2 if self.model_config.model_type == "moe" and use_megatron else 1
            result["dp_comm_time"] = dp_comm_time
            result["dp_comm_exposed_time"] = dp_comm_exposed_time
            if details:
                result['details'] = details
            return result
        
        model_info = self.model_chunk_dict[model_name].get_model_info()

        # dense
        rs_comm_size = model_info.dense_grad_bytes
        gather_comm_size = grad_bytes_to_param_bytes(model_info.dense_grad_bytes)
        
        # moe
        moe_rs_comm_size = model_info.moe_grad_bytes
        moe_gather_comm_size = grad_bytes_to_param_bytes(model_info.moe_grad_bytes)

        dense_dp_result = compute_dp_helper(rs_comm_size, gather_comm_size, self.strategy.dp_net, self.strategy.dp_size*self.strategy.cp_size, dp_group="dp_cp")
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
        te_dummy_wgrad_model_mem = dict(
            all_mem=model_info.te_dummy_wgrad_bytes,
            detail=dict(
                dummy_wgrad_bytes=model_info.te_dummy_wgrad_bytes,
                shape_count=len(model_info.te_dummy_wgrad_shapes),
                shapes=sorted(model_info.te_dummy_wgrad_shapes),
            ),
        )
        result["model_mem"] = (
            dense_model_mem['all_mem']
            + moe_model_mem['all_mem']
            + te_dummy_wgrad_model_mem["all_mem"]
        )
        result["model_mem_detail"] = dict(
            dense = dense_model_mem,
            moe = moe_model_mem,
            te_dummy_wgrad=te_dummy_wgrad_model_mem,
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

    def _stage_key_for_pp_rank(self, pp_rank: int):
        if pp_rank == 0:
            return FIRST_CHUNK
        if pp_rank == self.strategy.pp_size - 1:
            return LAST_CHUNK
        return MIDDLE_CHUNK

    def _vpp_stage_result_key(self, pp_rank: int):
        if self.strategy.pp_size <= 1 or pp_rank == 0:
            return "first_stage"
        if pp_rank == self.strategy.pp_size - 1:
            return "last_stage"
        return f"pp_stage_{pp_rank}"

    def _get_vpp_stage_chunk_names(self, pp_rank: int):
        stage_key = self._stage_key_for_pp_rank(pp_rank)
        return stage_key, list(self.vpp_stage_chunk_names.get(stage_key, []))

    def _get_peak_point_for_model(self, model_name: str):
        peak_point = self.pp_state_peak_point.get(model_name)
        if peak_point is not None:
            return peak_point
        model_obj = self.model_chunk_dict.get(model_name)
        if model_obj is None:
            model_obj = self.vpp_chunk_dict.get(model_name)
        if model_obj is None:
            raise KeyError(f"Unknown model chunk: {model_name}")
        peak_point = model_obj.compute_activations()
        self.pp_state_peak_point[model_name] = peak_point
        return peak_point

    def _sum_model_mem_for_chunks(self, chunk_names):
        dense_detail = dict(weight_bytes=0.0, grad_bytes=0.0, state_bytes=0.0)
        moe_detail = dict(weight_bytes=0.0, grad_bytes=0.0, state_bytes=0.0)
        te_dummy_wgrad_shapes = set()
        for chunk_name in chunk_names:
            model_info = self.vpp_chunk_dict[chunk_name].get_model_info()
            dense_detail["weight_bytes"] += model_info.dense_weight_bytes
            dense_detail["grad_bytes"] += model_info.dense_grad_bytes
            dense_detail["state_bytes"] += model_info.dense_state_bytes
            moe_detail["weight_bytes"] += model_info.moe_weight_bytes
            moe_detail["grad_bytes"] += model_info.moe_grad_bytes
            moe_detail["state_bytes"] += model_info.moe_state_bytes
            te_dummy_wgrad_shapes |= model_info.te_dummy_wgrad_shapes

        dense_model_mem = dict(
            all_mem=sum(dense_detail.values()),
            detail=dense_detail,
        )
        moe_model_mem = dict(
            all_mem=sum(moe_detail.values()),
            detail=moe_detail,
        )
        te_dummy_wgrad_bytes = sum(rows * cols * elem_size for rows, cols, elem_size in te_dummy_wgrad_shapes)
        te_dummy_wgrad_model_mem = dict(
            all_mem=te_dummy_wgrad_bytes,
            detail=dict(
                dummy_wgrad_bytes=te_dummy_wgrad_bytes,
                shape_count=len(te_dummy_wgrad_shapes),
                shapes=sorted(te_dummy_wgrad_shapes),
            ),
        )
        return dense_model_mem, moe_model_mem, te_dummy_wgrad_model_mem

    def _build_sync_vpp_local_phase_sequence(self, pp_rank: int):
        vp_size = self._vp_size()
        pp_size = self.strategy.pp_size
        stage_key, chunk_names = self._get_vpp_stage_chunk_names(pp_rank)
        if vp_size <= 1 or not chunk_names:
            return stage_key, []

        total_virtual_microbatches = self.strategy.micro_batch_num * vp_size
        group_size_per_vp_stage = getattr(
            self.strategy, "microbatch_group_size_per_vp_stage", None
        )
        if group_size_per_vp_stage is None:
            group_size_per_vp_stage = pp_size

        num_warmup_microbatches = (pp_size - pp_rank - 1) * 2 + (
            vp_size - 1
        ) * group_size_per_vp_stage
        num_warmup_microbatches = min(num_warmup_microbatches, total_virtual_microbatches)
        num_microbatches_remaining = total_virtual_microbatches - num_warmup_microbatches

        schedule_table = []
        for min_mb in range(0, self.strategy.micro_batch_num, group_size_per_vp_stage):
            max_mb = min(self.strategy.micro_batch_num, min_mb + group_size_per_vp_stage)
            for chunk_idx in range(vp_size):
                for real_mb in range(min_mb, max_mb):
                    schedule_table.append((real_mb, chunk_idx))

        def _fwd_ref(virtual_k: int):
            real_mb, chunk_idx = schedule_table[virtual_k]
            return {
                "phase": "fwd",
                "microbatch": real_mb,
                "chunk_idx": chunk_idx,
                "model_name": chunk_names[chunk_idx],
            }

        def _bwd_ref(virtual_k: int):
            real_mb, fwd_chunk_idx = schedule_table[virtual_k]
            chunk_idx = vp_size - 1 - fwd_chunk_idx
            return {
                "phase": "bwd",
                "microbatch": real_mb,
                "chunk_idx": chunk_idx,
                "model_name": chunk_names[chunk_idx],
            }

        sequence = []
        for virtual_k in range(num_warmup_microbatches):
            sequence.append(_fwd_ref(virtual_k))
        for k in range(num_microbatches_remaining):
            sequence.append(_fwd_ref(k + num_warmup_microbatches))
            sequence.append(_bwd_ref(k))
        for virtual_k in range(num_microbatches_remaining, total_virtual_microbatches):
            sequence.append(_bwd_ref(virtual_k))
        return stage_key, sequence

    def _build_vpp_chunk_memory_profile(self, model_name: str):
        peak_point: PeakPoint = self._get_peak_point_for_model(model_name)
        cache_bytes = peak_point.activation_mem_cache
        backward_window_peak = max(
            peak_point.bwd_peak_mem,
            peak_point.recomp_fwd_peak_mem,
            peak_point.recomp_bwd_peak_mem,
        )
        if backward_window_peak == peak_point.recomp_fwd_peak_mem:
            bwd_peak_path = peak_point.recomp_fwd_peak_path
            bwd_peak_stage = "recompute_forward"
        elif backward_window_peak == peak_point.recomp_bwd_peak_mem:
            bwd_peak_path = peak_point.recomp_bwd_peak_path
            bwd_peak_stage = "recompute_backward"
        else:
            bwd_peak_path = peak_point.bwd_peak_path
            bwd_peak_stage = "backward"
        return {
            "cache_size_bytes": cache_bytes,
            "fwd_allocated_delta": cache_bytes,
            "bwd_allocated_delta": -cache_bytes,
            "fwd_peak_in_chunk": peak_point.fwd_peak_mem,
            "bwd_peak_in_chunk": max(0.0, backward_window_peak - cache_bytes),
            "fwd_peak_path": peak_point.fwd_peak_path,
            "fwd_peak_stage": "forward",
            "bwd_peak_path": bwd_peak_path,
            "bwd_peak_stage": bwd_peak_stage,
        }

    def _format_vpp_cache_per_microbatch(self, cache_bytes_by_chunk):
        if not cache_bytes_by_chunk:
            return "0.0000 GB"
        values_gb = sorted({cache_bytes / 1024**3 for cache_bytes in cache_bytes_by_chunk.values()})
        if len(values_gb) == 1:
            return f"{values_gb[0]:.4f} GB"
        return f"{values_gb[0]:.4f} ~ {values_gb[-1]:.4f} GB"

    def _analysis_sync_vpp_stage_mem_impl(self, pp_rank: int):
        stage_key, chunk_names = self._get_vpp_stage_chunk_names(pp_rank)
        if not chunk_names:
            return {}

        result = {}
        dense_model_mem, moe_model_mem, te_dummy_wgrad_model_mem = self._sum_model_mem_for_chunks(chunk_names)
        result["micro_batch_num"] = self.strategy.micro_batch_num
        result["micro_batch_size"] = self.strategy.micro_batch_size
        result["parallel_config"] = {
            "parallelism": self.strategy.parallelism,
            "fp8": self.strategy.fp8,
            "recompute_status": {
                "layer_num": self.model_config.layer_num,
                "actual_layer_num": sum(self.vpp_chunk_dict[name].layer_num for name in chunk_names),
                "recompute_layer": self.strategy.recompute_layer_num,
                "recompute_recompute_granularity": self.strategy.recompute_granularity,
            },
        }
        result["memory_schedule"] = "sync_vpp_schedule"
        result["stage_type"] = stage_key
        result["stage_rank"] = pp_rank
        result["model_mem"] = (
            dense_model_mem["all_mem"]
            + moe_model_mem["all_mem"]
            + te_dummy_wgrad_model_mem["all_mem"]
        )
        result["model_mem_detail"] = dict(
            dense=dense_model_mem,
            moe=moe_model_mem,
            te_dummy_wgrad=te_dummy_wgrad_model_mem,
        )

        stage_sequence = self._build_sync_vpp_local_phase_sequence(pp_rank)[1]
        chunk_profiles = {
            name: self._build_vpp_chunk_memory_profile(name) for name in chunk_names
        }
        result["fwd_activation_cache_per_micro_batch"] = self._format_vpp_cache_per_microbatch(
            {name: profile["cache_size_bytes"] for name, profile in chunk_profiles.items()}
        )

        live_cache_bytes = 0.0
        live_cache_entries = 0
        max_live_cache_entries = 0
        peak_activation_mem = 0.0
        peak_path = ""
        peak_stage = ""
        for item in stage_sequence:
            profile = chunk_profiles[item["model_name"]]
            if item["phase"] == "fwd":
                peak_in_chunk = profile["fwd_peak_in_chunk"]
                phase_peak_path = profile["fwd_peak_path"]
                phase_peak_stage = profile["fwd_peak_stage"]
                allocated_delta = profile["fwd_allocated_delta"]
                if allocated_delta > 0:
                    live_cache_entries += 1
            else:
                peak_in_chunk = profile["bwd_peak_in_chunk"]
                phase_peak_path = profile["bwd_peak_path"]
                phase_peak_stage = profile["bwd_peak_stage"]
                allocated_delta = profile["bwd_allocated_delta"]
                if allocated_delta < 0 and profile["cache_size_bytes"] > 0:
                    live_cache_entries -= 1

            phase_peak_mem = live_cache_bytes + peak_in_chunk
            if phase_peak_mem >= peak_activation_mem:
                peak_activation_mem = phase_peak_mem
                peak_path = (
                    f"{item['model_name']}[mb{item['microbatch']},chunk{item['chunk_idx']}]: "
                    f"{phase_peak_path}"
                )
                peak_stage = phase_peak_stage

            live_cache_bytes += allocated_delta
            max_live_cache_entries = max(max_live_cache_entries, live_cache_entries)

        assert abs(live_cache_bytes) < 1e-6, (
            f"sync VPP memory aggregation should end with zero live cache, got {live_cache_bytes}"
        )
        assert live_cache_entries == 0, (
            f"sync VPP memory aggregation should end with zero live cache entries, got {live_cache_entries}"
        )

        result["cached_micro_batch_num"] = max_live_cache_entries
        result["peak_activation_mem_in_1F1B"] = peak_activation_mem
        result["peak_mem"] = result["model_mem"] + peak_activation_mem
        result["peak_mem_with_reserved"] = result["peak_mem"] / self.strategy.mem_factor
        result["memory_reserved_ratio"] = str(self.strategy.mem_factor)
        result["peak_path"] = f"{peak_path}, stage=[{peak_stage}]"
        convert_final_result_to_human_format(result)
        return result

    def analysis_mem(self):
        """Based the simulation result, analyze the memory usage"""
        vp_size = self._vp_size()
        if (
            vp_size > 1
            and self.vpp_stage_chunk_names.get(FIRST_CHUNK)
            and not self.strategy.pp_comm_async
        ):
            if self.strategy.pp_size == 1:
                return Result(self._analysis_sync_vpp_stage_mem_impl(0))
            result = {}
            for pp_rank in range(self.strategy.pp_size):
                result[self._vpp_stage_result_key(pp_rank)] = self._analysis_sync_vpp_stage_mem_impl(pp_rank)
            return Result(result)

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
            phase = self._compute_single_batch_phase_inputs(model_name)
            inter_exposed_time_per_batch = (
                phase["fwd_recv"]
                + phase["fwd_send"]
                + phase["bwd_recv"]
                + phase["bwd_send"]
            )
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
    
    def calculate_1f1b_bubble(self, pp, mbc, forward_times, backward_times, draw=False, stage_phases=None, return_schedules=False):
        if stage_phases is None:
            schedules = [[] for _ in range(pp)]
            fwd_ready = [[0] for _ in range(pp)]
            bwd_ready = [[0] for _ in range(pp)]

            for step in range(mbc):
                for rank in range(pp):
                    warmup_step = pp - 1 - rank
                    if step < warmup_step:
                        current_time = schedules[rank][-1]["end"] if schedules[rank] else 0
                        prev_fwd = fwd_ready[rank - 1][-1] if rank > 0 else 0
                        start_time = max(current_time, prev_fwd)
                        duration = forward_times[rank]
                        schedules[rank].append({"kind": "F", "mb": len(fwd_ready[rank]), "start": start_time, "duration": duration, "end": start_time + duration, "label": "fwd_compute"})
                        fwd_ready[rank].append(start_time + duration)
                    else:
                        current_time = schedules[rank][-1]["end"] if schedules[rank] else 0
                        prev_fwd = fwd_ready[rank - 1][-1] if rank > 0 else 0
                        start_time = max(current_time, prev_fwd)
                        duration = forward_times[rank]
                        schedules[rank].append({"kind": "F", "mb": len(fwd_ready[rank]), "start": start_time, "duration": duration, "end": start_time + duration, "label": "fwd_compute"})
                        fwd_ready[rank].append(start_time + duration)

                        current_time = schedules[rank][-1]["end"]
                        next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                        start_time = max(current_time, next_bwd)
                        duration = backward_times[rank]
                        schedules[rank].append({"kind": "B", "mb": len(bwd_ready[rank]), "start": start_time, "duration": duration, "end": start_time + duration, "label": "bwd_compute"})
                        bwd_ready[rank].append(start_time + duration)

            for step in range(pp - 1, -1, -1):
                for rank in range(step):
                    current_time = schedules[rank][-1]["end"]
                    next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                    start_time = max(current_time, next_bwd)
                    duration = backward_times[rank]
                    schedules[rank].append({"kind": "B", "mb": len(bwd_ready[rank]), "start": start_time, "duration": duration, "end": start_time + duration, "label": "bwd_compute"})
                    bwd_ready[rank].append(start_time + duration)

            max_time = max(s[-1]["end"] for s in schedules)
        else:
            schedules = [[] for _ in range(pp)]
            local_t = [0.0] * pp

            class _Op:
                __slots__ = ("kind", "mb", "gid", "dur", "peer", "label", "chunk_idx", "virtual_idx")

                def __init__(self, kind, mb, gid, dur, peer, label, chunk_idx=None, virtual_idx=None):
                    self.kind = kind
                    self.mb = mb
                    self.gid = gid
                    self.dur = dur
                    self.peer = peer
                    self.label = label
                    self.chunk_idx = chunk_idx
                    self.virtual_idx = virtual_idx

            def _record(rank, op, start, end):
                schedules[rank].append(
                    {
                        "kind": op.kind,
                        "mb": op.mb,
                        "start": start,
                        "duration": end - start,
                        "end": end,
                        "label": op.label,
                        "chunk_idx": op.chunk_idx,
                        "virtual_idx": op.virtual_idx,
                        "gid": op.gid,
                    }
                )

            def _append_ordered_comm(rank, send_ops, recv_ops, out):
                ordered = send_ops + recv_ops if (rank % 2) else recv_ops + send_ops
                out.extend(ordered)

            op_queues = [[] for _ in range(pp)]
            for rank in range(pp):
                spec = stage_phases[rank]
                num_warmup = min(pp - rank - 1, mbc)
                num_remaining = mbc - num_warmup
                fwd_idx = 0
                bwd_idx = 0

                def _append_recv_prev_forward(idx):
                    if rank > 0 and spec["fwd_recv"] > 0:
                        gid = ("fwd", idx, rank - 1, rank)
                        op_queues[rank].append(_Op("recv", idx, gid, spec["fwd_recv"], rank - 1, "recv_prev"))

                def _append_send_next_forward(idx, out=None):
                    if rank < pp - 1 and spec["fwd_send"] > 0:
                        gid = ("fwd", idx, rank, rank + 1)
                        target = op_queues[rank] if out is None else out
                        target.append(_Op("send", idx, gid, spec["fwd_send"], rank + 1, "send_next"))

                def _append_recv_next_backward(idx):
                    if rank < pp - 1 and spec["bwd_recv"] > 0:
                        gid = ("bwd", idx, rank + 1, rank)
                        op_queues[rank].append(_Op("recv", idx, gid, spec["bwd_recv"], rank + 1, "recv_next"))

                def _append_send_prev_backward(idx, out=None):
                    if rank > 0 and spec["bwd_send"] > 0:
                        gid = ("bwd", idx, rank, rank - 1)
                        target = op_queues[rank] if out is None else out
                        target.append(_Op("send", idx, gid, spec["bwd_send"], rank - 1, "send_prev"))

                for _ in range(num_warmup):
                    if rank != 0:
                        _append_recv_prev_forward(fwd_idx)
                    op_queues[rank].append(_Op("F", fwd_idx, None, spec["fwd_compute"], None, "fwd_compute"))
                    if rank != pp - 1:
                        _append_send_next_forward(fwd_idx)
                    fwd_idx += 1

                for i in range(num_remaining):
                    last_iteration = i == (num_remaining - 1)
                    if rank != 0 and i == 0:
                        _append_recv_prev_forward(fwd_idx)
                    op_queues[rank].append(_Op("F", fwd_idx, None, spec["fwd_compute"], None, "fwd_compute"))
                    if rank != pp - 1:
                        send_ops, recv_ops = [], []
                        _append_send_next_forward(fwd_idx, send_ops)
                        gid = ("bwd", bwd_idx, rank + 1, rank)
                        if spec["bwd_recv"] > 0:
                            recv_ops.append(_Op("recv", bwd_idx, gid, spec["bwd_recv"], rank + 1, "recv_next"))
                        _append_ordered_comm(rank, send_ops, recv_ops, op_queues[rank])
                    fwd_idx += 1

                    op_queues[rank].append(_Op("B", bwd_idx, None, spec["bwd_compute"], None, "bwd_compute"))

                    if last_iteration:
                        if rank != 0:
                            _append_send_prev_backward(bwd_idx)
                        bwd_idx += 1
                    else:
                        if rank != 0:
                            send_ops, recv_ops = [], []
                            _append_send_prev_backward(bwd_idx, send_ops)
                            gid = ("fwd", fwd_idx, rank - 1, rank)
                            if spec["fwd_recv"] > 0:
                                recv_ops.append(_Op("recv", fwd_idx, gid, spec["fwd_recv"], rank - 1, "recv_prev"))
                            _append_ordered_comm(rank, send_ops, recv_ops, op_queues[rank])
                        bwd_idx += 1

                for _ in range(num_warmup):
                    if rank != pp - 1:
                        _append_recv_next_backward(bwd_idx)
                    op_queues[rank].append(_Op("B", bwd_idx, None, spec["bwd_compute"], None, "bwd_compute"))
                    if rank != 0:
                        _append_send_prev_backward(bwd_idx)
                    bwd_idx += 1

            while any(op_queues):
                progressed = False

                for rank in range(pp):
                    if not op_queues[rank]:
                        continue
                    op = op_queues[rank][0]
                    if op.kind not in ("F", "B"):
                        continue
                    start = local_t[rank]
                    end = start + op.dur
                    _record(rank, op, start, end)
                    local_t[rank] = end
                    op_queues[rank].pop(0)
                    progressed = True

                matched = set()
                for rank in range(pp):
                    if rank in matched or not op_queues[rank]:
                        continue
                    op = op_queues[rank][0]
                    if op.kind not in ("send", "recv"):
                        continue
                    peer = op.peer
                    if peer is None or peer in matched or not op_queues[peer]:
                        continue
                    peer_op = op_queues[peer][0]
                    if peer_op.kind not in ("send", "recv"):
                        continue
                    if op.gid != peer_op.gid or op.kind == peer_op.kind:
                        continue
                    start_rank = local_t[rank]
                    start_peer = local_t[peer]
                    end = max(start_rank, start_peer) + max(op.dur, peer_op.dur)
                    _record(rank, op, start_rank, end)
                    _record(peer, peer_op, start_peer, end)
                    local_t[rank] = end
                    local_t[peer] = end
                    op_queues[rank].pop(0)
                    op_queues[peer].pop(0)
                    matched.add(rank)
                    matched.add(peer)
                    progressed = True

                if not progressed:
                    raise RuntimeError(f"1F1B phase scheduler deadlock: {[q[0].label if q else None for q in op_queues]}")

            max_time = max(local_t) if local_t else 0.0

        if draw:
            output_json_path = draw if isinstance(draw, str) else os.path.abspath("corrected_1F1B_pipeline_trace.json")
            export_pipeline_schedule_trace(schedules, output_json_path, title=f"perf_1f1b_pp{pp}_mbc{mbc}")

        schedules = normalize_schedule_records(schedules)
        if return_schedules:
            return max_time, schedules
        return max_time

    def _get_vpp_chunk_phase_inputs(self, pp_rank: int, chunk_idx: int):
        if pp_rank == 0:
            stage_key = FIRST_CHUNK
            fallback_name = FIRST_CHUNK
        elif pp_rank == self.strategy.pp_size - 1:
            stage_key = LAST_CHUNK
            fallback_name = LAST_CHUNK
        else:
            stage_key = MIDDLE_CHUNK
            fallback_name = MIDDLE_CHUNK
        chunk_names = self.vpp_stage_chunk_names.get(stage_key, [])
        model_name = chunk_names[chunk_idx] if chunk_idx < len(chunk_names) else fallback_name
        return model_name, self._compute_single_batch_phase_inputs(model_name)

    def _compute_interleaved_sync_schedule(self, return_schedules=False, draw=False):
        return self._compute_interleaved_sync_schedule_from_ppschedule(return_schedules=return_schedules, draw=draw)

    def _compute_interleaved_sync_schedule_from_ppschedule(self, return_schedules=False, draw=False):
        pp_size = self.strategy.pp_size
        vp_size = self._vp_size()
        mbc = self.strategy.micro_batch_num
        assert pp_size > 1 and vp_size > 1
        if self.strategy.pp_comm_async:
            raise RuntimeError(
                "perf interleaved sync schedule does not support async VPP; "
                "disable pp_comm_async or use simulate() for async VPP runtime behavior"
            )
        from types import SimpleNamespace
        from simumax.core.base_struct import BaseModel, AtomModel, FwdQue, BwdStk, State_Thread
        from simumax.core.transformer.pipeline_schedule import PpSchedule

        schedules = [[] for _ in range(pp_size)]
        local_t = [0.0] * pp_size

        class _DummyStageModel(BaseModel):
            def __init__(self, *, strategy, system, hidden_size, phase, model_name, chunk_idx, vp_size):
                super().__init__(specific_name=model_name)
                self.strategy = strategy
                self.system = system
                self.model_config = SimpleNamespace(hidden_size=hidden_size)
                self.phase = phase
                self.chunk_idx = chunk_idx
                self.vp_size = vp_size
                self.layers = [AtomModel(fwd_cost=phase["fwd_compute"], bwd_cost=phase["bwd_compute"], specific_name="perf_stage")]

            def prefill(self, args, call_stk='', com_buff=None):
                self.call_stk = call_stk + self.call_stk
                self._real_mb = args.microbatch
                self._mb_virtual = getattr(args, "virtual_microbatch", args.microbatch)

            def prefill_fwd(self):
                q = super().prefill_fwd()
                q._perf_kind = "F"
                q._perf_label = self.call_stk.lstrip("-")
                q._perf_mb = self._real_mb
                q._perf_chunk_idx = self.chunk_idx
                return q

            def prefill_bwd(self):
                q = super().prefill_bwd()
                q._perf_kind = "B"
                q._perf_label = self.call_stk.lstrip("-")
                q._perf_mb = self._real_mb
                q._perf_chunk_idx = self.chunk_idx
                return q

        class _Op:
            __slots__ = (
                "kind",
                "mb",
                "gid",
                "dur",
                "label",
                "chunk_idx",
                "virtual_idx",
                "bundle_ops",
                "posted",
                "remaining",
                "start_t",
                "max_end",
                "ready_t",
                "done",
                "wait_gid",
            )
            def __init__(
                self,
                kind,
                mb,
                gid,
                dur,
                label,
                chunk_idx=None,
                virtual_idx=None,
                bundle_ops=None,
            ):
                self.kind = kind
                self.mb = mb
                self.gid = gid
                self.dur = dur
                self.label = label
                self.chunk_idx = chunk_idx
                self.virtual_idx = virtual_idx
                self.bundle_ops = bundle_ops or []
                self.posted = False
                self.remaining = len(self.bundle_ops)
                self.start_t = None
                self.max_end = 0.0
                self.ready_t = None
                self.done = False
                self.wait_gid = None

        def _record(rank, op, start, end):
            schedules[rank].append(
                {
                    "kind": op.kind,
                    "mb": op.mb,
                    "start": start,
                    "duration": end - start,
                    "end": end,
                    "label": op.label,
                    "chunk_idx": op.chunk_idx,
                    "virtual_idx": op.virtual_idx,
                    "gid": op.gid,
                }
            )

        job_queues = [[] for _ in range(pp_size)]
        for pp_rank in range(pp_size):
            if pp_rank == 0:
                stage_key = FIRST_CHUNK
            elif pp_rank == pp_size - 1:
                stage_key = LAST_CHUNK
            else:
                stage_key = MIDDLE_CHUNK
            stage_models = []
            for chunk_idx, model_name in enumerate(self.vpp_stage_chunk_names.get(stage_key, [])):
                _, phase = self._get_vpp_chunk_phase_inputs(pp_rank, chunk_idx)
                stage_models.append(
                    _DummyStageModel(
                        strategy=self.strategy,
                        system=self.system,
                        hidden_size=self.model_config.hidden_size,
                        phase=phase,
                        model_name=f"{model_name}_model",
                        chunk_idx=chunk_idx,
                        vp_size=vp_size,
                    )
                )
            sched = PpSchedule(self.strategy, self.system, stage_models)
            rank = get_pp_stage_representative_rank(pp_rank, self.strategy)
            args = SimpleNamespace(thread_state=State_Thread(), rank=rank, microbatch=0)
            jobs = sched.prefill_batch(args, com_buff=None)
            for job in jobs:
                if isinstance(job, FwdQue):
                    if getattr(job, "_perf_kind", None) == "F":
                        dur = sum(getattr(leaf, "fwd_cost", 0.0) for leaf in job.que)
                        job_queues[pp_rank].append(
                            _Op("F", job._perf_mb, None, dur, job._perf_label, chunk_idx=job._perf_chunk_idx)
                        )
                    else:
                        bundle_ops = []
                        for leaf in job.que:
                            label = leaf.__class__.__name__
                            kind = "send" if label.startswith("send") else "recv"
                            bundle_ops.append(
                                _Op(kind, None, leaf.id, getattr(leaf, "fwd_cost", 0.0), label)
                            )
                        if bundle_ops:
                            job_queues[pp_rank].append(
                                _Op("comm", None, None, 0.0, "comm", bundle_ops=bundle_ops)
                            )
                elif isinstance(job, BwdStk):
                    if getattr(job, "_perf_kind", None) == "B":
                        dur = sum(getattr(leaf, "bwd_cost", 0.0) for leaf in job.stk)
                        job_queues[pp_rank].append(
                            _Op("B", job._perf_mb, None, dur, job._perf_label, chunk_idx=job._perf_chunk_idx)
                        )
                    else:
                        bundle_ops = []
                        for leaf in reversed(job.stk):
                            label = leaf.__class__.__name__
                            kind = "send" if label.startswith("send") else "recv"
                            bundle_ops.append(
                                _Op(kind, None, leaf.id, getattr(leaf, "bwd_cost", 0.0), label)
                            )
                        if bundle_ops:
                            job_queues[pp_rank].append(
                                _Op("comm", None, None, 0.0, "comm", bundle_ops=bundle_ops)
                            )

        heap = []
        ver = [0] * pp_size
        waiting = {}

        def _push(rank):
            ver[rank] += 1
            heapq.heappush(heap, (local_t[rank], rank, ver[rank]))

        for rank in range(pp_size):
            _push(rank)

        while any(job_queues) or waiting:
            if not heap:
                raise RuntimeError(f"interleaved sync scheduler deadlock: {[q[0].label if q else None for q in job_queues]}")
            _, rank, token = heapq.heappop(heap)
            if token != ver[rank]:
                continue
            if not job_queues[rank]:
                continue
            op = job_queues[rank][0]
            if op.kind in ("F", "B"):
                start = local_t[rank]
                end = start + op.dur
                _record(rank, op, start, end)
                local_t[rank] = end
                job_queues[rank].pop(0)
                _push(rank)
                continue

            if op.kind != "comm":
                raise RuntimeError(f"unknown op kind in perf interleaved scheduler: {op.kind}")

            op.posted = True
            if op.start_t is None:
                op.start_t = local_t[rank]
            submit_t = local_t[rank]
            local_cursor = submit_t
            op.max_end = max(op.max_end, submit_t)
            op.wait_gid = None

            for sub_op in op.bundle_ops:
                if sub_op.done:
                    continue
                if sub_op.ready_t is None:
                    # A blocking batched p2p call issues all non-None ops
                    # together, so every sub-op in the bundle shares the same
                    # local submission timestamp.
                    sub_op.ready_t = submit_t

                waiter = waiting.get(sub_op.gid)
                if waiter is None:
                    waiting[sub_op.gid] = (rank, op, sub_op)
                    if op.wait_gid is None:
                        op.wait_gid = sub_op.gid
                    continue

                peer_rank, peer_step, peer_sub_op = waiter
                if peer_rank == rank:
                    if peer_step is not op or peer_sub_op is not sub_op:
                        raise RuntimeError(f"duplicate waiting rank for gid {sub_op.gid}")
                    if op.wait_gid is None:
                        op.wait_gid = sub_op.gid
                    continue
                if peer_sub_op.kind == sub_op.kind:
                    raise RuntimeError(
                        f"same-kind rendezvous for gid {sub_op.gid}: {peer_sub_op.kind}"
                    )
                waiting.pop(sub_op.gid, None)

                end = max(sub_op.ready_t, peer_sub_op.ready_t) + max(sub_op.dur, peer_sub_op.dur)
                _record(rank, sub_op, sub_op.ready_t, end)
                _record(peer_rank, peer_sub_op, peer_sub_op.ready_t, end)
                sub_op.done = True
                peer_sub_op.done = True
                local_cursor = max(local_cursor, end)
                op.max_end = max(op.max_end, end)
                local_t[peer_rank] = max(local_t[peer_rank], end)
                peer_step.max_end = max(peer_step.max_end, end)

                if (
                    peer_step.wait_gid == sub_op.gid
                    and job_queues[peer_rank]
                    and job_queues[peer_rank][0] is peer_step
                ):
                    _push(peer_rank)

            local_t[rank] = max(local_t[rank], local_cursor)

            op.remaining = sum(0 if sub_op.done else 1 for sub_op in op.bundle_ops)

            if op.remaining == 0:
                local_t[rank] = max(local_t[rank], op.max_end)
                job_queues[rank].pop(0)
                _push(rank)

        max_time = max(local_t) if local_t else 0.0
        if draw:
            output_json_path = draw if isinstance(draw, str) else os.path.abspath("corrected_vpp_1F1B_pipeline_trace.json")
            export_pipeline_schedule_trace(
                schedules,
                output_json_path,
                title=f"perf_sync_vpp_pp{pp_size}_vp{vp_size}_mbc{mbc}",
                serialize_comm_lanes=False,
            )
        schedules = normalize_schedule_records(schedules)
        if return_schedules:
            return max_time, schedules
        return max_time

    def export_pp_schedule_trace(self, output_json_path):
        output_json_path = os.path.abspath(output_json_path)
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        vp_size = self._vp_size()
        if vp_size > 1 and self.vpp_stage_chunk_names.get(FIRST_CHUNK):
            _, schedules = self._compute_interleaved_sync_schedule(return_schedules=True)
            export_pipeline_schedule_trace(
                schedules,
                output_json_path,
                title=f"perf_sync_vpp_pp{self.strategy.pp_size}_vp{vp_size}_mbc{self.strategy.micro_batch_num}",
                serialize_comm_lanes=False,
            )
            return output_json_path

        stage_phases = [self._compute_single_batch_phase_inputs(FIRST_CHUNK)]
        if self.strategy.pp_size > 2:
            stage_phases.extend([self._compute_single_batch_phase_inputs(MIDDLE_CHUNK)] * (self.strategy.pp_size - 2))
        if self.strategy.pp_size > 1:
            stage_phases.append(self._compute_single_batch_phase_inputs(LAST_CHUNK))
        _, schedules = self.calculate_1f1b_bubble(
            self.strategy.pp_size,
            self.strategy.micro_batch_num,
            forward_times=[sum((spec["fwd_recv"], spec["fwd_compute"], spec["fwd_send"])) for spec in stage_phases],
            backward_times=[sum((spec["bwd_recv"], spec["bwd_compute"], spec["bwd_send"])) for spec in stage_phases],
            stage_phases=stage_phases,
            return_schedules=True,
        )
        export_pipeline_schedule_trace(schedules, output_json_path, title=f"perf_sync_pp{self.strategy.pp_size}_mbc{self.strategy.micro_batch_num}")
        return output_json_path
    def _chunk_stage_key(self, model_name):
        if model_name in (FIRST_CHUNK, MIDDLE_CHUNK, LAST_CHUNK):
            return model_name
        for stage_key, chunk_names in self.vpp_stage_chunk_names.items():
            if model_name in chunk_names:
                return stage_key
        return model_name

    def _compute_single_batch_phase_inputs(self, model_name):
        model_obj = self.model_chunk_dict.get(model_name)
        if model_obj is None:
            model_obj = self.vpp_chunk_dict.get(model_name)
        if model_obj is None:
            raise KeyError(f"Unknown model chunk: {model_name}")
        cost_info = model_obj.get_cost_info()

        p2p_time = 0.0
        if self.strategy.pp_size > 1:
            pp_comm_size = get_pp_p2p_comm_size(
                self.strategy,
                self.model_config.hidden_size,
                self.dtype_to_element_size[self.strategy.dtype],
            )
            p2p_time = self.system.compute_net_op_time("p2p", pp_comm_size, 2, net=self.strategy.pp_net)

        stage_key = self._chunk_stage_key(model_name)
        if self.strategy.pp_size <= 1:
            fwd_recv = fwd_send = bwd_recv = bwd_send = 0.0
        elif stage_key == FIRST_CHUNK:
            fwd_recv, fwd_send = 0.0, p2p_time
            bwd_recv, bwd_send = p2p_time, 0.0
        elif stage_key == LAST_CHUNK:
            fwd_recv, fwd_send = p2p_time, 0.0
            bwd_recv, bwd_send = 0.0, p2p_time
        else:
            fwd_recv = fwd_send = p2p_time
            bwd_recv = bwd_send = p2p_time

        return {
            "fwd_recv": fwd_recv,
            "fwd_compute": cost_info.fwd_compute_time + cost_info.fwd_net_time,
            "fwd_send": fwd_send,
            "bwd_recv": bwd_recv,
            "bwd_compute": (
                cost_info.bwd_compute_time
                + cost_info.bwd_net_time
                + cost_info.recompute_compute_time
                + cost_info.recompute_net_time
            ),
            "bwd_send": bwd_send,
        }

    def _compute_single_batch_fwd_bwd_time(self, model_name, chunk = False):
            phase = self._compute_single_batch_phase_inputs(model_name)
            fwd_chunk_time = phase["fwd_recv"] + phase["fwd_compute"] + phase["fwd_send"]
            bwd_chunk_time = phase["bwd_recv"] + phase["bwd_compute"] + phase["bwd_send"]
            return (fwd_chunk_time, bwd_chunk_time) if not chunk else fwd_chunk_time + bwd_chunk_time
    
    def _compute_pp_total_time(self):
        vp_size = self._vp_size()
        if vp_size > 1 and self.vpp_stage_chunk_names.get(FIRST_CHUNK):
            if self.strategy.pp_comm_async:
                raise RuntimeError(
                    "analysis_cost() does not support async VPP perf timing yet; "
                    "use pp_comm_async=False for perf timing or simulate() for async VPP traces"
                )
            return self._compute_interleaved_sync_schedule()

        stage_phases = [self._compute_single_batch_phase_inputs(FIRST_CHUNK)]
        has_middle_chunks = self.strategy.pp_size > 2
        has_last_chunk = self.strategy.pp_size > 1
        if has_middle_chunks:
            stage_phases.extend([self._compute_single_batch_phase_inputs(MIDDLE_CHUNK)] * (self.strategy.pp_size - 2))
        if has_last_chunk:
            stage_phases.append(self._compute_single_batch_phase_inputs(LAST_CHUNK))

        single_iter_time = self.calculate_1f1b_bubble(
            self.strategy.pp_size,
            self.strategy.micro_batch_num,
            forward_times=[sum((spec["fwd_recv"], spec["fwd_compute"], spec["fwd_send"])) for spec in stage_phases],
            backward_times=[sum((spec["bwd_recv"], spec["bwd_compute"], spec["bwd_send"])) for spec in stage_phases],
            draw=False,
            stage_phases=stage_phases,
        )
        return single_iter_time
    
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
        single_iter_time_no_dp_opim = self._compute_pp_total_time() # (mbc * chunk_time + bubble_time) 

        # Straggler modeling is applied at machine granularity: GPUs within one
        # node are assumed stable, while node-to-node runtime may fluctuate.
        # The effective sample count is capped by the number of nodes and the
        # active dense-/expert-DP replica counts.
        dp = self.strategy.dp_size
        edp = self.strategy.edp_size
        if self.strategy.enable_straggler_model:
            effective_worker_count = get_effective_straggler_sample_count(
                world_size=self.strategy.world_size,
                num_per_node=self.system.num_per_node,
                dp_size=dp,
                edp_size=edp,
            )
            increase_ratio = estimate_straggler_increase_ratio(effective_worker_count)
        else:
            increase_ratio = 1.0
        single_iter_time_no_dp_opim = increase_ratio*single_iter_time_no_dp_opim # (mbc * chunk_time + bubble_time) * increase_ratio
        
        duration_times = [single_iter_time_no_dp_opim + get_dp_and_optim(FIRST_CHUNK)]
        duration_times.append(single_iter_time_no_dp_opim + get_dp_and_optim(MIDDLE_CHUNK)) if self.strategy.pp_size > 2 else 0
        duration_times.append(single_iter_time_no_dp_opim + get_dp_and_optim(LAST_CHUNK)) if self.strategy.pp_size > 1 else 0

        final_duration_time_per_iter = max(duration_times)
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
        all_result['straggle_ratio'] = increase_ratio

        def format_chunk_time(model_chunk, chunk_time, duration_time):
            return {
                model_chunk:{
                    'duration_time(chunk_timexmbc+dp_optim+bubble)': duration_time,
                    'chunk_time(fwd+bwd)':chunk_time,
                    'dp_and_optim_time': get_dp_and_optim(model_chunk),
                    'bubble_time': single_iter_time_no_dp_opim/increase_ratio - mbc*(chunk_time),
                    'straggler_time': single_iter_time_no_dp_opim *(increase_ratio - 1)/increase_ratio,
                }
            }
        all_result['all_chunk_times'] = format_chunk_time(FIRST_CHUNK, chunk_time, duration_times[0])
        all_result['all_chunk_times'].update(format_chunk_time(MIDDLE_CHUNK, chunk_time_middle_stage, duration_times[1]) if pp_size > 2 else {})
        all_result['all_chunk_times'].update(format_chunk_time(LAST_CHUNK, chunk_time_lstage, duration_times[-1]) if pp_size > 1 else {})

        all_result.update({
            'duration_time_per_iter': final_duration_time_per_iter,
            'throughput_per_accelerator': TGS,
            'throughput per GPU (TFLOP/s/GPU)': TFLOPS,
            'throughput per GPU per token (TFLOP/s/GPU/token)': TFLOPS_PER_TOKEN,
            'mfu_6nd_with_attn': new_mfu_6nd_with_attn,
            'mfu':new_mfu_6nd_with_attn,
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
        else:
            all_result['param_numel_info'].update({
                    "activations" : all_result['param_numel_info']['all'],
                    "activations_ratio" : f'{100:.2f}%',
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
    
    def _run(self):
        if self._chunk_profile_cache_enabled() and len(self._prepared_chunk_names) == len(self.model_chunk_dict):
            return
        # Fake first stage input

        input_info_first_stage = InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(self.strategy.micro_batch_size, self.strategy.seq_len//self.strategy.cp_size)
                )
            ]
        )
        self.path_debug_context = PathDebugContext(
            point_datas={},
            point_datas_with_recomp={},
            target_point=self.debug_points,
            path_list=[],
        )
        _ = self.model_chunk_dict[FIRST_CHUNK](
            input_info_first_stage, self.path_debug_context
        )
        self.pp_state_peak_point[FIRST_CHUNK] = self.model_chunk_dict[FIRST_CHUNK].compute_activations()
        if self.strategy.pp_size > 2:
            seq_len = (
                self.strategy.seq_len // self.strategy.tp_size
                if self.strategy.enable_sequence_parallel
                else self.strategy.seq_len
            )
            input_info_last_stage = InputOutputInfo(
                tensors=[
                    TensorSize(
                        shape=(
                            self.strategy.micro_batch_size,
                            seq_len//self.strategy.cp_size,
                            self.model_config.hidden_size,
                        )
                    )
                ]
            )
            self.path_debug_context_last_stage = PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                # target_point=self.debug_points_last_stage,
                path_list=[],
            )
            _ = self.model_chunk_dict[MIDDLE_CHUNK](
                input_info_last_stage, self.path_debug_context_last_stage
            )    
            self.pp_state_peak_point[MIDDLE_CHUNK] = self.model_chunk_dict[MIDDLE_CHUNK].compute_activations()
        if self.strategy.pp_size > 1:
            seq_len = (
                self.strategy.seq_len // self.strategy.tp_size
                if self.strategy.enable_sequence_parallel
                else self.strategy.seq_len
            )
            input_info_last_stage = InputOutputInfo(
                tensors=[
                    TensorSize(
                        shape=(
                            self.strategy.micro_batch_size,
                            seq_len // self.strategy.cp_size,
                            self.model_config.hidden_size,
                        )
                    )
                ]
            )
            self.path_debug_context_last_stage = PathDebugContext(
                point_datas={},
                point_datas_with_recomp={},
                target_point=self.debug_points_last_stage,
                path_list=[],
            )
            _ = self.model_chunk_dict[LAST_CHUNK](
                input_info_last_stage, self.path_debug_context_last_stage
            )    
            self.pp_state_peak_point[LAST_CHUNK] = self.model_chunk_dict[LAST_CHUNK].compute_activations()

        # Run interleaving virtual chunks to materialize cost_info for timing analysis.
        if self._vp_size() > 1 and self.vpp_chunk_dict:
            input_info_first_stage_vpp = InputOutputInfo(
                tensors=[
                    TensorSize(
                        shape=(self.strategy.micro_batch_size, self.strategy.seq_len // self.strategy.cp_size)
                    )
                ]
            )
            seq_len_hidden = (
                self.strategy.seq_len // self.strategy.tp_size
                if self.strategy.enable_sequence_parallel
                else self.strategy.seq_len
            )
            input_info_hidden_stage_vpp = InputOutputInfo(
                tensors=[
                    TensorSize(
                        shape=(
                            self.strategy.micro_batch_size,
                            seq_len_hidden // self.strategy.cp_size,
                            self.model_config.hidden_size,
                        )
                    )
                ]
            )
            for stage_key, chunk_names in self.vpp_stage_chunk_names.items():
                for chunk_name in chunk_names:
                    vpp_chunk = self.vpp_chunk_dict[chunk_name]
                    if vpp_chunk.preprocess:
                        _ = vpp_chunk(input_info_first_stage_vpp, PathDebugContext(point_datas={}, point_datas_with_recomp={}, target_point=[], path_list=[]))
                    else:
                        _ = vpp_chunk(input_info_hidden_stage_vpp, PathDebugContext(point_datas={}, point_datas_with_recomp={}, target_point=[], path_list=[]))
                    self.pp_state_peak_point[chunk_name] = vpp_chunk.compute_activations()

    def get_pp_stage_peak_mem(self, mem_result, peak_mem_key, toG:bool = False):
        assert peak_mem_key in ["peak_mem_with_reserved", "peak_mem"], f"peak_mem_key should be in ['peak_mem_with_reserved', 'peak_mem'] but got {peak_mem_key}"
        if isinstance(mem_result, Result):
            mem_result = mem_result.data
        pp_size = self.strategy.pp_size
        if "peak_mem" in mem_result:
            peak_mem =  HumanReadableSize.from_string(
                mem_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
            return dict(
                first_stage = peak_mem/ 1024 / 1024 / 1024 if toG else peak_mem, 
            )

        peak_mem_list = {}
        for stage_key, stage_result in mem_result.items():
            if not isinstance(stage_result, dict) or peak_mem_key not in stage_result:
                continue
            peak_mem_list[stage_key] = HumanReadableSize.from_string(
                stage_result.get(peak_mem_key),
                base=1024,
                units=HumanReadableSize.BYTE_UNITS,
                target_unit="B",
            ).get_value()
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
            peak_cached_mem_bytes = max(self.get_pp_stage_peak_mem(mem_result, "peak_mem", False).values())
            
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

    def search_max_micro_batch_size_fixed_gbs(self, pp_size, dp_size, global_batch_size, memory_utils = 1.0, gmi_error=6, use_reserved_memory=True, save_all=True, verbose: bool = True):
        """
        Fixes `global_batch_size` and searches for the maximum possible `micro_batch_size` under the current parallel strategy.

        Args:
            gmi_error (int): Per-rank memory margin in GiB reserved for NCCL
                buffers, allocator/runtime overhead, and other components that
                are not modeled explicitly.
            verbose (bool): Whether to print search-progress logs such as vocab
                padding information while evaluating candidates.
        """
        gmi_error = gmi_error * 1024**3
        PEAK_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list = [], [], [], []
        accelerator_mem_bytes = self.system.accelerator.mem_gbs * 1024**3 * memory_utils
        origin_micro_batch_size = self.strategy.micro_batch_size
        origin_micro_batch_num = self.strategy.micro_batch_num
        origin_search_verbose = getattr(self, "_search_verbose", True)
        self._search_verbose = verbose

        try:
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
                peak_cached_mem_bytes = max(self.get_pp_stage_peak_mem(mem_result, PEAK_KEY, False).values())
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

            return all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list
        finally:
            self.strategy.micro_batch_size = origin_micro_batch_size
            self.strategy.micro_batch_num = origin_micro_batch_num
            self._search_verbose = origin_search_verbose


    def log_available_strategy(self, mfu, peak_mem):
        if getattr(self, "_search_verbose", True):
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
            'param_num': cost_result.data['param_numel_info']['all'],
            'system': self.system.sys_name,
            'parallelism': f'{dtype}.dense{self.model_config.dense_layers}.{self.strategy.parallelism}',
            'recompute_status': self.strategy.recompute_status,
            'mfu': cost_result.data["mfu_6nd_with_attn"],
            'TFLOPS': cost_result.data['throughput per GPU (TFLOP/s/GPU)'],
            'TGS_per_gpu' : cost_result.data['throughput_per_accelerator'],
            'iter_time':  cost_result.data["duration_time_per_iter"],
            'straggle_ratio':  cost_result.data['straggle_ratio'],
            'activations_ratio': cost_result.data['param_numel_info']['activations_ratio'],
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
        if self.strategy.megatron_recompute:
            raise NotImplementedError(
                "search does not support megatron_recompute yet; "
                "please evaluate megatron_recompute strategies explicitly"
            )
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
            gmi_error (int): Per-rank memory margin in GiB reserved for runtime
                and communication overhead not modeled explicitly.
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

    def search_best_parallel_strategy(self,
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
                                                 dump_path:str=None,
                                                 verbose: bool = True):
        if self.strategy.megatron_recompute:
            raise NotImplementedError(
                "search does not support megatron_recompute yet; "
                "please evaluate megatron_recompute strategies explicitly"
            )
        """
        Searches for the optimal combination of parallel strategies (tp/ep/pp) and full recompute layer configuration that maximizes performance under fixed global batch size constraints.

        Args:
            world_size (int): world size
            gmi_error (int): Per-rank memory margin in GiB reserved for NCCL
                buffers, allocator/runtime overhead, and other components that
                are not modeled explicitly.
            micro_batch_size (int):  fixed micro batch size
            global_batch_size (int): fixed global batch size
            all_search_result (dict): all search result of this model, include (tp, ep, pp, recompute_layer_num) combination.
        Returns:
            best_strategy (dict): the best strategy of this model, include (tp, ep, pp, full_recompute_layer_num) combination.
        """
        
        # 256张卡 world_size
        # tp: 1 2 4 8
        # ep: 1 2 4 8
        # 且 layer num整除
        if not isinstance(recompute_search_type, list):
            recompute_search_type = [recompute_search_type]

        layer_num = self.model_config.layer_num
        if tp_search_list is None:
            tp_search_list = [1, 2, 4, 8]  if self.model_config.model_type == "dense" else [1]
        if  ep_search_list is None:
            ep_search_list = [1, 2, 4, 8] if self.model_config.model_type == "moe" else [1]
        if pp_search_list is None:
            pp_search_list = list(range(1, layer_num+1))    
            
        if self.model_config.expert_num == 1:
            ep_search_list = [1]
        else:
            tp_search_list = [1]

        origin_search_verbose = getattr(self, "_search_verbose", True)
        self._search_verbose = verbose
        global_best_strategy = {}
        best_strategy_cost_path = (
            f"{dump_path}/best_strategy_costs" if dump_path is not None else None
        )

        def build_valid_candidate_strategy(
            world_size, tp_size, ep_size, etp_size, pp_size, num_layers_in_last_pipeline_stage
        ):
            candidate_strategy = deepcopy(self.strategy)
            candidate_strategy.world_size = world_size
            candidate_strategy.tp_size = tp_size
            candidate_strategy.ep_size = ep_size
            candidate_strategy.pp_size = pp_size
            candidate_strategy.etp_size = etp_size
            candidate_strategy.num_layers_in_first_pipeline_stage = None
            candidate_strategy.num_layers_in_last_pipeline_stage = (
                num_layers_in_last_pipeline_stage
            )

            origin_strategy = self.strategy
            try:
                candidate_strategy.sanity_check()
                self.strategy = candidate_strategy
                self._cross_sanity_check()
                return candidate_strategy
            except (AssertionError, ValueError, ZeroDivisionError) as exc:
                if verbose:
                    print(
                        "skip invalid strategy by sanity_check:",
                        {
                            "world_size": world_size,
                            "tp_size": tp_size,
                            "cp_size": candidate_strategy.cp_size,
                            "ep_size": ep_size,
                            "etp_size": etp_size,
                            "pp_size": pp_size,
                            "reason": str(exc),
                        },
                    )
                return None
            finally:
                self.strategy = origin_strategy

        def check_pp_valid(layer_num, pp_size):
            if pp_size > 1:
                num_layers_per_pp = math.ceil(layer_num/pp_size)
                is_pp_valid = num_layers_per_pp > 0
                num_layers_in_last_pipeline_stage = layer_num - (num_layers_per_pp * (pp_size - 1))
                is_pp_valid = is_pp_valid and num_layers_in_last_pipeline_stage > 0
            else:
                num_layers_in_last_pipeline_stage = None
                is_pp_valid = True
            return is_pp_valid, num_layers_in_last_pipeline_stage
        
        
        if verbose:
            print(f"Start search strategy for world_size={world_size}, model_type={self.model_config.model_type}, model_name={self.model_config.model_name}, system={self.system.sys_name}")
            print(f"- tp_search_list={tp_search_list}, ep_search_list={ep_search_list}, pp_search_list={pp_search_list}")
            print(f"- layer_num={layer_num}")
            print(f"- moe_pad_expert_input_to_capacity={self.model_config.moe_pad_expert_input_to_capacity}")
            print(f"- capacity={self.model_config.capacity}")
        
        global_best_mfu = -1
        try:
            for tp_size in tp_search_list:
                for ep_size in ep_search_list:
                    for pp_size in pp_search_list:
                        etp_size = tp_size if use_etp else 1
                        is_pp_valid, num_layers_in_last_pipeline_stage = check_pp_valid(layer_num=layer_num,
                                                    pp_size=pp_size)

                        candidate_strategy = None
                        if is_pp_valid:
                            candidate_strategy = build_valid_candidate_strategy(
                                world_size=world_size,
                                tp_size=tp_size,
                                ep_size=ep_size,
                                etp_size=etp_size,
                                pp_size=pp_size,
                                num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
                            )

                        if candidate_strategy is not None:
                            self.strategy = candidate_strategy


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
                                        ori_variance_status = self.strategy.recompute_variance
                                        self.strategy.recompute_variance = True
                                        search_best_strategy = self.search_best_strategy_no_recompute(gmi_error=gmi_error,
                                                                                                        use_reserved_memory=use_reserved_memory,
                                                                                                        best_mfu=global_best_mfu,
                                                                                                        all_search_result=all_search_result)
                                        self.strategy.recompute_variance = ori_variance_status

                                    elif recompute_type == 'full_block':
                                        self.strategy.recompute_granularity = "full_block"
                                        ori_variance_status = self.strategy.recompute_variance
                                        self.strategy.recompute_variance = False # megatron-LM's full recompute does not support variance
                                        search_best_strategy = self.search_best_recompute_layer_num(
                                                                                layer_num=self.model_config.layer_num,
                                                                                use_reserved_memory = use_reserved_memory,
                                                                                gmi_error=gmi_error,
                                                                                best_mfu=global_best_mfu,
                                                                                all_search_result=all_search_result,
                                                                                save_path=best_strategy_cost_path)
                                        self.strategy.recompute_variance = ori_variance_status
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
                                    if verbose:
                                        print(f"miss efficiency:")
                                        pprint(self.system.miss_efficiency)
                                    self.system.reset_record_info()

            if dump_path is not None and len(global_best_strategy) > 0:
                model_name = self.model_config.model_name
                system_name = self.system.sys_name
                os.makedirs(dump_path, exist_ok=True)

                if 'peak_mem' in global_best_strategy and isinstance(global_best_strategy['peak_mem'], dict):
                    global_best_strategy['peak_mem'] = str(global_best_strategy['peak_mem']) # serialize dict to string to avoid csv dump error
                best_strategy_df = pd.DataFrame(global_best_strategy, index=[0])
                best_strategy_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seqlen{self.strategy.seq_len}_worldsize{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_best_strategy.csv")
                if verbose:
                    print(best_strategy_df)

                if all_search_result is not None:
                    all_search_result_df = pd.DataFrame(all_search_result)
                    all_search_result_df = all_search_result_df.sort_values(by ='mfu',  ascending=False)
                    all_search_result_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seqlen{self.strategy.seq_len}_worldsize{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_all_search_strategies.csv")

            return global_best_strategy
        finally:
            self._search_verbose = origin_search_verbose
    
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
        
        # with open(f"{save_path}/strategy_config.json", "w") as f:
        #     f.write(str(self.strategy))

        with open(f"{save_path}/system_config.json", "w") as f:
            f.write(str(self.system))
        
        with open(f"{save_path}/net_info.json", "w") as f:
            json.dump(self.system.real_comm_bw, f, indent=4)
            
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
                
            with open(f"{save_path}/net_info.json", "w") as f:
                json.dump(self.system.real_comm_bw, f, indent=4)
            
            with open(f"{save_path}/system_config.json", "w") as f:
                f.write(str(self.system))

            with open(f"{save_path}/model_config.json", "w") as f:
                f.write(str(self.model_config))
            
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
        
    
    def simulate(self, save_path, merge_lanes=True):
        """This function simulates operator scheduling and communication synchronization to evaluate end-to-end performance and generate execution traces. """
        run_simulation(self, save_path, merge_lanes=merge_lanes)
 
