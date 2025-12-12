"""performance model for LLM"""

from abc import ABC, abstractmethod
import os
import math
import json
from copy import deepcopy
from typing import List, Union, Dict
from sympy import divisors
import matplotlib.pyplot as plt
import pandas as pd
from simumax.core.base_struct import PathDebugContext
from simumax.core.config import StrategyConfig, SystemConfig, ModelConfig, set_capture_graph_only, TMP_PATH, SIMU_CHECK, SIMU_DEBUG, ENABLE_SIMU_GRAPH
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

        self._run()
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
        os.makedirs(TMP_PATH, exist_ok=True)

    def __del__(self):
        try:
            import shutil
            if not SIMU_CHECK:
                if os.path.exists(TMP_PATH):
                    shutil.rmtree(TMP_PATH)
        except Exception as e:
            print(f"删除文件时出错: {e}")
            
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
            chunk_weight_accessed_time = 3 * state_weight_bytes # why 3倍?
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
    
    def calculate_1f1b_bubble(self, pp, mbc, forward_times, backward_times, draw=False):
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
                    duration = forward_times[rank]
                    schedules[rank].append(('F', len(fwd_ready[rank]), start_time, duration, start_time+duration))
                    fwd_ready[rank].append(start_time + duration)
                else:
                    "F-B"
                    current_time = schedules[rank][-1][4] if schedules[rank] else 0
                    prev_fwd = fwd_ready[rank - 1][-1] if rank > 0 else 0
                    start_time = max(current_time, prev_fwd)
                    duration = forward_times[rank]
                    schedules[rank].append(('F', len(fwd_ready[rank]), start_time, duration, start_time+duration))
                    fwd_ready[rank].append(start_time + duration)

                    current_time = schedules[rank][-1][4]
                    next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                    start_time = max(current_time, next_bwd)
                    duration = backward_times[rank]
                    schedules[rank].append(('B', len(bwd_ready[rank]), start_time, duration, start_time+duration))
                    bwd_ready[rank].append(start_time + duration)


        for step in range(pp-1,-1,-1):
            for rank in range(step):
                "B"
                current_time = schedules[rank][-1][4]
                next_bwd = bwd_ready[rank + 1][-1] if rank < pp - 1 else 0
                start_time = max(current_time, next_bwd)
                duration = backward_times[rank]
                schedules[rank].append(('B', len(bwd_ready[rank]), start_time, duration, start_time+duration))
                bwd_ready[rank].append(start_time + duration)

        max_time = max([s[-1][4] for s in schedules])

        # f_b_time = [x+y for x,y in zip(forward_times, backward_times)]

        # idx = np.argmax(f_b_time)
        # bubble = sum(f_b_time)+sum(forward_times[idx+1:])+sum(backward_times[idx+1:]) - (pp-idx)*(f_b_time[idx])
        # all_time = bubble + mbc*f_b_time[idx]
        if draw:
            # 可视化调度图
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = {'F': 'skyblue', 'B': 'salmon'}

            for rank, tasks in enumerate(schedules):
                for task_type, mb, start, duration, end in tasks:
                    ax.barh(y=pp - 1 - rank, width=duration, left=start,
                            height=0.6, color=colors[task_type], edgecolor='black')
                    ax.text(start + duration / 2, pp - 1 - rank, f'{task_type}{mb}',
                            va='center', ha='center', fontsize=9, color='black')

            ax.set_yticks(range(pp))
            ax.set_yticklabels([f"Stage {i}" for i in reversed(range(pp))])
            ax.set_xlabel("Time")
            ax.set_title(f"Corrected 1F1B Pipeline Execution Timeline (pp={pp}, mbc={mbc})")
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            plt.savefig("corrected_1F1B_pipeline.png")

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
    
    def _compute_pp_total_time(self):
        fwd_chunk_time, bwd_chunk_time = self._compute_single_batch_fwd_bwd_time(FIRST_CHUNK)
        forward_times = [fwd_chunk_time]
        backward_times = [bwd_chunk_time]
        has_middle_chunks = self.strategy.pp_size > 2
        has_last_chunk = self.strategy.pp_size > 1
        if has_middle_chunks:
            fwd_chunk_time, bwd_chunk_time = self._compute_single_batch_fwd_bwd_time(MIDDLE_CHUNK)
            forward_times.extend([fwd_chunk_time]*(self.strategy.pp_size - 2))
            backward_times.extend([bwd_chunk_time]*(self.strategy.pp_size - 2))
        if has_last_chunk:
            fwd_chunk_time, bwd_chunk_time = self._compute_single_batch_fwd_bwd_time(LAST_CHUNK)
            forward_times.append(fwd_chunk_time)
            backward_times.append(bwd_chunk_time)

        single_iter_time = self.calculate_1f1b_bubble(self.strategy.pp_size, self.strategy.micro_batch_num, forward_times, backward_times, draw = False)
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
        single_iter_time_no_dp_opim = self._compute_pp_total_time()
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

        def format_chunk_time(model_chunk, chunk_time, duration_time):
            return {
                model_chunk:{
                    'duration_time(chunk_timexmbc+dp_optim+bubble)': duration_time,
                    'chunk_time(fwd+bwd)':chunk_time,
                    'dp_and_optim_time': get_dp_and_optim(model_chunk),
                    'bubble_time': single_iter_time_no_dp_opim - mbc*(chunk_time)
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
        # Fake first stage input

        input_info_first_stage = InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(self.strategy.micro_batch_size, self.strategy.seq_len)
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
                            seq_len,
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
                            seq_len,
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
                    is_etp_valid = (world_size %(ep_size* etp_size) == 0) # TODO(sherry): 临时限定etp_size*ep_size < self.system.num_per_node

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
 