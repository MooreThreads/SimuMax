"""performance model for LLM"""

from abc import ABC, abstractmethod
import os
import json
from copy import deepcopy
from typing import List, Union, Dict
from sympy import divisors
import matplotlib.pyplot as plt
import pandas as pd
from simumax.core.base_struct import PathDebugContext
from simumax.core.config import StrategyConfig, SystemConfig, ModelConfig, TMP_PATH, SIMU_CHECK, SIMU_DEBUG
from simumax.core.base_struct import InputOutputInfo, TensorSize, Result
from simumax.core.transformer.language_model import LLMModel, PeakPoint
from simumax.core.utils import (
    HumanReadableSize,
    human_readable_bytes,
    convert_final_result_to_human_format,
    merge_dict,
    rm_tmp
)


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
            condition = (ep_size*tp_size <= num_gpu_per_nodes) # When tp *ep exceeds the number of nodes, the communication bandwidth will be reduced, and the default communication between machines will be carried out.
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

    def analysis_net(self, re_analysis = False):
        if self.system.intra_with_pcie:
            self.analysis_pcie_net(re_analysis)
        else:
            self.analysis_high_link_net(re_analysis)

    def run_estimate(self):
        assert self.is_configured, "should call configure() first"
        self.model_config.maybe_pad_vocab_size(self.strategy.tp_size)
        self.analysis_net(re_analysis = True)

        self.build()

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

    def _compute_chunk_time(self, model_name='first_stage_chunk'):
        batch_stat_with_recomp = ( 
            self._analysis_single_batch_cost_impl(
                enable_recompute=True, model_name=model_name
            )
        )
        batch_stat_no_recomp = self._analysis_single_batch_cost_impl(
            enable_recompute=False, model_name=model_name
        )
        comm_result = self._analysis_comm_time(
            batch_stat_with_recomp, batch_stat_no_recomp, model_name=model_name
        )
        compute_result = self._analysis_compute_time(
            batch_stat_with_recomp, batch_stat_no_recomp, model_name=model_name
        )
        batch_compute_stat = compute_result[
            "batch_compute_stat"
        ]
        chunk_time = (
            batch_compute_stat["cost_info"]["fwd_compute_time"]
            + batch_compute_stat["cost_info"]["bwd_compute_time"]
            + batch_compute_stat["cost_info"]["recompute_compute_time"]
            + comm_result["intra_comm_time"][
                "intra_exposed_time_per_batch"
            ]
            + comm_result["inter_comm_time"][
                "inter_exposed_time_per_batch"
            ]
            * 2
        )
        return batch_stat_with_recomp, batch_stat_no_recomp, comm_result, compute_result, batch_compute_stat, chunk_time
        
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
    
        def compute_dp_helper(rs_comm_size, gather_comm_size, dp_net, dp_size):
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
                    comm_stage="dp"
                )
                all_gather_time = num_gather_bucket * self.system.compute_net_op_time(
                    "all_gather", bucket_size, comm_num=dp_size, net=dp_net,comm_stage="dp"
                )
                dp_comm_time += all_gather_time + reduce_scatter_time
                details['reduce_scatter_time'] = reduce_scatter_time
                details['all_gather_time'] = all_gather_time
            else:
                dp_comm_time += num_reduce_bucket * self.system.compute_net_op_time(
                    "all_reduce", bucket_size, comm_num=dp_size, net=dp_net, comm_stage="dp"
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

        rs_comm_size = model_info.grad_bytes/2  if self.strategy.grad_reduce_in_bf16 else model_info.grad_bytes 
        gather_comm_size = model_info.grad_bytes / 4 * self.dtype_to_element_size[self.strategy.dtype] 

        moe_rs_comm_size = model_info.moe_grad_bytes / 2 if self.strategy.grad_reduce_in_bf16 else model_info.moe_grad_bytes
        moe_gather_comm_size = model_info.moe_grad_bytes / 4 * self.dtype_to_element_size[self.strategy.dtype]

        dense_dp_result = compute_dp_helper(rs_comm_size, gather_comm_size, self.strategy.dp_net, self.strategy.dp_size)
        moe_dp_result = compute_dp_helper(moe_rs_comm_size, moe_gather_comm_size, self.strategy.dp_net, self.strategy.edp_size) # FIXME: support edp_net decision
        all_result = {
            'dp_comm_exposed_time': dense_dp_result['dp_comm_exposed_time'] + moe_dp_result['dp_comm_exposed_time'],
            'dense': dense_dp_result,
            'moe': moe_dp_result,
        }
        return all_result

    def _analysis_mem_impl(
        self,
        micro_batch_num,
        skip_ckpt_micro_batch_num=None,
        model_name="first_stage_chunk",
    ):
        assert (
            skip_ckpt_micro_batch_num is None
            or skip_ckpt_micro_batch_num <= micro_batch_num
        )
        result = {}
        model_info = self.model_chunk_dict[model_name].get_model_info()
        # act_info = self.model_chunk_dict[model_name].get_act_info()
        # act_info_with_recomp = self.model_chunk_dict[
        #     model_name
        # ].get_act_info_with_recomp()
        if self.strategy.enable_recompute and skip_ckpt_micro_batch_num is None:
            skip_ckpt_micro_batch_num = 0
        elif not self.strategy.enable_recompute:
            skip_ckpt_micro_batch_num = micro_batch_num

        cur_mb_with_recomp = micro_batch_num > skip_ckpt_micro_batch_num
        skip_ckpt_micro_batch_num_prev = (
            skip_ckpt_micro_batch_num
            if cur_mb_with_recomp
            else skip_ckpt_micro_batch_num - 1
        )
        with_recomp_micro_batch_num_prev = (  # pylint: disable=invalid-name
            (micro_batch_num - 1 - skip_ckpt_micro_batch_num)
            if cur_mb_with_recomp
            else 0
        )
        result["micro_batch_num"] = self.strategy.micro_batch_num
        result["micro_batch_size"] = self.strategy.micro_batch_size
        result['parallel_config'] = {
            'parallelism': self.strategy.parallelism,
            'fp8': self.strategy.fp8,
            'recompute_status':{
                'layer_num': self.model_config.layer_num,
                'actual_layer_num': self.model_chunk_dict['first_stage_chunk'].layer_num,
                'recompute_layer': self.strategy.recompute_layer_num,
                'recompute_recompute_granularity': self.strategy.recompute_granularity,
                # 'mlp_recompute_detail':self.strategy.parse_mlp_recompute(0).__dict__(),
                # 'attention_recompute_detail':self.strategy.parse_mlp_recompute(0).__dict__(),
            }
        }
        if self.strategy.grad_reduce_in_bf16:
                model_info.grad_bytes = model_info.grad_bytes/2 # TODO(sherry): this is a hack to make it work, need to fix

        dense_model_mem = dict(
            model_mem = model_info.weight_bytes + model_info.grad_bytes + model_info.state_bytes,
            weight_bytes = model_info.weight_bytes,
            grad_bytes = model_info.grad_bytes,
            state_bytes = model_info.state_bytes
        )
        moe_model_mem = dict(
            model_mem = model_info.moe_weight_bytes + model_info.moe_grad_bytes + model_info.moe_state_bytes,
            weight_bytes = model_info.moe_weight_bytes,
            grad_bytes = model_info.moe_grad_bytes,
            state_bytes = model_info.moe_state_bytes
        )
        result["model_mem"] = dense_model_mem["model_mem"] + moe_model_mem["model_mem"]


        result["model_mem_detail"] = dict(
            dense = dense_model_mem,
            moe = moe_model_mem
        )
        # skip gradient ckpt micro batch
        # result["act_mem_detail"] = repr(act_info)
        result["skip_ckpt_micro_batch_num_prev"] = skip_ckpt_micro_batch_num_prev
        result["with_recomp_micro_batch_num_prev"] = with_recomp_micro_batch_num_prev # micro_batch_num - 1
        result["cur_mb_with_recomp"] = cur_mb_with_recomp


        #-------------------------- 0. get with/no recompute details --------------------------

        cur_no_recompute_act_info:PeakPoint = self.pp_state_peak_point[model_name]["no_recompute_peak_point"]
        cur_with_recompute_act_info:PeakPoint = self.pp_state_peak_point[model_name]["with_recompute_peak_point"]
        cur_act_info:PeakPoint = cur_with_recompute_act_info if cur_mb_with_recomp else cur_no_recompute_act_info

        # result["cur_mbs_no_recompute_act_mem_detail"] = deepcopy(cur_no_recompute_act_info.to_dict())
        # result["cur_mbs_with_recompute_act_mem_detail"] = deepcopy(cur_with_recompute_act_info.to_dict())
        #-------------------------- get with/no recompute details --------------------------


        # result["act_cached_mem_prev_mbs"] = (
        #     skip_ckpt_micro_batch_num_prev * act_info.activation_mem_cache
        #     + with_recomp_micro_batch_num_prev
        #     * act_info_with_recomp.activation_mem_cache
        # ) 
        
        #-------------------------- 1. compute act_cached_mem_prev_mbs --------------------------
        result["fwd_activation_cache_per_micro_batch"] = f"{cur_with_recompute_act_info.activation_mem_cache/1024/1024/1024:.4f} GB"
        result["peak_activation_mem_in_1F1B"] = cur_with_recompute_act_info.peak_mem
        

        result["act_cached_mem_prev_mbs"] = (
            skip_ckpt_micro_batch_num_prev * cur_no_recompute_act_info.activation_mem_cache 
            + with_recomp_micro_batch_num_prev
            * cur_with_recompute_act_info.activation_mem_cache
        ) 

        #-------------------------- compute act_cached_mem_prev_mbs --------------------------


        # cur_act_info = act_info_with_recomp if cur_mb_with_recomp else act_info

        # result["cur_mbs_act_mem_detail"] = deepcopy(act_info.to_dict())
        
        peak_model_mem = result["model_mem"]


        # result["fwd_peak_allocated_mem"] = (
        #     peak_model_mem
        #     + result["act_cached_mem_prev_mbs"]
        #     + cur_act_info.fwd_peak_mem
        # )
        # result["bwd_peak_allocated_mem"] = (
        #     peak_model_mem
        #     + result["act_cached_mem_prev_mbs"]
        #     + cur_act_info.bwd_peak_mem
        # )
        
        #-------------------------- 2. compute fwd/bwd peak mem --------------------------

        result["fwd_peak_allocated_mem"] = (
            peak_model_mem
            + result["act_cached_mem_prev_mbs"]
            + cur_act_info.fwd_peak_mem
        )

        result["bwd_peak_allocated_mem"] = (
            peak_model_mem
            + result["act_cached_mem_prev_mbs"]
            + max(cur_act_info.bwd_peak_mem, cur_act_info.recomp_fwd_peak_mem, cur_act_info.recomp_bwd_peak_mem)
        )
        #-------------------------- compute fwd/bwd peak mem --------------------------

        # result["peak_cached_mem"] = (
        #     max(result["bwd_peak_allocated_mem"], result["fwd_peak_allocated_mem"])
        #     / self.strategy.mem_factor
        # )

        #-------------------------- 3. compute total peak peak mem --------------------------
        result["peak_mem"] = (
            max(result["bwd_peak_allocated_mem"], result["fwd_peak_allocated_mem"])
        )
        result["peak_mem_with_reserved"] = (
            max(result["bwd_peak_allocated_mem"], result["fwd_peak_allocated_mem"])
            / self.strategy.mem_factor
        )
        result["memory_reserved_ratio"] = str(self.strategy.mem_factor)
        result["peak_path"] = f"{cur_with_recompute_act_info.peak_path}, stage=[{cur_with_recompute_act_info.peak_stage}]"
        #-------------------------- compute total peak peak mem --------------------------

        is_first_stage = model_name == "first_stage_chunk"
        debug_points = (
            self.debug_points if is_first_stage else self.debug_points_last_stage
        )
        path_debug_context = (
            self.path_debug_context
            if is_first_stage
            else self.path_debug_context_last_stage
        )

        if debug_points:
            debug_res_info = {}
            for point in debug_points:
                data = path_debug_context.get_point_datas(
                    enable_recompute=cur_mb_with_recomp
                )
                if point not in data:
                    continue
                point_data = data[point]
                point_data.valid_debug_info()
                debug_res_info[point_data.point] = {
                    "prev_cache_mem": human_readable_bytes(point_data.prev_cache_mem),
                    "fwd_peak_no_cache_mem": human_readable_bytes(
                        point_data.fwd_peak_no_cache_mem
                    ),
                    "bwd_peak_no_cache_mem": human_readable_bytes(
                        point_data.bwd_peak_no_cache_mem
                    ),
                    "fwd_peak_allocated_mem": human_readable_bytes(
                        peak_model_mem
                        + result["act_cached_mem_prev_mbs"]
                        + point_data.prev_cache_mem
                        + point_data.fwd_peak_no_cache_mem
                    ),
                    "bwd_peak_allocated_mem": human_readable_bytes(
                        peak_model_mem
                        + result["act_cached_mem_prev_mbs"]
                        + point_data.prev_cache_mem
                        + point_data.bwd_peak_no_cache_mem
                    ),
                }
            result["debug_res_info"] = debug_res_info

        # Convert to human format
        convert_final_result_to_human_format(result)
        return result

    def analysis_mem(self):
        """Based the simulation result, analyze the memory usage"""
        if self.strategy.pp_size == 1:
            result = self._analysis_mem_impl(
                micro_batch_num=1, model_name="first_stage_chunk"
            )
        else:
            result = {"first_stage": {}, "last_stage": {}}
            result["first_stage"] = self._analysis_mem_impl(
                micro_batch_num=self.strategy.pp_size, model_name="first_stage_chunk"
            ) # 这里应该是对应的1F1B, stage1的ac需要hold pp_size份mbs（micro batch size）
            result["last_stage"] = self._analysis_mem_impl(
                micro_batch_num=1, model_name="last_stage_chunk"
            )
        if self.strategy.pp_size>2:
            result["middle_stage"] = self._analysis_mem_impl(
                micro_batch_num=self.strategy.pp_size-1, model_name="middle_stage_chunk"
            )# 这里应该是对应的1F1B, stage2的ac需要hold pp_size-1份mbs（micro batch size）
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

    def _analysis_compute_time(self, batch_stat_with_recomp, batch_stat_no_recomp, model_name):
        result = {}
        micro_batch_num = self.strategy.micro_batch_num
        # skip_ckpt_micro_batch_num = self.strategy.skip_ckpt_micro_batch_num
        if self.strategy.enable_recompute:
            batch_stat = batch_stat_with_recomp
        else:
            batch_stat = batch_stat_no_recomp
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

    def _analysis_comm_time(self, batch_stat_with_recomp, batch_stat_no_recomp, model_name):
        result = {}
        micro_batch_num = self.strategy.micro_batch_num
        dp_comm_result = self._compute_dp_time(model_name)
        # TODO: add ckpt bubble and add strategy extra comm time, # e.g sp grad reduce
        intra_exposed_time_with_recomp_per_batch = sum(  # pylint: disable=invalid-name
            batch_stat_with_recomp["cost_info"][k]
            for k in ["fwd_net_time", "bwd_net_time", "recompute_net_time"]
        )
        intra_exposed_time_no_recomp_per_batch = sum(  # pylint: disable=invalid-name
            batch_stat_no_recomp["cost_info"][k]
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
            inter_exposed_time_per_batch = 2 * self.system.compute_net_op_time(
                "p2p", pp_comm_size, 2, net=self.strategy.pp_net, comm_stage="pp"
            )  # 2 p2p

        else:
            inter_exposed_time_per_batch = 0
        # intra_exposed_time_with_recomp = intra_exposed_time_with_recomp_per_batch
        # intra_exposed_time_with_recomp = intra_exposed_time_with_recomp_per_batch * \
        #                                  micro_batch_num
        # intra_exposed_time_no_recomp = intra_exposed_time_no_recomp_per_batch * micro_batch_num
        inter_exposed_time = inter_exposed_time_per_batch * micro_batch_num
        result["dp_comm_time"] = dp_comm_result
        # Now we don't consider the mix of recompute and non-recompute
        if self.strategy.enable_recompute:
            intra_exposed_time_per_batch = intra_exposed_time_with_recomp_per_batch
            intra_exposed_time = (
                intra_exposed_time_with_recomp_per_batch * micro_batch_num
            )
        else:
            intra_exposed_time_per_batch = intra_exposed_time_no_recomp_per_batch
            intra_exposed_time = (
                intra_exposed_time_no_recomp_per_batch * micro_batch_num
            )
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

    def _analysis_single_iter_cost_impl2(self):
        FIRST_CHUNK = "first_stage_chunk"
        MIDDLE_CHUNK = "middle_stage_chunk"
        LAST_CHUNK = "last_stage_chunk"

        def compute_single_iter_cost_helper(model_name = "first_stage_chunk"):
            assert model_name in ["first_stage_chunk", "middle_stage_chunk", 'last_stage_chunk'], f"model_name {model_name} not supported"
            batch_stat_with_recomp = self._analysis_single_batch_cost_impl(
                enable_recompute=True, model_name=model_name
            )
            batch_stat_no_recomp = self._analysis_single_batch_cost_impl(
                enable_recompute=False, model_name=model_name
            )
            # 1.comm_result： dp_time + fwd/bwd/recompute net time + pp_time
            comm_cost_result = self._analysis_comm_time(
                batch_stat_with_recomp, batch_stat_no_recomp, model_name
            )
             # 2.compute result: 
            # - fwd/bwd/recompute compute time, stored in compute_result['cost_info'], summarized in compute_result['fwd/bwd/recompute_compute_time']. compute_result['cost_info']['dp_comm_time] is computed by model.grad_bytes.
            # - optim update time, stored in compute_result['optim_time'], computed by model.state_bytes.
            # - fwd/bwd/recompute fwd/bwd/recompute net time is ignored in compute time, but included in comm_result['intra_comm_time']
            compute_cost_result = self._analysis_compute_time(
                batch_stat_with_recomp, batch_stat_no_recomp
            )
          
            breakdown_result = {}
            breakdown_result["fwd_compute_time"] = compute_cost_result["fwd_compute_time"]
            breakdown_result["recompute_time"] = compute_cost_result["recompute_time"]
            breakdown_result["bwd_compute_time"] = compute_cost_result["bwd_compute_time"]
            breakdown_result["optim_time"] = compute_cost_result["optim_time"][
                "optim_exposed_time"
            ]
            # breakdown_result['grad_accumulation'] = self.strategy.micro_batch_num * self.system.compute_mem_access_time(self.model_chunk_dict[model_name]._model_info.grad_bytes)
            breakdown_result["intra_exposed_time"] = comm_cost_result["intra_comm_time"][
                "intra_exposed_time"
            ]
            breakdown_result["inter_exposed_time"] = comm_cost_result["inter_comm_time"][
                "inter_exposed_time"
            ]
            breakdown_result["dp_exposed_time"] = comm_cost_result["dp_comm_time"][
                "dp_comm_exposed_time"
            ]
            breakdown_result['flexible_time'] = 20 # traning log/h2d

            cost_info = self.model_chunk_dict[model_name]._cost_info
            chunk_time = (
                cost_info.fwd_compute_time
                + cost_info.bwd_compute_time
                + cost_info.recompute_compute_time
                + comm_cost_result["intra_comm_time"]["intra_exposed_time_per_batch"]
                + comm_cost_result["inter_comm_time"]["inter_exposed_time_per_batch"] * 2
            )
            breakdown_result['bubble_time'] = self._compute_bubble_time(chunk_time)

            all_tokens_per_iter = self.strategy.seq_len * self.strategy.global_batch_size

            all_result = {}
            all_result["comm_cost_details"] = comm_cost_result
            all_result["compute_cost_details"] = compute_cost_result
            all_result["breakdown_result"] = breakdown_result
            all_result['micro_batch_num']  = self.strategy.micro_batch_num
            # all_result["1F1B_time(=breakdown_result_time remove dp/optim/bubble time)"] = chunk_time
            all_result["all_tokens_per_iter"] = all_tokens_per_iter

            return  all_result
        
        # def compute_pp_total_time_helper():
            # def compute_fwd_bwd_time(model_name):
            #     # if len(compute_result) == 0:
            #     #     return -1, -1
            #     # chunk_time = (
            #     #     batch_compute_stat["cost_info"]["fwd_compute_time"]
            #     #     + batch_compute_stat["cost_info"]["bwd_compute_time"]
            #     #     + batch_compute_stat["cost_info"]["recompute_compute_time"]
            #     #     + comm_result["intra_comm_time"]["intra_exposed_time_per_batch"]
            #     #     + comm_result["inter_comm_time"]["inter_exposed_time_per_batch"] * 2
            #     # )
            #     if self.strategy.pp_size > 1:
            #         pp_comm_size = (
            #             self.micro_hidden_states_size
            #             * self.dtype_to_element_size[self.strategy.dtype]
            #         )
            #         pp_comm_size = (
            #             pp_comm_size / self.strategy.tp_size
            #             if self.strategy.enable_sequence_parallel
            #             else pp_comm_size
            #         )
            #         pp_time = 2 * self.system.compute_net_op_time(
            #             "p2p", pp_comm_size, 2, net=self.strategy.pp_net
            #         )  # 2 p2p

            #     else:
            #         pp_time = 0
        
            #     cost_info = self.model_chunk_dict[model_name].get_cost_info()
                
            #     fwd_chunk_time = (cost_info.fwd_compute_time + 
            #                       cost_info.fwd_net_time + 
            #                       pp_time)

            #     bwd_chunk_time = (cost_info.bwd_compute_time + 
            #                       cost_info.bwd_net_time + 
            #                       cost_info.recompute_compute_time + 
            #                       cost_info.recompute_net_time + 
            #                       pp_time)
    
            #     # fwd_time = (cost_info['fwd_compute_time'] + 
            #     #                 cost_info['fwd_net_exposed_time']
            #     #              )
            #     # bwd_time = (cost_info['bwd_compute_time'] + 
            #     #             cost_info['bwd_net_exposed_time'] + 
            #     #             cost_info['recompute_compute_time'] +
            #     #             cost_info['recompute_net_exposed_time']
            #     #             )
            #     return fwd_chunk_time, bwd_chunk_time
            # fwd_chunk_time, bwd_chunk_time = compute_fwd_bwd_time(FIRST_CHUNK)
            # forward_times = [fwd_chunk_time]
            # backward_times = [bwd_chunk_time]
            # has_middle_chunks = self.strategy.pp_size > 2
            # has_last_chunk = self.strategy.pp_size > 1
            # if has_middle_chunks:
            #     fwd_chunk_time, bwd_chunk_time = compute_fwd_bwd_time(MIDDLE_CHUNK)
            #     forward_times.extend([fwd_chunk_time]*(self.strategy.pp_size - 2))
            #     backward_times.extend([bwd_chunk_time]*(self.strategy.pp_size - 2))
            # if has_last_chunk:
            #     fwd_chunk_time, bwd_chunk_time = compute_fwd_bwd_time(LAST_CHUNK)
            #     forward_times.append(fwd_chunk_time)
            #     backward_times.append(bwd_chunk_time)
   
            # single_iter_time = self.calculate_1f1b_bubble(self.strategy.pp_size, self.strategy.micro_batch_num, forward_times, backward_times)
            # single_iter_time +=  self._compute_dp_time()['dp_comm_exposed_time']
            # single_iter_time += self._compute_optim_time()['optim_exposed_time']
            # return single_iter_time
            
        def compute_mfu_helper(all_result, duration_time_per_iter):
            # breakdown_result = all_result['breakdown_result']
            # breakdown_result['bubble_time'] = bubble_time
            # breakdown_result['uneven_bubble_time'] = uneven_bubble_time
            duration_time_per_iter = duration_time_per_iter
            # all_result['bubble_time'] = bubble_time 
            # all_result['uneven_bubble_time'] = uneven_bubble_time
            all_result['duration_time_per_iter'] = duration_time_per_iter
            # TODO(sherry): add bubble time 

            chunk_time = sum(all_result['breakdown_result'].values()) 
            all_result['breakdown_result']['bubble_time'] = duration_time_per_iter - chunk_time

            throughput_per_accelerator = (
                all_result["all_tokens_per_iter"]
                / (duration_time_per_iter / 1000)
                / self.strategy.world_size
            )
            all_result["throughput_per_accelerator"] = throughput_per_accelerator
            all_result["mfu_6nd_with_attn"] = (
                self.model_config.flops_per_token(
                    context_seq_len=self.strategy.seq_len, with_attn=True
                )
                * throughput_per_accelerator
                / self.system.accelerator.op["default"].tflops
                / 1e12 # convert byte-flops to tflops
            )

            compute_cost_result = all_result['compute_cost_details']
            mfu = (
                compute_cost_result["model_flops"]
                / (duration_time_per_iter / 1000) # convert ms to s
                / self.system.accelerator.op["default"].tflops
                / 1e12 # convert byte-flops to tflops
            )
            TFLOPs = (compute_cost_result["model_flops"]
                / (duration_time_per_iter / 1000) # convert ms to s
                / 1e12
                )

            all_result["mfu"] = mfu
            all_result['model_flops'] = compute_cost_result["model_flops"]
            all_result["TFLOPs"] = TFLOPs 
            return mfu
        
        res = {
            FIRST_CHUNK: {},
            MIDDLE_CHUNK: {},
            LAST_CHUNK: {},
        }
        

        res[FIRST_CHUNK] = compute_single_iter_cost_helper(model_name=FIRST_CHUNK)
        if self.strategy.pp_size > 2:
            res[MIDDLE_CHUNK] = compute_single_iter_cost_helper(model_name=MIDDLE_CHUNK)

        if self.strategy.pp_size > 1:
            res[LAST_CHUNK] = compute_single_iter_cost_helper(model_name=LAST_CHUNK)
        
        single_iter_time = sum(v for _, v in res[FIRST_CHUNK]['breakdown_result'].items())

        mfu = 0
        TFLOPs = 0
        mfu += compute_mfu_helper(res[FIRST_CHUNK], single_iter_time)
        TFLOPs += res[FIRST_CHUNK]["TFLOPs"]
        if self.strategy.pp_size > 1:
            mfu += compute_mfu_helper(res[LAST_CHUNK], single_iter_time)
            TFLOPs += res[LAST_CHUNK]["TFLOPs"]
        if self.strategy.pp_size > 2:
            mfu += compute_mfu_helper(res[MIDDLE_CHUNK], single_iter_time) * (self.strategy.pp_size -2)
            TFLOPs += res[MIDDLE_CHUNK]["TFLOPs"] * (self.strategy.pp_size -2)

        
        mfu /= self.strategy.pp_size
        TFLOPs /= self.strategy.pp_size

        res['mfu'] = mfu
        res['mfu_6nd_with_attn'] = res[FIRST_CHUNK]['mfu_6nd_with_attn']
        res['TFLOPs'] = TFLOPs
        res['duration_time_per_iter'] = single_iter_time
        
        convert_final_result_to_human_format(res)
        return res 
    
    def _analysis_single_iter_cost_impl(self):
        # we construct the result in the following hierarchy:
        # first level: useful FlopS、mfu、all FlopS、throughput、duration_per_iter
        # second level: time break down = compute time + comm_time + bubble_time
        # third level-0:  compute time = fwd time + recom_time + bwd_time + optim update time
        # third level-1:  comm_time_: tp_time(tp_time、tp_time_can_overlap) + pp_time
        all_result = {}
        batch_stat_with_recomp = self._analysis_single_batch_cost_impl(
            enable_recompute=True, model_name = "first_stage_chunk"
        )
        batch_stat_no_recomp = self._analysis_single_batch_cost_impl(
            enable_recompute=False, model_name = "first_stage_chunk"
        )
        # 1.comm_result： dp_time + fwd/bwd/recompute net time + pp_time
        comm_result = self._analysis_comm_time(
            batch_stat_with_recomp, batch_stat_no_recomp, model_name = "first_stage_chunk"
        )
        # 2.compute result: 
        # - fwd/bwd/recompute compute time, stored in compute_result['cost_info'], summarized in compute_result['fwd/bwd/recompute_compute_time']. compute_result['cost_info']['dp_comm_time] is computed by model.grad_bytes.
        # - optim update time, stored in compute_result['optim_time'], computed by model.state_bytes.
        # - fwd/bwd/recompute fwd/bwd/recompute net time is ignored in compute time, but included in comm_result['intra_comm_time']
        compute_result_first_stage = self._analysis_compute_time(
            batch_stat_with_recomp, batch_stat_no_recomp, model_name = "first_stage_chunk"
        )
        # 3. all time
        batch_compute_stat = compute_result_first_stage["batch_compute_stat"]
        bubble_uneven_time = 0

        # can't be overlap for now
        # first_stage chunk time
        chunk_time = (
            batch_compute_stat["cost_info"]["fwd_compute_time"]
            + batch_compute_stat["cost_info"]["bwd_compute_time"]
            + batch_compute_stat["cost_info"]["recompute_compute_time"]
            + comm_result["intra_comm_time"]["intra_exposed_time_per_batch"]
            + comm_result["inter_comm_time"]["inter_exposed_time_per_batch"] * 2
        )
        if self.strategy.pp_size > 1:
            batch_stat_with_recomp_last_stage = (  # pylint: disable=invalid-name
                self._analysis_single_batch_cost_impl(
                    enable_recompute=True, model_name="last_stage_chunk"
                )
            )
            batch_stat_no_recomp_last_stage = self._analysis_single_batch_cost_impl(
                enable_recompute=False, model_name="last_stage_chunk"
            )
            comm_result_last_stage = self._analysis_comm_time(
                batch_stat_with_recomp_last_stage, batch_stat_no_recomp_last_stage, model_name="last_stage_chunk"
            )
            compute_result_last_stage = self._analysis_compute_time(
                batch_stat_with_recomp_last_stage, batch_stat_no_recomp_last_stage, model_name="last_stage_chunk"
            )
            batch_compute_stat_last_stage = compute_result_last_stage[
                "batch_compute_stat"
            ]
            chunk_time_last_stage = (
                batch_compute_stat_last_stage["cost_info"]["fwd_compute_time"]
                + batch_compute_stat_last_stage["cost_info"]["bwd_compute_time"]
                + batch_compute_stat_last_stage["cost_info"]["recompute_compute_time"]
                + comm_result_last_stage["intra_comm_time"][
                    "intra_exposed_time_per_batch"
                ]
                + comm_result_last_stage["inter_comm_time"][
                    "inter_exposed_time_per_batch"
                ]
                * 2
            )
            bubble_uneven_time = (
                abs((chunk_time_last_stage - chunk_time))
                * self.strategy.micro_batch_num
            )

        bubble_time = self._compute_bubble_time(chunk_time) # bubble_time = chunk_time * (self.strategy.pp_size - 1)
        
        breakdown_result = {}
        breakdown_result["fwd_compute_time"] = compute_result_first_stage["fwd_compute_time"]
        breakdown_result["recompute_time"] = compute_result_first_stage["recompute_time"]
        breakdown_result["bwd_compute_time"] = compute_result_first_stage["bwd_compute_time"]
        breakdown_result["optim_time"] = compute_result_first_stage["optim_time"][
            "optim_exposed_time"
        ]
        breakdown_result["intra_exposed_time"] = comm_result["intra_comm_time"][
            "intra_exposed_time"
        ]
        breakdown_result["inter_exposed_time"] = comm_result["inter_comm_time"][
            "inter_exposed_time"
        ]
        breakdown_result["dp_exposed_time"] = comm_result["dp_comm_time"][
            "dp_comm_exposed_time"
        ]
        breakdown_result["bubble_time"] = bubble_time
        breakdown_result["bubble_uneven_time"] = bubble_uneven_time
        all_time = sum(v for _, v in breakdown_result.items())
        if self.strategy.pp_size > 1:
            breakdown_result_last_stage = {}
            breakdown_result_last_stage["fwd_compute_time"] = compute_result_last_stage["fwd_compute_time"]
            breakdown_result_last_stage["recompute_time"] = compute_result_last_stage["recompute_time"]
            breakdown_result_last_stage["bwd_compute_time"] = compute_result_last_stage["bwd_compute_time"]
            breakdown_result_last_stage["optim_time"] = compute_result_last_stage["optim_time"][
                "optim_exposed_time"
            ]
            breakdown_result_last_stage["intra_exposed_time"] = comm_result_last_stage["intra_comm_time"][
                "intra_exposed_time"
            ]
            breakdown_result_last_stage["inter_exposed_time"] = comm_result_last_stage["inter_comm_time"][
                "inter_exposed_time"
            ]
            breakdown_result_last_stage["dp_exposed_time"] = comm_result_last_stage["dp_comm_time"][
                "dp_comm_exposed_time"
            ]
            
            if self.strategy.pp_size > 2:
                chunk_time_middle_stage = self._compute_chunk_time(model_name='middle_stage_chunk')[-1]
            else:
                chunk_time_middle_stage = 0
            
            
            bubble_time_last_stage = chunk_time + chunk_time_middle_stage*(self.strategy.pp_size-2)
            breakdown_result_last_stage["bubble_time"] = bubble_time_last_stage
            all_time_last_stage = sum(v for _, v in breakdown_result_last_stage.items())
            breakdown_result_last_stage['all_time'] = all_time_last_stage
            all_result["breakdown_result_last_stage"] = breakdown_result_last_stage
            # all_time = all_time_last_stage # FIXME(sherry)：使用last stage的all_time
        # 4.compute first level
        all_tokens_per_iter = self.strategy.seq_len * self.strategy.global_batch_size
        duration_time_per_iter = all_time # TODO(sherry)：使用all time进行后续所有stage的mfu计算
        model_flops = compute_result_first_stage["model_flops"]
        throughput_per_accelerator = (
            all_tokens_per_iter
            / (duration_time_per_iter / 1000)
            / self.strategy.world_size
        )
        mfu = (
            model_flops
            / (duration_time_per_iter / 1000) # convert ms to s
            / self.system.accelerator.op["default"].tflops
            / 1e12 # convert byte-flops to tflops
        )
        TFLOPS = (model_flops
            / (duration_time_per_iter / 1000) # convert ms to s
            / 1e12
            )

        theory_flops = self.model_config.flops_per_token(context_seq_len=self.strategy.seq_len, with_attn=True) * all_tokens_per_iter //  self.strategy.world_size
        all_result["comm_details"] = comm_result
        all_result["compute_details"] = compute_result_first_stage
        all_result["breakdown_result"] = breakdown_result
        all_result["chunk_time"] = chunk_time
        all_result["all_tokens_per_iter"] = all_tokens_per_iter
        all_result["duration_time_per_iter"] = duration_time_per_iter
        all_result["mfu_6nd_with_attn"] = (
            self.model_config.flops_per_token(
                context_seq_len=self.strategy.seq_len, with_attn=True
            )
            * throughput_per_accelerator
            / self.system.accelerator.op["default"].tflops
            / 1e12 # convert byte-flops to tflops
        )
        if self.strategy.pp_size > 1:
            mfu_last_stage = (
                compute_result_last_stage["model_flops"]
                / (duration_time_per_iter / 1000)
                / self.system.accelerator.op["default"].tflops
                / 1e12
            )
            mfu += mfu_last_stage
            tflops_last_stage =(
                compute_result_last_stage["model_flops"]
                / (duration_time_per_iter / 1000)
                / 1e12
            )
            TFLOPS += tflops_last_stage
        if self.strategy.pp_size > 2:
            batch_stat_with_recomp_middle_stage = (  
                self._analysis_single_batch_cost_impl(
                    enable_recompute=True, model_name="middle_stage_chunk"
                )
            )
            batch_stat_no_recomp_middle_stage = self._analysis_single_batch_cost_impl(
                enable_recompute=False, model_name="middle_stage_chunk"
            )
            compute_result_middle_stage = self._analysis_compute_time(
                batch_stat_with_recomp_middle_stage, batch_stat_no_recomp_middle_stage, model_name="middle_stage_chunk"
            )
            mfu_middle_stage = (
                compute_result_middle_stage["model_flops"]
                / (duration_time_per_iter / 1000)
                / self.system.accelerator.op["default"].tflops
                / 1e12
            )
            mfu += mfu_middle_stage*(self.strategy.pp_size-2)
            tflops_middle_stage = (
                compute_result_middle_stage["model_flops"]
                / (duration_time_per_iter / 1000)
                / 1e12
            )
            TFLOPS += tflops_middle_stage * (self.strategy.pp_size - 2)
        mfu /= self.strategy.pp_size
        TFLOPS /= self.strategy.pp_size
        # mfu_6nd_with_attn: The MFU is calculated based on the FLOPS of standard attention
        # mfu: The MFU is calculated based on the actual FLOPS of attention. If it is flashattention, it will have one more base flops than standard attention
        all_result["mfu"] = mfu

        all_result["throughput_per_accelerator"] = throughput_per_accelerator
        all_result["throughput per GPU (TFLOP/s/GPU)"] = theory_flops / (duration_time_per_iter/1000)/1e12
        all_result['flops_info'] = {
            'theory_flops': theory_flops,
            'model_flops': model_flops,
        }
        
        # new_result = self._analysis_single_iter_cost_impl2()
        # all_result['new_duration_time_per_iter'] = new_result['duration_time_per_iter']
        # all_result['sherry'] = new_result

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
        _ = self.model_chunk_dict["first_stage_chunk"](
            input_info_first_stage, self.path_debug_context
        )
        self.pp_state_peak_point["first_stage_chunk"] = self.model_chunk_dict["first_stage_chunk"].compute_activations()
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
            _ = self.model_chunk_dict["middle_stage_chunk"](
                input_info_last_stage, self.path_debug_context_last_stage
            )    
            self.pp_state_peak_point["middle_stage_chunk"] = self.model_chunk_dict["middle_stage_chunk"].compute_activations()
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
            _ = self.model_chunk_dict["last_stage_chunk"](
                input_info_last_stage, self.path_debug_context_last_stage
            )    
            self.pp_state_peak_point["last_stage_chunk"] = self.model_chunk_dict["last_stage_chunk"].compute_activations()

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
           
            self.strategy.micro_batch_num = micro_batch_num # TODO(sherry): 固定global_batch_size  
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

    def search_selective_recompute(self, global_batch_size,  use_reserved_memory):
        # self.search_selective_recompute(global_batch_size=global_batch_size, use_reserved_memory=True, search_best = True, best_mfu=cur_best_mfu, all_search_result=all_search_result)
        """
        Fixed global_batch_size and search for all (selective recompute, micro_batch_size, micro_batch_count) combinations of MFU under the current parallel strategy. 

        :param global_batch_size: global batch size
        :param use_reserved_memory: whether to use reserved memory to avoid OOM
        """

        self.strategy.recompute_granularity = "selective_recompute"
        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        from itertools import product
        all_search_strategy = {}
        params = ['attn_recompute', 'mla_rms_recompute', 'mlp_recompute', 'mlp_rms_recompute']
        combinations = [dict(zip(params, combo)) for combo in product([False, True], repeat=4)]
        for pp_size in [1, 2]:
            for recompute_params in combinations:
                if  recompute_params['mla_rms_recompute'] and not recompute_params['attn_recompute']:
                    continue
                if  recompute_params['mlp_rms_recompute'] and not recompute_params['mlp_recompute']:
                    continue
                self.strategy.pp_size = pp_size
                self.strategy.attn_recompute = recompute_params['attn_recompute']
                self.strategy.mla_rms_recompute = recompute_params['mla_rms_recompute']
                self.strategy.mlp_recompute = recompute_params['mlp_recompute']
                self.strategy.mlp_rms_recompute = recompute_params['mlp_rms_recompute']
                rm_tmp()
                all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list = self.search_max_micro_batch_size_fixed_gbs(pp_size, self.strategy.dp_size, global_batch_size, use_reserved_memory=True, save_all=True)
                
                if len(all_search_micro_batch_size) > 0:
                    for search_micro_batch_size, search_micro_batch_num, peak_mem_list, cost_result  in zip(all_search_micro_batch_size, all_search_micro_batch_num, all_peak_cached_mem_list, all_cost_list):
                        print("find!", peak_mem_list)
                        best_strategy = {}
                        best_strategy['tp_size'] = self.strategy.tp_size
                        best_strategy['ep_size'] = self.strategy.ep_size
                        best_strategy['pp_size'] = self.strategy.pp_size
                        best_strategy['dp_size'] = self.strategy.dp_size
                        best_strategy['micro_batch_size'] = search_micro_batch_size
                        best_strategy['micro_batch_num'] = search_micro_batch_num
                        best_strategy.update(recompute_params)
                        best_strategy['best_mfu'] = cost_result.data['mfu']
                        best_strategy['best_TFLOPs'] = cost_result.data['throughput per GPU (TFLOP/s/GPU)']
                        best_strategy[PEAK_MEM_KEY] = deepcopy(peak_mem_list)
                        best_strategy['bw'] = deepcopy(self.system.real_comm_bw)
                        merge_dict(best_strategy, all_search_strategy)
        return all_search_strategy

    def dump_paralism_and_recompute_perf(self, mem_result, cost_result):
        # from pprint import pprint
        # pprint(mem_result.data)
        perf = {
            'model_name': self.model_config.model_name,
            'system': self.system.sys_name,
            'parallelism': self.strategy.parallelism,
            'recompute_status': self.strategy.recompute_status,
            'TGS_per_gpu' : cost_result.data['throughput_per_accelerator'],
            'TFLOPS': cost_result.data['throughput per GPU (TFLOP/s/GPU)'],
            'mfu': cost_result.data["mfu"],
            'iter_time':  cost_result.data["duration_time_per_iter"],
            'peak_mem':  mem_result.data["peak_mem"] if "peak_mem" in mem_result.data else {s:v['peak_mem'] for s,v in mem_result.data.items()}
        }
        return perf

    def dump_best_strategy(self, best_strategy):
        # pprint(best_strategy)
        perf = {
        }
        return perf
    
    def search_best_selective_recompute(self, use_reserved_memory, gmi_error, best_mfu=None, all_search_result = None):
        self.strategy.recompute_granularity = "selective_recompute"
        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        from itertools import product
        best_strategy = {}
        params = ['attn_recompute', 'mla_rms_recompute', 'mlp_recompute', 'mlp_rms_recompute']
        combinations = [dict(zip(params, combo)) for combo in product([False, True], repeat=4)]
        for recompute_params in combinations:
                if  recompute_params['mla_rms_recompute'] and not recompute_params['attn_recompute']:
                    continue
                if  recompute_params['mlp_rms_recompute'] and not recompute_params['mlp_recompute']:
                    continue
                self.strategy.attn_recompute = recompute_params['attn_recompute']
                self.strategy.mla_rms_recompute = recompute_params['mla_rms_recompute']
                self.strategy.mlp_recompute = recompute_params['mlp_recompute']
                self.strategy.mlp_rms_recompute = recompute_params['mlp_rms_recompute']
                self.run_estimate()
                mem_result = self.analysis_mem()
                cost_result = self.analysis_cost()
                peak_mem = self.get_pp_stage_peak_mem(mem_result, PEAK_MEM_KEY, True)
                peak_mem = max(peak_mem.values())
                if peak_mem + gmi_error <= self.system.accelerator.mem_gbs:
                    cur_perf = self.dump_paralism_and_recompute_perf(mem_result, cost_result)
                    if cur_perf['mfu'] > best_mfu:
                        best_mfu = cur_perf['mfu']
                        best_strategy = cur_perf
                    if all_search_result is not None:
                        merge_dict(cur_perf, all_search_result)
        return best_strategy

    def search_best_full_recompute_layer_num(self, 
                                        layer_num, 
                                        use_reserved_memory: bool, 
                                        gmi_error:int,
                                        best_mfu,
                                        all_search_result:dict):
        """
         Searches for the number of full recompute layers of the highest MFU that can be placed in memory under the current micro_batch_size, micro_batch_count, and parallel policies. 

        Args:
            layer_num (int): layer number
            use_reserved_memory (bool): whether to use reserved memory
            gmi_error (int): The error between gmi and the actual allocated storage, the current s5000 is 6GB.
            best_mfu (float): best mfu
            all_search_result (dict): all search result
        Returns:
            dict: search result
        """
        
        accelerator_mem_gbytes = self.system.accelerator.mem_gbs  - gmi_error
          # gmi has 6 GB error

        self.strategy.recompute_granularity = "full_block"
        PEAK_MEM_KEY = "peak_mem_with_reserved" if use_reserved_memory else "peak_mem"
        best_strategy = dict()
        left, right = 0, layer_num // self.strategy.pp_size -1
        # right = min(right, layer_num-1)
        ori_recompute_layer_num = self.strategy.recompute_layer_num 

        while left <= right:
            recompute_layer_num = (left + right) // 2  
            assert recompute_layer_num < (layer_num // self.strategy.pp_size), f'recompute_layer_num: {recompute_layer_num}, layer_num: {layer_num}, pp_size: {self.strategy.pp_size}'
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
                right = recompute_layer_num -1
                
                # Save best search results
                if cost_result.data['mfu'] >= best_mfu:
                    best_mfu = cost_result.data['mfu']
                    best_strategy['model_name'] = self.model_config.model_name
                    best_strategy['system'] = self.system.sys_name
                    best_strategy["num_gpus_per_node"] = self.system.num_per_node
                    best_strategy["memory_capacity"] = f"{self.system.accelerator.mem_gbs} GB"
                    best_strategy["intra_comm_bw"] = f"{self.system.networks['high_intra_node'].bandwidth.gbps} GB/s"
                    best_strategy["inter_comm_bw"] = f"{self.system.networks['inter_node'].bandwidth.gbps} GB/s"
                    best_strategy["world_size"] = self.strategy.world_size
                    best_strategy['tp_size'] = self.strategy.tp_size
                    best_strategy['ep_size'] = self.strategy.ep_size
                    best_strategy['pp_size'] = self.strategy.pp_size
                    best_strategy['dp_size'] = self.strategy.dp_size
                    best_strategy['edp_size'] = self.strategy.edp_size
                    best_strategy['etp_size'] = self.strategy.etp_size
                    best_strategy['micro_batch_size'] = self.strategy.micro_batch_size
                    best_strategy['micro_batch_num'] = self.strategy.micro_batch_num
                    best_strategy['recompute_layer_num'] = recompute_layer_num

                    best_strategy['best_mfu'] = cost_result.data['mfu']
                    best_strategy['best_TFLOPs'] = cost_result.data['throughput per GPU (TFLOP/s/GPU)']
                    best_strategy['TGS_per_gpu'] = cost_result.data['throughput_per_accelerator']
                    best_strategy[PEAK_MEM_KEY] = str(deepcopy(peak_mem_list))
                    best_strategy['comm_bw_info'] = str(deepcopy(self.system.real_comm_bw))
                    best_strategy['estimate_details'] = {
                        'mem_result': str(mem_result),
                        'compute_result': str(cost_result),
                        'model_arch':str(self.model_chunk_dict),
                        'strategy_config': str(self.strategy),
                        'system_config': str(self.system),
                        'model_config': str(self.model_config)
                    }
                    print(f"Find result  parallelism={self.strategy.parallelism}, recompute={self.strategy.recompute_status},mfu={cost_result.data['mfu']} gbs={self.strategy.global_batch_size} peak_cached_mem_bytes={peak_cached_mem_gbytes}GB")

                if all_search_result is not None:
                    """
                    cur_serach_result = dict()
                    cur_serach_result["model_name"] = model_name    
                    cur_serach_result["arch"] = arch_name
                    cur_serach_result["num_gpus_per_node"] = self.system.num_per_node
                    cur_serach_result["memory_capacity"] = f"{self.system.accelerator.mem_gbs} GB"
                    cur_serach_result["intra_comm_bw"] = f"{self.system.networks['high_intra_node'].bandwidth.gbps} GB/s"
                    cur_serach_result["inter_comm_bw"] = f"{self.system.networks['inter_node'].bandwidth.gbps} GB/s"
                    cur_serach_result["world_size"] = self.strategy.world_size
                    cur_serach_result["data_type"] = "fp8" if self.strategy.fp8 else "bf16"
                    cur_serach_result["tp_size"] = self.strategy.tp_size
                    cur_serach_result["ep_size"] = self.strategy.ep_size
                    cur_serach_result["pp_size"] = self.strategy.pp_size
                    cur_serach_result["dp_size"] = self.strategy.dp_size
                    cur_serach_result['edp_size'] = self.strategy.edp_size
                    cur_serach_result['etp_size'] = self.strategy.etp_size
                    cur_serach_result["micro_batch_size"] = self.strategy.micro_batch_size
                    cur_serach_result["micro_batch_num"] = self.strategy.micro_batch_num
                    cur_serach_result["gbs"] = self.strategy.dp_size * self.strategy.micro_batch_size * self.strategy.micro_batch_num
                    cur_serach_result["recompute_layer_num"] = recompute_layer_num
                    cur_serach_result["layer_num"] = layer_num
                    cur_serach_result["mfu"] = cost_result.data['mfu']
                    cur_serach_result['TFLOPs'] = cost_result.data['throughput per GPU (TFLOP/s/GPU)']
                    cur_serach_result['comm_bw_info'] = deepcopy(self.system.real_comm_bw)
                    cur_serach_result[PEAK_MEM_KEY] = peak_mem_list
                    """
                    perf = self.dump_paralism_and_recompute_perf(mem_result, cost_result)
                    merge_dict(perf, all_search_result)

        self.strategy.recompute_layer_num = ori_recompute_layer_num # recompute_layer_num

        return best_strategy
    
    def search_best_strategy_with_full_recompute(self,
                                                 world_size:int,  
                                                 gmi_error:int,
                                                 micro_batch_size:int,
                                                 global_batch_size:int, 
                                                 all_search_result:dict,
                                                 tp_search_list:List = None,
                                                 ep_search_list:List = None,
                                                 pp_search_list:List = None,
                                                 use_etp:bool = False,
                                                 recompute:str = 'full_block',
                                                 dump_path:str=None,
                                                 dump_best_strategy_details:bool=False):
        """
        Searches for the optimal combination of parallel strategies (tp/ep/pp) and full recompute layer configuration that maximizes performance under fixed global batch size constraints.

        Args:
            world_size (int): world size
            gmi_error (int): The error between gmi and the actual allocated storage, the current s5000 is 6GB.
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
        layer_num = self.model_config.layer_num
        if tp_search_list is None:
            tp_search_list = [1, 2, 4, 8]  if self.model_config.model_type == "dense" else [1]
        if  ep_search_list is None:
            ep_search_list = [1, 2, 4, 8] if self.model_config.model_type == "moe" else [1]
        if pp_search_list is None:
            pp_search_list = list(range(1, layer_num+1))    

        global_best_strategy = {}

        print(f"Start search strategy for world_size={world_size}, model_type={self.model_config.model_type}, model_name={self.model_config.model_name}, system={self.system.sys_name}")
        print(f"- tp_search_list={tp_search_list}, ep_search_list={ep_search_list}, pp_search_list={pp_search_list}")
        print(f"- layer_num={layer_num}")
        print(f"- moe_pad_expert_input_to_capacity={self.model_config.moe_pad_expert_input_to_capacity}")
        print(f"- capacity={self.model_config.capacity}")

        def distribute(layer_num, pp_size):
            if pp_size > layer_num:
                return []
            quotient = layer_num // pp_size
            remainder = layer_num % pp_size
            if remainder == 0:
                return [[quotient] * pp_size]
            else:
                first = [quotient + remainder] + [quotient] * (pp_size - 1)
                last = [quotient] * (pp_size - 1) + [quotient + remainder]
                if remainder % 2 == 0:
                    split = [quotient + remainder // 2] + [quotient] * (pp_size - 2) + [quotient + remainder // 2]
                else:
                    split = [quotient + remainder // 2 + 1] + [quotient] * (pp_size - 2) + [quotient + remainder // 2]
                return [first, last, split]
        
        global_best_mfu = -1
        for tp_size in tp_search_list:
            for ep_size in ep_search_list:
                for pp_size in pp_search_list:
                    is_tp_valid = self.model_config.head_num % tp_size == 0 and self.model_config.kv_head_num % tp_size == 0
                    is_dp_valid =  world_size % (pp_size * tp_size) == 0
                    dp_size = world_size // (pp_size * tp_size)
                    is_ep_valid = dp_size % ep_size == 0
                    etp_size = tp_size if use_etp else 1
                    is_etp_valid = (world_size %(ep_size* etp_size) == 0) and (etp_size*ep_size < self.system.num_per_node) # TODO(sherry): 临时限定etp_size*ep_size < self.system.num_per_node
                    is_etp_valid = (world_size %(ep_size* etp_size) == 0) # TODO(sherry): 临时限定etp_size*ep_size < self.system.num_per_node

                    if is_dp_valid and is_tp_valid and is_ep_valid and is_etp_valid:
                        layer_distributes = distribute(layer_num, pp_size)
                        for layer_distribute in layer_distributes: 
                            num_layers_in_first_pipeline_stage = layer_distribute[0] if len(layer_distribute) > 1 else None
                            num_layers_in_last_pipeline_stage = layer_distribute[-1] if len(layer_distribute) > 2 else None
                            # middle_pp_stage_layer_num = layer_distribute[1:-1]  if len(layer_distribute) > 2 else None
                        # set strategy  
                        self.strategy.world_size = world_size
                        self.strategy.tp_size = tp_size
                        self.strategy.ep_size = ep_size
                        self.strategy.pp_size = pp_size
                        self.strategy.etp_size = etp_size
                        self.strategy.num_layers_in_first_pipeline_stage = num_layers_in_first_pipeline_stage
                        self.strategy.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage

                
                        search_micro_batch_num = global_batch_size // (self.strategy.dp_size * micro_batch_size)
                        
                        if global_batch_size % (self.strategy.dp_size * micro_batch_size) != 0:
                            continue
                        self.strategy.micro_batch_num = search_micro_batch_num
                        self.strategy.micro_batch_size = micro_batch_size

                        if micro_batch_size != 0 and search_micro_batch_num != 0:
                            if recompute == 'full_block':
                                search_best_strategy = self.search_best_full_recompute_layer_num(
                                                                        layer_num=self.model_config.layer_num, 
                                                                        use_reserved_memory = True,
                                                                        gmi_error=gmi_error,
                                                                        best_mfu=global_best_mfu,
                                                                        all_search_result=all_search_result)
                            elif recompute == 'selective':
                                self.strategy.recompute_layer_num = max(self.model_config.layer_num//pp_size, num_layers_in_first_pipeline_stage if num_layers_in_first_pipeline_stage else 0, num_layers_in_last_pipeline_stage if num_layers_in_last_pipeline_stage else 0)
                                search_best_strategy = self.search_best_selective_recompute(
                                    use_reserved_memory=True,
                                    gmi_error=gmi_error,
                                    best_mfu=global_best_mfu,
                                    all_search_result=all_search_result
                                )
                            else:
                                raise NotImplementedError(f'recompute strategy {recompute} not implemented')
                            
                            if search_best_strategy and 'mfu' in search_best_strategy:
                                global_best_strategy = search_best_strategy
                                global_best_mfu = search_best_strategy['mfu']

        if dump_path is not None and len(global_best_strategy) > 0:
            model_name = self.model_config.model_name
            system_name = self.system.sys_name
            os.makedirs(dump_path, exist_ok=True)


            if 'peak_mem' in global_best_strategy and isinstance(global_best_strategy['peak_mem'], dict):
                global_best_strategy['peak_mem'] = str(global_best_strategy['peak_mem']) # serialize dict to string to avoid csv dump error
            estimate_details_of_best_strategy = global_best_strategy.pop('estimate_details', None)
            best_strategy_df = pd.DataFrame(global_best_strategy, index=[0])
            best_strategy_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seq_len{self.strategy.seq_len}_world_size{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_best_strategy.csv") 
            print(best_strategy_df)

            if all_search_result is not None:
                all_search_result_df = pd.DataFrame(all_search_result)
                all_search_result_df = all_search_result_df.sort_values(by ='mfu',  ascending=False)
                all_search_result_df.to_csv(f"{dump_path}/{model_name}_{system_name}_seq_len{self.strategy.seq_len}_world_size{self.strategy.world_size}_gbs{self.strategy.global_batch_size}_all_search_strategies.csv")
            
            if dump_best_strategy_details and estimate_details_of_best_strategy:
                save_path = f'{dump_path}/best_strategy_details'
                os.makedirs(save_path, exist_ok=True)
                for k, v in estimate_details_of_best_strategy.items():
                    with open(f"{save_path}/{k}.json", "w") as f:
                        f.write(v)
        return global_best_strategy                        
    
    def analysis(self, save_path=None):
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
        tp = self.strategy.tp_size
        ep = self.strategy.ep_size
        pp = self.strategy.pp_size
        print(f'-------------SIMUMAX SUMMARY \033[33mTP={tp},EP={ep},PP={pp}\033[0m -------------')
        print(f'- parallelism = {self.strategy.parallelism}')
        print(f'- recompute = {self.strategy.recompute_status}')
        print(f"- \033[31mdtype = {'fp8' if self.strategy.fp8 else 'bf16'}, grad_reduce = {'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}\033[0m")
        print(f"- system = {self.system.sys_name}")
        print(f"- model = {self.model_config.model_type}")
        print(f"- \033[32mmfu = {compute_result.data['mfu_6nd_with_attn']:.2f}\033[0m")
        print(f"- \033[32mTFLOPS = {compute_result.data['throughput per GPU (TFLOP/s/GPU)']:.2f} (tflops={compute_result.data['flops_info']['theory_flops']}, duration={compute_result.data['duration_time_per_iter']})\033[0m")
        print(f"- TGS_per_gpu = {compute_result.data['throughput_per_accelerator']}")
        print(f"- \033[31mpeak_alloc_mem = {peak_mem}\033[0m")
        # print(f"- peak_alloc_mem_with_reserved = {peak_mem_with_reserved}")
        # print(f'- net = {self.strategy.net} ')
        # print(f"------------------------------------------")
        
        return {
            'peak_mem': peak_mem,
            'TFLOPS': compute_result.data['throughput per GPU (TFLOP/s/GPU)'],
            'TGS_per_gpu': compute_result.data['throughput_per_accelerator'],
            'mfu': compute_result.data['mfu_6nd_with_attn'],
            'parallelism': self.strategy.parallelism,
            'recompute': self.strategy.recompute_status,
            'dtype': f"{'fp8' if self.strategy.fp8 else 'bf16'},grad_reduce in {'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}"
        }
 