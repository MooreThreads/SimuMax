"""performance model for LLM"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union

from .base_struct import PathDebugContext
from .config import StrategyConfig, SystemConfig, ModelConfig
from .base_struct import InputOutputInfo, TensorSize, Result
from .transformer.language_model import LLMModel
from .utils import (
    HumanReadableSize,
    human_readable_bytes,
    convert_final_result_to_human_format,
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

    def run_estimate(self):
        assert self.is_configured, "should call configure() first"

        self.build()

        self._run()


class PerfLLM(PerfBase):
    """Performance model for LLM"""

    def __init__(self) -> None:
        super().__init__()
        self.model_chunk_dict = {}
        self.path_debug_context = PathDebugContext()
        self.path_debug_context_last_stage = PathDebugContext()

    def build(self):
        """
        build first stage model chunk and last stage model chunk
        """
        self.model_chunk_dict = {}

        # Build First Stage Model Chunk
        # Only consider the even divide case fow now
        layer_num = self.model_config.layer_num // self.strategy.pp_size
        if self.strategy.pp_size > 1:
            self.model_chunk_dict["first_stage_chunk"] = LLMModel(
                layer_num=layer_num,
                preprocess=True,
                postprocess=False,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
            )
        else:
            self.model_chunk_dict["first_stage_chunk"] = LLMModel(
                layer_num=layer_num,
                preprocess=True,
                postprocess=True,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
            )
        # # Build Last Stage Model Chunk
        if self.strategy.pp_size > 1:
            self.model_chunk_dict["last_stage_chunk"] = LLMModel(
                layer_num=layer_num,
                preprocess=False,
                postprocess=True,
                model_config=self.model_config,
                strategy=self.strategy,
                system=self.system,
            )

    def _cross_sanity_check(self) -> bool:
        assert (
            self.model_config.layer_num % self.strategy.pp_size == 0
        ), "layer num should be divisible by pp_size"

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

    def _compute_optim_time(self):
        # we use the chunk weight accessed time as the optim time
        result = {"optim_time": 0, "optim_exposed_time": 0}
        model_info = self.model_chunk_dict["first_stage_chunk"].get_model_info()
        state_weight_bytes = model_info.state_bytes
        chunk_weight_accessed_time = 3 * state_weight_bytes
        optim_time = self.system.compute_mem_access_time(chunk_weight_accessed_time)
        optim_exposed_time = optim_time  # no overlap for now
        result["optim_time"] = optim_time
        result["optim_exposed_time"] = optim_exposed_time
        return result

    def _compute_dp_time(self):
        # TODO: support overlap
        result = {"dp_comm_time": 0, "dp_comm_exposed_time": 0}
        dp_comm_time = 0
        model_info = self.model_chunk_dict["first_stage_chunk"].get_model_info()
        comm_size = model_info.grad_bytes  # FIXME: moe need to be handled separately
        dp_net = self.strategy.dp_net
        bucket_size = (
            max(40000000, 1000000 * self.strategy.dp_size) * 4
        )  # consider bucket size
        num_bucket = (comm_size - 1) / bucket_size + 1
        if self.strategy.zero_state >= 1:
            dp_comm_time += num_bucket * self.system.compute_net_op_time(
                "all_gather", bucket_size, comm_num=self.strategy.dp_size, net=dp_net
            )
            dp_comm_time += num_bucket * self.system.compute_net_op_time(
                "reduce_scatter",
                bucket_size,
                comm_num=self.strategy.dp_size,
                net=dp_net,
            )
        else:
            dp_comm_time += num_bucket * self.system.compute_net_op_time(
                "all_reduce", bucket_size, comm_num=self.strategy.dp_size, net=dp_net
            )

        dp_comm_exposed_time = dp_comm_time  # no overlap for now

        result["dp_comm_time"] = dp_comm_time
        result["dp_comm_exposed_time"] = dp_comm_exposed_time
        return result

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
        act_info = self.model_chunk_dict[model_name].get_act_info()
        act_info_with_recomp = self.model_chunk_dict[
            model_name
        ].get_act_info_with_recomp()
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
        result["model_mem"] = (
            model_info.weight_bytes + model_info.grad_bytes + model_info.state_bytes
        )
        result["model_mem_detail"] = deepcopy(model_info.__dict__)
        # skip gradient ckpt micro batch
        # result["act_mem_detail"] = repr(act_info)
        result["skip_ckpt_micro_batch_num_prev"] = skip_ckpt_micro_batch_num_prev
        result["with_recomp_micro_batch_num_prev"] = with_recomp_micro_batch_num_prev
        result["cur_mb_with_recomp"] = cur_mb_with_recomp
        result["act_cached_mem_prev_mbs"] = (
            skip_ckpt_micro_batch_num_prev * act_info.activation_mem_cache
            + with_recomp_micro_batch_num_prev
            * act_info_with_recomp.activation_mem_cache
        )
        cur_act_info = act_info_with_recomp if cur_mb_with_recomp else act_info

        result["cur_mbs_act_mem_detail"] = deepcopy(cur_act_info.__dict__)
        peak_model_mem = result["model_mem"]

        result["fwd_peak_allocated_mem"] = (
            peak_model_mem
            + result["act_cached_mem_prev_mbs"]
            + cur_act_info.fwd_peak_mem
        )
        result["bwd_peak_allocated_mem"] = (
            peak_model_mem
            + result["act_cached_mem_prev_mbs"]
            + cur_act_info.bwd_peak_mem
        )
        result["peak_cached_mem"] = (
            max(result["bwd_peak_allocated_mem"], result["fwd_peak_allocated_mem"])
            / self.strategy.mem_factor
        )
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
            )
            result["last_stage"] = self._analysis_mem_impl(
                micro_batch_num=1, model_name="last_stage_chunk"
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

    def _analysis_compute_time(self, batch_stat_with_recomp, batch_stat_no_recomp):
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
        optim_result = self._compute_optim_time()
        result["optim_time"] = optim_result
        result["fwd_flops"] = batch_stat["compute_info"]["fwd_flops"] * micro_batch_num
        result["recompute_flops"] = (
            batch_stat["compute_info"]["recompute_flops"] * micro_batch_num
        )
        result["bwd_flops"] = batch_stat["compute_info"]["bwd_flops"] * micro_batch_num
        result["model_flops"] = result["fwd_flops"] + result["bwd_flops"]
        return result

    def _analysis_comm_time(self, batch_stat_with_recomp, batch_stat_no_recomp):
        result = {}
        micro_batch_num = self.strategy.micro_batch_num
        dp_comm_result = self._compute_dp_time()
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
                "p2p", pp_comm_size, 2, net=self.strategy.pp_net
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

    def _analysis_single_iter_cost_impl(self):
        # we construct the result in the following hierarchy:
        # first level: useful FlopS、mfu、all FlopS、throughput、duration_per_iter
        # second level: time break down = compute time + comm_time + bubble_time
        # third level-0:  compute time = fwd time + recom_time + bwd_time + optim update time
        # third level-1:  comm_time_: tp_time(tp_time、tp_time_can_overlap) + pp_time
        all_result = {}
        batch_stat_with_recomp = self._analysis_single_batch_cost_impl(
            enable_recompute=True
        )
        batch_stat_no_recomp = self._analysis_single_batch_cost_impl(
            enable_recompute=False
        )
        # 1.comm_result
        comm_result = self._analysis_comm_time(
            batch_stat_with_recomp, batch_stat_no_recomp
        )
        # 2.compute result
        compute_result = self._analysis_compute_time(
            batch_stat_with_recomp, batch_stat_no_recomp
        )
        # 3. all time
        batch_compute_stat = compute_result["batch_compute_stat"]
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
                batch_stat_with_recomp_last_stage, batch_stat_no_recomp_last_stage
            )
            compute_result_last_stage = self._analysis_compute_time(
                batch_stat_with_recomp_last_stage, batch_stat_no_recomp_last_stage
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

        bubble_time = self._compute_bubble_time(chunk_time)
        breakdown_result = {}
        breakdown_result["fwd_compute_time"] = compute_result["fwd_compute_time"]
        breakdown_result["recompute_time"] = compute_result["recompute_time"]
        breakdown_result["bwd_compute_time"] = compute_result["bwd_compute_time"]
        breakdown_result["optim_time"] = compute_result["optim_time"][
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
        # 4.compute first level
        all_tokens_per_iter = self.strategy.seq_len * self.strategy.global_batch_size
        duration_time_per_iter = all_time
        model_flops = compute_result["model_flops"]
        throughput_per_accelerator = (
            all_tokens_per_iter
            / (duration_time_per_iter / 1000)
            / self.strategy.world_size
        )
        mfu = (
            model_flops
            / (duration_time_per_iter / 1000)
            / self.system.accelerator.op["default"].tflops
            / 1e12
        )

        all_result["comm_details"] = comm_result
        all_result["compute_details"] = compute_result
        all_result["breakdown_result"] = breakdown_result
        all_result["chunk_time"] = chunk_time
        all_result["all_tokens_per_iter"] = all_tokens_per_iter
        all_result["duration_time_per_iter"] = duration_time_per_iter
        all_result["mfu"] = mfu
        all_result["mfu_6nd_with_attn"] = (
            self.model_config.flops_per_token(
                context_seq_len=self.strategy.seq_len, with_attn=True
            )
            * throughput_per_accelerator
            / self.system.accelerator.op["default"].tflops
            / 1e12
        )
        all_result["mfu_6nd"] = (
            self.model_config.flops_per_token(
                context_seq_len=self.strategy.seq_len, with_attn=False
            )
            * throughput_per_accelerator
            / self.system.accelerator.op["default"].tflops
            / 1e12
        )
        all_result["throughput_per_accelerator"] = throughput_per_accelerator

        # convert to format
        convert_final_result_to_human_format(all_result)
        return all_result

    def analysis_cost(self):
        result = self._analysis_single_iter_cost_impl()
        return Result(result)

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

    def search_max_micro_batch_size(self):
        left = 1
        right = 2**16
        accelerator_mem_bytes = self.system.accelerator.mem_gbs * 1024**3
        origin_micro_batch_size = self.strategy.micro_batch_size
        origin_micro_batch_num = self.strategy.micro_batch_num
        self.strategy.micro_batch_num = self.strategy.pp_size * 1000
        while left < right:
            micro_batch_size = left + ((right - left) >> 1)
            self.strategy.micro_batch_size = micro_batch_size
            # run
            self.run_estimate()
            # mem analysis
            mem_result = self.analysis_mem()
            if mem_result.get("first_stage") is None:
                peak_cached_mem_bytes = HumanReadableSize.from_string(
                    mem_result.get("peak_cached_mem"),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
            else:
                first_stage_mem_result = mem_result.get("first_stage")
                first_stage_peak_cached_mem = HumanReadableSize.from_string(
                    first_stage_mem_result.get("peak_cached_mem"),
                    base=1024,
                    units=HumanReadableSize.BYTE_UNITS,
                    target_unit="B",
                ).get_value()
                last_stage_mem_result = mem_result.get("last_stage")
                last_stage_peak_cached_mem = HumanReadableSize.from_string(
                    last_stage_mem_result.get("peak_cached_mem"),
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

        self.strategy.micro_batch_size = origin_micro_batch_size
        self.strategy.micro_batch_num = origin_micro_batch_num
        return max_micro_batch_size
