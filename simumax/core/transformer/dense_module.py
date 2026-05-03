"""basic dense transformer module"""

from copy import deepcopy
from simumax.core.tensor import TensorSize, Float8Tensor
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext, LinearBase
from simumax.core.base_struct import (all_gather, reduce_scatter, all_reduce, all_gather_bwd, all2all_fwd, all2all_bwd,
                           AtomModel, LeafModel, FwdQue,
                           send_next, send_prev, recv_next, recv_prev,
                           async_send_next, async_send_prev, async_recv_next, async_recv_prev, async_wait_recv_next, async_wait_recv_prev,
                           sync_send_next, sync_send_prev, sync_wait_recv_next, sync_wait_recv_prev,
                           COM_BUFF)
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig, AttentionRecomputeConfig, MLPRecomputeConfig, ENABLE_SIMU_GRAPH
from simumax.core.utils import format_model_info_microbatch_tag, get_rank_group
import simumax.core.transformer.simu_ops as simu_ops
from simumax.core.transformer.function import ConcatFunction,SplitFunction

#region ------------------ Atomic module ------------------
class Embedding(MetaModule):
    """
    Parallel Embedding Layer
    """

    def __init__(self,
        hidden_size, 
        vocab_size, 
        strategy:StrategyConfig, 
        system:SystemConfig, specific_name=''
        ) -> None:
        super().__init__(strategy, system, specific_name)
        assert vocab_size % self.strategy.tp_size == 0
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size // self.strategy.tp_size
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time))
        if self.strategy.enable_sequence_parallel  and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Embedding"                
            )
            self.layers.append(reduce_scatter(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank))
            state.comm_order += 1
        elif self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Embedding" 
            )
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=0, global_rank=args.rank))
            state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        # FIXME(sherry): seq_len is not correct for sequence parallel
        # if self.strategy.enable_sequence_parallel:
        #     seq_len /= self.strategy.tp_size
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        # hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel:
            seq_len /= self.strategy.tp_size  
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.hidden_size))]
        )
        return output_info

    def _pre_op(self):
        assert len(self.input_info.tensors[0].shape) == 2, "input size should equal 2"

    def _comp_leaf_intra_net_info(self):
        # 1.FWD
        if self.strategy.tp_size > 1:

            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time( # All reduce + Slice
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Embedding"
            )

        # 2.Bwd weight
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.bwd_grad_w_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Embedding"
            )

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        input_size = batch_size * seq_len * 4  
        weight_size = self.vocab_size * self.hidden_size * self.element_size
        output_size = batch_size * seq_len * self.hidden_size * self.element_size
        # FIXME: Aggregation will cause some model weight to be added repeatedly,
        # resulting in an overestimation of the peak
        
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_mem_no_cache = weight_size

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.vocab_size * self.hidden_size
        self._model_info.weight_numel = weight_numel * self.strategy.tp_size # Statistics the parameters of all tp ranks
        self._model_info.dense_weight_bytes = weight_numel * self.element_size
        self._model_info.dense_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        
        optimizer_group_size = self.strategy.dp_size * self.strategy.cp_size
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= optimizer_group_size

    def _comp_leaf_flops_info(self):
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        input_size = batch_size * seq_len * 4
        weight_size = self.vocab_size * self.hidden_size * self.element_size
        output_size = batch_size * seq_len * self.hidden_size * self.element_size
        main_grad_size = self.vocab_size * self.hidden_size * 4
        
        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = 0
        self._compute_info.bwd_grad_w_accessed_mem = 2*main_grad_size # 2 for read and write

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        repr_info = f"hidden_size={self.hidden_size}," f"vocab_size={self.vocab_size}"
        return repr_info

class LinearCol(LinearBase):
    """Support for column parallel linear layer"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        enable_fp8:bool = True,
        is_last_recompute = False,
        use_variance_tail_model: bool = False,
        disable_tensor_parallel = False,
        specific_name='ColumnParallelLinear'
    ) -> None:
        super().__init__(input_size, output_size, strategy, system, specific_name)
        assert output_size % self.strategy.tp_size == 0
        self.layer_idx = layer_idx
        self.input_size = input_size
        self.output_size = output_size if disable_tensor_parallel else output_size // self.strategy.tp_size # Split the output size across tp_size, tensor parallel on attention heads dimension
        self.use_bias = use_bias  # FIXME(for now unless)
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        if self.strategy.fp8 and enable_fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            # fwd compute with sp
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            self.layers.append(all_gather(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank))#'comm all_gather input/ bwd:rs'
            state.comm_order += 1

        elif self.strategy.tp_size > 1:
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=0, bwd_cost=cost, global_rank=args.rank))
            state.comm_order += 1
        #linear
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                                 specific_name='Linear'))
        
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            self.layers.append(all_gather_bwd(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=0, bwd_cost=cost, global_rank=args.rank))  #gather again in bwd to save memory
            state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
    
    @property
    def micro_input_tensor(self):
        """
        full activation size
        """
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel:
            # collect the full sequence data by all-gather, the seq_size is seq_len * tp_size 
            seq_len *= self.strategy.tp_size
        return TensorSize(shape = [batch_size, seq_len, hidden_size], dtype=self.input_info.tensors[0].dtype)
        
    @property
    def micro_hidden_state_size(self):
        """
        full activation size
        """
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel:
            # collect the full sequence data by all-gather, the seq_size is seq_len * tp_size 
            seq_len *= self.strategy.tp_size
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        # hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * self.output_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.enable_sequence_parallel:
            seq_len *= self.strategy.tp_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.output_size))]
        )
        return output_info
    
    def set_breakpoints(self, status):
        self.is_breakpoints = status

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert self.input_size == hidden_size

    def _comp_leaf_intra_net_info(self):
        # all-gather + matmul(fwd)
        # 1.FWD
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            # fwd compute with sp
            # Gather the hidden states of the full sequence from all tp ranks, then compute
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearCol_FWD_SP"
            )

        else:
            # identity operation
            pass

        # 2.Recompute
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time
        # 3.Bwd act
        # grad_input = grad_output.matmul(weight)  # [b*s, output_size] * [output_size, input_size]
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearCol_BWD_ACT_SP"
            )
        elif self.strategy.tp_size > 1:
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearCol_BWD_ACT_TP"
            )
        # 4.Bwd weight
        # grad_weight = grad_output.t().matmul(total_input)
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.bwd_grad_w_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearCol_BWD_W_SP"

            )
    
    def _comp_gemm_mem_accessed_size(self):
        weight_size = self.input_size * self.output_size * self.w_element_size  # fp8_enabled = True, w_element_size=1
        input_size = self.micro_hidden_state_size * self.a_element_size         # fp8_enabled = True, a_element_size=1
        output_size = self.micro_output_grad_size * self.element_size           # fp8_enabled = True, o_element_size=2
        return weight_size, input_size, output_size
    
    def _comp_leaf_act_info_impl(self):
        # self._act_info.activation_mem_cache = self._comp_input_cache_size() # input cache before all-gather
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size
        )
        if self.strategy.enable_sequence_parallel and not self.strategy.fp8:
            # Note: sp only cache the activation slice, when bf16
            self._act_info.activation_mem_cache /= self.strategy.tp_size
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        # if seqence parallel is anabled, the input sequence is multiplied by tp_size(after All-gather)
        weight_size, input_size, output_size = self._comp_gemm_mem_accessed_size()
        
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size
        self._model_info.weight_numel = weight_numel * self.strategy.tp_size # Statistics the parameters of all tp ranks
        self._model_info.dense_weight_bytes = weight_numel * self.w_element_size # fp8_enabled = True, w_element_size=1
        self._model_info.dense_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel # w/m/v
        )
        optimizer_group_size = self.strategy.dp_size * self.strategy.cp_size
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= optimizer_group_size
        self._record_te_dummy_wgrad_shape()

    def _comp_leaf_flops_info(self):
        base_flops = 2 * self.micro_hidden_state_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        # weight_size = self.input_size * self.output_size * self.w_element_size  # fp8_enabled = True, w_element_size=1
        # input_size = self.micro_hidden_state_size * self.a_element_size         # fp8_enabled = True, a_element_size=1
        # output_size = self.micro_output_grad_size * self.element_size           # fp8_enabled = True, o_element_size=2
        weight_size, input_size, output_size = self._comp_gemm_mem_accessed_size()

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size =  self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0
        ))

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_matmul",
                bwd_grad_act_op="fp8_matmul",
                bwd_grad_w_op="fp8_matmul",
                enable_recompute=self.enable_recompute,
            )
        else:
            self._comp_cost_info_impl(
                fwd_op="matmul",
                bwd_grad_act_op="matmul",
                bwd_grad_w_op="matmul",
                enable_recompute=self.enable_recompute,
            )

    def extra_repr(self):
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"use_bias={self.use_bias},"
            # f"fp8_enabled={self.strategy.fp8},"
            f"enable_recompute={self.enable_recompute}, TP={self.strategy.tp_size}"
        )
        return repr_info

class LinearRow(LinearBase):
    """support row parallel linear layer"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute = False,
        use_variance_tail_model: bool = False,
        specific_name='RowParallelLinear'

    ) -> None:
        super().__init__(input_size, output_size, strategy, system,specific_name)
        assert input_size % self.strategy.tp_size == 0
        self.layer_idx = layer_idx
        self.input_size = input_size // self.strategy.tp_size
        self.output_size = output_size
        self.use_bias = use_bias  # FIXME(for now unless)
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        self.has_cached_inputs = has_cached_inputs
        if self.strategy.fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                            bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                            specific_name='Linear'))

        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            
            self.layers.append(reduce_scatter(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size,  com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank))
            state.comm_order += 1
        elif self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size,  com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=0, global_rank=args.rank))
            state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    # @property
    # def weight(self):
    #     return TensorSize(shape=(self.input_size/self.strategy.tp_size, self.output_size))
    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return TensorSize(shape = [batch_size, seq_len, hidden_size], dtype=self.input_info.tensors[0].dtype)

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel:
            seq_len *= self.strategy.tp_size   
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.enable_sequence_parallel:
            seq_len /= self.strategy.tp_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.output_size))]
        )
        return output_info

        return [self.input_size, self.output_size]
    def set_breakpoints(self, status):
        self.is_breakpoints = status

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert (
            self.input_size == hidden_size
        ), f"input_size: {self.input_size} vs hidden_size: {hidden_size}"
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def _comp_leaf_intra_net_info(self):
        # matmul + reduce-scatter/all-reduce
        # 1.FWD
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearROW_FWD_SP"
            )

        elif self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearROW_FWD_TP"
            )

        # 2.Recompute
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time
        # 3.Bwd act and weight
        # grad_input = grad_output.matmul(weight)  # [b*s, output_size] * [output_size, input_size]
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            comm_size = (
                self.micro_output_grad_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            # all gather need for bwd_grad_act and bwd_grad_w, we put it here
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="LinearROW_BWD_SP"
            )
        else:
            # identity operation
            pass
    
    def _comp_gemm_mem_accessed_size(self):
        weight_size = self.input_size * self.output_size * self.w_element_size  # fp8_enabled = True, w_element_size=1
        input_size = self.micro_hidden_state_size * self.a_element_size         # fp8_enabled = True, a_element_size=1
        output_size = self.micro_output_grad_size * self.element_size           # fp8_enabled = True, o_element_size=2
        return weight_size, input_size, output_size
    
    def _comp_leaf_act_info_impl(self):
        
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache -= (
                self.micro_hidden_state_size * self.a_element_size
            )
        weight_size, input_size, output_size = self._comp_gemm_mem_accessed_size()
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size
        self._model_info.weight_numel = weight_numel * self.strategy.tp_size # Statistics the parameters of all tp ranks
        self._model_info.dense_weight_bytes = weight_numel * self.w_element_size # fp8_enabled = True, w_element_size=1
        self._model_info.dense_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        optimizer_group_size = self.strategy.dp_size * self.strategy.cp_size
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= optimizer_group_size
        self._record_te_dummy_wgrad_shape()

    def _comp_leaf_flops_info(self):
        base_flops = 2 * self.micro_hidden_state_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size, input_size, output_size = self._comp_gemm_mem_accessed_size()
        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size =  self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0
        )
        )
        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_matmul",
                bwd_grad_act_op="fp8_matmul",
                bwd_grad_w_op="fp8_matmul",
                enable_recompute=self.enable_recompute,
            )
        else:
            self._comp_cost_info_impl(
                fwd_op="matmul",
                bwd_grad_act_op="matmul",
                bwd_grad_w_op="matmul",
                enable_recompute=self.enable_recompute,
            )

    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"use_bias={self.use_bias},"
            # f"fp8_enabled={self.strategy.fp8},"
            f"enable_recompute={self.enable_recompute}, TP={self.strategy.tp_size}"
        )
        return repr_info


class LayerNorm(MetaModule):
    """Normalization layer, Only support rms_norm now"""

    def __init__(
        self,
        norm_size: int,
        norm_type: str,
        use_fused_norm: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        assert norm_type in ["rms_norm"]
        self.norm_size = norm_size
        self.norm_type = norm_type
        self.use_fused_norm = use_fused_norm
        self.enable_recompute = enable_recompute
        self.has_cached_inputs = has_cached_inputs

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    # def _step(self, t):
    #     fwd_cost = self._cost_info.fwd_compute_time
    #     t[0] += fwd_cost
        
    # def _bwd(self, t):
    #     bwd_cost =  (self._cost_info.bwd_grad_act_time + 
    #                  self._cost_info.bwd_grad_w_time)
    #     t[0] += bwd_cost

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, hidden_size))]
        )
        return output_info

    @property
    def weight(self):
        return TensorSize(shape=(self.norm_size,))
    
    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert self.norm_size == hidden_size

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_leaf_act_info_impl(self):
        # to_fp32 -> power_2 -> mean -> rsqrt -> mul -> to weight dtype -> mul
        # weight_size = self.norm_size
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        rstd_size = self.micro_hidden_state_size / self.norm_size * self.element_size

        if self.use_fused_norm:
            self._act_info.activation_mem_cache = (
                self.micro_hidden_state_size * self.element_size
            )
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= (
                    self.micro_hidden_state_size * self.element_size
                )
            self._act_info.fwd_peak_mem_no_cache = input_size + output_size
            self._act_info.bwd_peak_mem_no_cache = input_size + output_size + rstd_size
        else:
            input_fp32_size = (
                self.micro_hidden_state_size * self.dtype_to_element_size["fp32"]
            )
            output_fp32_size = input_fp32_size
            rstd_fp32_size = (
                self.micro_hidden_state_size
                / self.norm_size
                * self.dtype_to_element_size["fp32"]
            )
            # Only the main memory cached is counted, and the rest will be ignored
            self._act_info.activation_mem_cache += input_fp32_size  # exp
            self._act_info.activation_mem_cache += rstd_fp32_size  # rsqrt
            self._act_info.activation_mem_cache += (
                input_fp32_size + rstd_fp32_size
            )  # mul1
            self._act_info.activation_mem_cache += output_size  # mul2

            # peak point should be first mul operation
            self._act_info.fwd_peak_mem_no_cache = (
                input_fp32_size
                + rstd_fp32_size
                + input_fp32_size
                + output_fp32_size
                + rstd_fp32_size
            )
            # same as fwd
            self._act_info.bwd_peak_mem_no_cache = self._act_info.fwd_peak_mem_no_cache

        self._act_info_with_recomp = self._act_info

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.norm_size
        self._model_info.weight_numel = weight_numel
        self._model_info.dense_weight_bytes = weight_numel * self.element_size
        self._model_info.dense_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        optimizer_group_size = self.strategy.dp_size * self.strategy.cp_size
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= optimizer_group_size

    def _comp_leaf_flops_info(self):
        # ignore memory bound kernel flops for now
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        weight_size = self.norm_size * self.element_size
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        rstd_size = self.micro_hidden_state_size / self.norm_size * self.element_size
        if self.use_fused_norm:
            # FWD
            self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
            # 3 kernels (dx grad, dw partial grad, dw reduce-sum)
            # BWD gamma
            self._compute_info.bwd_grad_w_accessed_mem = (
                input_size + weight_size + weight_size
            )
            # BWD act
            self._compute_info.bwd_grad_act_accessed_mem = (
                input_size + weight_size + output_size + rstd_size
            )
        else:
            weight_size = self.norm_size * self.element_size
            input_fp32_size = (
                self.micro_hidden_state_size * self.dtype_to_element_size["fp32"]
            )
            output_fp32_size = input_fp32_size
            # FWD: to fp32 -> power_2 -> mean -> rsqrt -> mul -> to weight dtype -> mul
            if self.element_size != "fp32":
                # to fp32
                self._compute_info.fwd_accessed_mem += input_size + input_fp32_size
                self._compute_info.fwd_accessed_mem += output_fp32_size + output_size
            self._compute_info.fwd_accessed_mem += (
                4 * input_fp32_size + 4 * rstd_size + output_size + weight_size
            )
            # BWD
            self._compute_info.bwd_grad_w_accessed_mem = 2 * output_size + weight_size
            if self.element_size != "fp32":
                self._compute_info.bwd_grad_act_accessed_mem += (
                    output_size + output_fp32_size
                )
                self._compute_info.bwd_grad_act_accessed_mem += (
                    input_size + input_fp32_size
                )
            self._compute_info.bwd_grad_act_accessed_mem += (
                11 * input_fp32_size + 5 * rstd_size + input_size + weight_size
            )
        # Recompute
        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        repr_info = (
            f"norm_size={self.norm_size},"
            f"norm_type={self.norm_type},"
            f"use_fused_norm={self.use_fused_norm},"
            f"enable_recompute={self.enable_recompute}"
        )
        return repr_info

class Cat(MetaModule):
    """A simple module just to split, cat, reshape tensor"""
    def __init__(
        self,
        strategy: StrategyConfig,
        system: SystemConfig,
        output_dim=None,
    ) -> None:
        super().__init__(strategy, system)
        self.output_dim = output_dim

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.output_dim))]
        )
        return output_info
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(fwd_cost=0,
                                 bwd_cost=0))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_act_info_impl(self):
        """
        Mainly layout operators, ignore for now
        """
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_mem_no_cache = 0

    def _comp_leaf_flops_info(self):
        """
        Mainly layout operators, ignore for now
        """
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0


    def _comp_leaf_mem_accessed_info(self):
        self._compute_info.fwd_accessed_mem = 0
        self._compute_info.bwd_grad_act_accessed_mem = 0
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )
class CoreAttention(MetaModule):
    """Scaled Dot-Product Attention"""

    def __init__(
        self,
        head_size: int,
        head_num: int,
        kv_head_num: int,
        use_math_sdp: bool,
        use_flash_sdp: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name='DotProductAttention',
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.use_math_sdp = use_math_sdp
        self.use_flash_sdp = use_flash_sdp
        self.attention_sparse_ratio = self.strategy.attention_sparse_ratio
        if self.strategy.tp_size > 1:
            assert head_num % self.strategy.tp_size == 0
            assert kv_head_num % self.strategy.tp_size == 0
            head_num = head_num / self.strategy.tp_size
            kv_head_num = kv_head_num / self.strategy.tp_size
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.head_size = head_size
        self.v_head_dim = head_size
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-name:{self.__class__.__name__}"
        rank_info = get_rank_group(args.rank, self.strategy)
        self._append_cp_a2a_layers(args, model_info, rank_info, com_buff=com_buff)
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                                 specific_name='AttentionScore'))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
    
    @property
    def micro_hidden_state_size(self):
        # query: [s, b, n, 192]
        # key: [s, b, n, 192]
        # key: [s, b, n, 128]
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0) # s
        seq_len = self.input_info.tensors[0].size(1) # b
        hidden_size = self.input_info.tensors[0].size(2) # n * head_size
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.head_num * self.head_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, hidden_size))]
        )
        return output_info

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert self.head_size * (2 * self.kv_head_num + self.head_num) == hidden_size
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def get_input_shapes_desc(self, stage):
        hidden_states = self.input_info.tensors[0]
        batch, seq_len = hidden_states.shape[:2]
        if self.strategy.cp_size > 1:
            seq_len = seq_len * self.strategy.cp_size
            head_num = self.head_num // self.strategy.cp_size
            kv_head_num = self.kv_head_num//self.strategy.cp_size
        else:
            head_num = self.head_num
            kv_head_num = self.kv_head_num
        qkv_contiguous = False if 's5000' in self.system.sys_name else True # TODO(sherry): get qkv_contiguous by input stride
        shape_str = f'batch={int(batch)}, seq_len={int(seq_len)}, head_num={int(head_num)}, kv_head_num={int(kv_head_num)}, qk_head_dim={int(self.head_size)}, v_head_dim={int(self.v_head_dim)}, qkv_contiguous={qkv_contiguous}'
        return shape_str

    def _get_cp_a2a_stage_specs(self):
        if not (self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "a2a"):
            return None

        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        q_mem = batch_size * self.head_num * seq_len * self.head_size * self.element_size
        k_mem = batch_size * self.kv_head_num * seq_len * self.head_size * self.element_size
        v_mem = batch_size * self.kv_head_num * seq_len * self.v_head_dim * self.element_size
        o_do_mem = (
            batch_size * self.head_num * seq_len * self.v_head_dim * self.element_size
        )
        bwd_pre = [("Attention_BWD_CP2_DOUT", o_do_mem)]
        if not self.strategy.te_cp_a2a_saves_pre_posta2a_output:
            bwd_pre.insert(0, ("Attention_BWD_CP2_OUT", o_do_mem))
        return {
            "fwd_pre": [
                ("Attention_FWD_CP1_Q", q_mem),
                ("Attention_FWD_CP1_K", k_mem),
                ("Attention_FWD_CP1_V", v_mem),
            ],
            "fwd_post": [("Attention_FWD_CP2", o_do_mem)],
            "bwd_pre": bwd_pre,
            "bwd_post": [
                ("Attention_BWD_CP1_DQ", q_mem),
                ("Attention_BWD_CP1_DK", k_mem),
                ("Attention_BWD_CP1_DV", v_mem),
            ],
        }

    def _append_cp_a2a_layers(self, args, model_info, rank_info, com_buff=None):
        stage_specs = self._get_cp_a2a_stage_specs()
        if stage_specs is None:
            return

        state = args.thread_state
        cp_rank = rank_info.get(
            "cp_rank", (args.rank // self.strategy.tp_size) % self.strategy.cp_size
        )
        cp_group_id = rank_info.get(
            "cp_group_id",
            f"tp:{rank_info['tp_rank']}-pp:{rank_info['pp_rank']}-dp:{rank_info['dp_rank']}",
        )

        def append_comm(comm_cls, stage_name, comm_size):
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.cp_size,
                net=self.strategy.cp_net,
                comm_stage=stage_name,
            )
            fwd_cost = cost if comm_cls is all2all_fwd else 0
            bwd_cost = cost if comm_cls is all2all_bwd else 0
            self.layers.append(
                comm_cls(
                    f"{state.comm_order}-{model_info}-cp_group:{cp_group_id}-stage:{stage_name}",
                    cp_rank,
                    self.strategy.cp_size,
                    com_buff=com_buff,
                    fwd_cost=fwd_cost,
                    bwd_cost=bwd_cost,
                    global_rank=args.rank,
                )
            )
            state.comm_order += 1

        for stage_name, comm_size in stage_specs["fwd_pre"]:
            append_comm(all2all_fwd, stage_name, comm_size)
        for stage_name, comm_size in reversed(stage_specs["bwd_post"]):
            append_comm(all2all_bwd, stage_name, comm_size)
        for stage_name, comm_size in stage_specs["fwd_post"]:
            append_comm(all2all_fwd, stage_name, comm_size)
        for stage_name, comm_size in reversed(stage_specs["bwd_pre"]):
            append_comm(all2all_bwd, stage_name, comm_size)

    @property
    def cp_a2a_mode(self):
        return getattr(self.strategy, "cp_a2a_mode", "async_cp")

    @property
    def cp_a2a_saved_output_is_independent(self):
        return (
            self.strategy.cp_size > 1
            and self.strategy.cp_comm_type == "a2a"
            and self.strategy.te_cp_a2a_saves_pre_posta2a_output
        )

    def _saved_output_cache_mem(self, out_mem):
        # The normal W_o cache keeps the post-attention output consumed by the
        # following linear. TE >= v2.8 additionally saves pre-PostA2A O
        # (`out_part`) for CP A2A backward, so that cache is independent.
        #
        # `cache_outputs` covers other distinct-output cases such as fp8 where
        # attention and W_o keep different representations.
        return (
            out_mem
            if (self.cache_outputs or self.cp_a2a_saved_output_is_independent)
            else 0
        )

    def _cp_a2a_three_tensor_peak(self, first_mem, second_mem, third_mem):
        total_mem = first_mem + second_mem + third_mem
        if self.cp_a2a_mode == "sync_cp":
            # Serial q -> k -> v helper peak is not symmetric. The sync_cp real
            # snapshot peaks when:
            # - original inputs are still live
            # - reordered send buffers for q/k/v are all live
            # - raw recv buffers for k/v are still live
            # - returned / reordered outputs for q/k are already materialized
            # i.e. orig + send + raw_suffix(second+third) + returned_prefix(first+second).
            return 2 * total_mem + (second_mem + third_mem) + (first_mem + second_mem)
        return 4 * total_mem

    def _cp_a2a_two_tensor_peak(self, first_mem, second_mem):
        total_mem = first_mem + second_mem
        if self.cp_a2a_mode == "sync_cp":
            # For sync_cp with two tensors [x0, x1], the helper runs serially but
            # the caller-side originals remain live until the tuple assignment
            # `x0, x1 = flash_attn_a2a_communicate([x0, x1], ...)` completes.
            # The sync peak therefore includes:
            #   original(x0, x1)
            # + send(x0, x1)
            # + raw(x1)
            # + returned(x0)
            # which simplifies to 3 * (x0 + x1) + max(x0, x1).
            return 3 * total_mem + max(first_mem, second_mem)
        return 4 * total_mem

    def _cp_a2a_one_tensor_peak(self, tensor_mem):
        # Single-tensor A2A keeps caller input, send buffer, raw recv, and
        # returned/reordered output live at the helper peak.
        return 4 * tensor_mem

    def _comp_cp_a2a_flash_peak_info(self, q_mem, k_mem, v_mem, out_mem):
        qkv_mem = q_mem + k_mem + v_mem
        saved_output_cache_mem = self._saved_output_cache_mem(out_mem)
        saves_pre_posta2a_output = self.cp_a2a_saved_output_is_independent
        peak_fwd_prea2a = self._cp_a2a_three_tensor_peak(q_mem, k_mem, v_mem)
        peak_fwd_fa = 3 * qkv_mem + out_mem
        # Forward PostA2A only moves a single tensor, so sync/async share the same peak.
        peak_fwd_posta2a = 2 * qkv_mem + 4 * out_mem
        if saves_pre_posta2a_output:
            # TE >= v2.8 saves pre-PostA2A O for backward. Backward only needs
            # to move dO back to the attention layout; saved O is already there.
            peak_bwd_prea2a = (
                saved_output_cache_mem + self._cp_a2a_one_tensor_peak(out_mem)
            )
            out_like_baseline = saved_output_cache_mem + 2 * out_mem
        else:
            # Older TE paths save post-A2A O, so both saved O and incoming dO
            # are moved before flash-attention backward.
            peak_bwd_prea2a = self._cp_a2a_two_tensor_peak(out_mem, out_mem)
            out_like_baseline = 4 * out_mem
        # Backward FA is modeled as stage-local max:
        #   prepare -> kernel -> finalize.
        # Old TE path keeps saved post-A2A O, incoming dO, helper O_part, and
        # helper dO_part live. TE >= v2.8 keeps saved pre-A2A O in cache and
        # only helper-produces dO_part; the caller dO remains live.
        peak_bwd_fa_prepare = qkv_mem + out_like_baseline
        peak_bwd_fa_kernel = 2 * qkv_mem + out_like_baseline + q_mem + k_mem
        peak_bwd_fa_finalize = qkv_mem + out_like_baseline
        peak_bwd_fa = max(
            peak_bwd_fa_prepare,
            peak_bwd_fa_kernel,
            peak_bwd_fa_finalize,
        )
        peak_bwd_posta2a = out_like_baseline + self._cp_a2a_three_tensor_peak(
            q_mem, k_mem, v_mem
        )
        return {
            "peak_fwd_prea2a": peak_fwd_prea2a,
            "peak_fwd_fa": peak_fwd_fa,
            "peak_fwd_posta2a": peak_fwd_posta2a,
            "peak_bwd_prea2a": peak_bwd_prea2a,
            "peak_bwd_fa_prepare": peak_bwd_fa_prepare,
            "peak_bwd_fa_kernel": peak_bwd_fa_kernel,
            "peak_bwd_fa_finalize": peak_bwd_fa_finalize,
            "peak_bwd_fa": peak_bwd_fa,
            "peak_bwd_posta2a": peak_bwd_posta2a,
        }


    def _comp_leaf_intra_net_info(self):
        if self.strategy.cp_size > 1:
            batch_size = self.input_info.tensors[0].size(0)
            seq_len = self.input_info.tensors[0].size(1)
            
            q_size = batch_size * self.head_num * seq_len * self.head_size
            # repeat kv
            k_size = batch_size * self.kv_head_num * seq_len * self.head_size  # batch_size * self.kv_head_num * seq_len * self.head_size
            v_size = batch_size * self.kv_head_num * seq_len * self.v_head_dim   # batch_size * self.kv_head_num * seq_len * self.head_size
            o_do_size = batch_size * self.head_num * seq_len * self.v_head_dim # attention output and do
            qkv_mem = (q_size + k_size + v_size) * self.element_size
            kv_mem = (k_size + v_size) * self.element_size
            o_do_mem = o_do_size * self.element_size 
            if self.strategy.cp_comm_type == "a2a":                
                stage_specs = self._get_cp_a2a_stage_specs()
                for bucket, stage_group in (
                    ("fwd_net_time", stage_specs["fwd_pre"] + stage_specs["fwd_post"]),
                    ("bwd_grad_act_net_time", stage_specs["bwd_post"] + stage_specs["bwd_pre"]),
                ):
                    for stage_name, comm_size in stage_group:
                        setattr(
                            self._cost_info,
                            bucket,
                            getattr(self._cost_info, bucket)
                            + self.system.compute_net_op_time(
                                "all2all",
                                comm_size,
                                comm_num=self.strategy.cp_size,
                                net=self.strategy.cp_net,
                                comm_stage=stage_name,
                            ),
                        )
            elif self.strategy.cp_comm_type == "all_gather":
                # 1. forward ag x 1
                fwd_comm_size = (
                    (kv_mem) * self.strategy.cp_size # get the full kv chunks by ag
                    * self.dtype_to_element_size[self.strategy.dtype]
                )
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all_gather",
                    fwd_comm_size,
                    comm_num=self.strategy.cp_size,
                    net=self.strategy.cp_net,
                    comm_stage="Attention_FWD_CP"
                )
                # 2. backward ag x1 + ag x 1
                bwd_comm_size1 = fwd_comm_size # repeat fwd ag to get the full kv chunks(needed by attention bwd)
                bwd_comm_size2 = fwd_comm_size # dgrad of full kv chunks, rs to each cp rank
                self._cost_info.bwd_net_time += self.system.compute_net_op_time(
                    "all_gather",
                    bwd_comm_size1,
                    comm_num=self.strategy.cp_size,
                    net=self.strategy.cp_net,
                    comm_stage="Attention_BWD_CP1"
                )
                self._cost_info.bwd_net_time += self.system.compute_net_op_time( 
                    "reduce_scatter",
                    bwd_comm_size2,
                    comm_num=self.strategy.cp_size,
                    net=self.strategy.cp_net,
                    comm_stage="Attention_BWD_CP2"
                )
            else:
                raise NotImplementedError(f"cp_comm_type {self.strategy.cp_comm_type} not implemented yet.")   

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)

        q_size = batch_size * self.head_num * seq_len * self.head_size
        # repeat kv
        k_size = batch_size * self.kv_head_num * seq_len * self.head_size  # batch_size * self.kv_head_num * seq_len * self.head_size
        v_size = batch_size * self.kv_head_num * seq_len * self.head_size   # batch_size * self.kv_head_num * seq_len * self.head_size
        lse_size = batch_size * self.head_num * seq_len

        # output_grad_size = batch_size * seq_len * hidden_size  
        output_grad_size = batch_size * seq_len * self.head_size * self.head_num
        qkv_mem = (q_size + k_size + v_size) * self.element_size
        if self.use_flash_sdp:
            lse_mem = lse_size * self.element_size
            out_mem = output_grad_size * self.element_size
            saved_output_cache_mem = self._saved_output_cache_mem(out_mem)
            self._act_info.activation_mem_cache = (
                qkv_mem + lse_mem + saved_output_cache_mem
            )
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= qkv_mem

            self._act_info.fwd_peak_mem_no_cache = qkv_mem + lse_mem + out_mem
            self._act_info.bwd_peak_mem_no_cache = (
                2 * q_size + 2 * k_size + 2 * v_size + lse_size + output_grad_size
            ) * self.element_size - saved_output_cache_mem

            if self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "a2a":
                # Peak semantics in SimuMax:
                # - fwd_peak_mem_no_cache is measured before this module's cache is folded into
                #   global_cache_mem, so the current module's saved cache may contribute here.
                # - bwd_peak_mem_no_cache is measured while this module's saved cache is already
                #   included in global_cache_mem, so saved cache must not be counted again.
                #
                # For cp=a2a + flash attention, model the module as:
                #   caller_qkv -> PreA2A -> q_cp/k_cp/v_cp -> FA -> out_cp -> PostA2A -> out
                # and take the max over the stage-local live sets.
                # Forward candidates.
                peak_info = self._comp_cp_a2a_flash_peak_info(
                    q_size * self.element_size,
                    k_size * self.element_size,
                    v_size * self.element_size,
                    out_mem,
                )
                peak_fwd_save = (
                    qkv_mem + self._act_info.activation_mem_cache
                )
                self._act_info.fwd_peak_mem_no_cache = max(
                    peak_info["peak_fwd_prea2a"],
                    peak_info["peak_fwd_fa"],
                    peak_info["peak_fwd_posta2a"],
                    peak_fwd_save,
                )

                # Backward candidates exclude this module's saved cache, because it is already
                # included in global_cache_mem before bwd_peak_mem_no_cache is evaluated.
                #
                # Important nuance for flash attention:
                # - Forward saves the FA output before PostA2A (`out_part`) for backward.
                # - Backward evaluates bwd_peak_mem_no_cache with cache_for_bwd_mem already
                #   in global_cache_mem, so the saved `out_part` must be excluded here.
                # - Helper-produced tensors outside cache (e.g. `dout_part` / A2A helpers)
                #   stay in bwd_peak_mem_no_cache as temporary/live memory.
                self._act_info.bwd_peak_mem_no_cache = max(
                    peak_info["peak_bwd_prea2a"],
                    peak_info["peak_bwd_fa"],
                    peak_info["peak_bwd_posta2a"],
                ) - saved_output_cache_mem

            elif self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "all_gather":
                kv_mem = (k_size + v_size) * self.element_size
                self._act_info.fwd_peak_mem_no_cache += kv_mem * (self.strategy.cp_size -1) # Forward temporary peak memory, which is the temporary memory after ag
                self._act_info.bwd_peak_mem_no_cache += 2 *kv_mem * (self.strategy.cp_size -1) 
            # TODO(sherry): add cache outputs   
            return
        # kv repeat
        softmax_size = batch_size * self.head_num * seq_len * seq_len
        self._act_info.activation_mem_cache = (
            q_size + k_size + v_size + softmax_size
        ) * self.element_size
        if self.has_cached_inputs and self.head_num == self.kv_head_num:
            # repeat activation
            self._act_info.activation_mem_cache -= (
                q_size + k_size + v_size
            ) * self.element_size
        self._act_info.fwd_peak_mem_no_cache = 2 * softmax_size * self.element_size
        bwd_soft_factor = 3  # naive impl: softmax output + output grad + input grad
        # TODO: mask and dropout will be added later
        if self.system.accelerator.backend == "cuda":
            # The sdp interface of a certain pytorch will use an extra memory,
            # not sure if it is fixed now
            bwd_soft_factor += 1
        elif self.use_math_sdp and self.system.accelerator.backend == "musa":
            # torch_musa sdp kernel reuse the output grad memory
            bwd_soft_factor -= 1
        self._act_info.bwd_peak_mem_no_cache = (
            bwd_soft_factor * softmax_size * self.element_size
        )

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.cp_size > 1:
            if self.strategy.cp_comm_type == "a2a":
                assert self.head_num % self.strategy.cp_size == 0, (
                    f"head_num {self.head_num} must be divisible by cp_size {self.strategy.cp_size}"
                )
                seq_len = seq_len * self.strategy.cp_size
                head_num = self.head_num // self.strategy.cp_size
            elif self.strategy.cp_comm_type == "all_gather":
                raise NotImplementedError(f"cp_comm_type {self.strategy.cp_comm_type} not implemented yet.")
            else:
                raise NotImplementedError(f"cp_comm_type {self.strategy.cp_comm_type} not implemented yet.")
        else:
            head_num = self.head_num
        batch_size = self.input_info.tensors[0].size(0)
        base_flops = (
            2 * batch_size * head_num * self.head_size * seq_len * seq_len
        )  # 1 bmm
        base_flops *= 1 - self.attention_sparse_ratio
        # fwd is calculated by 2 bmm, bwd is calculated by 4 bmm
        self._compute_info.fwd_flops = 2 * base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 4 * base_flops
        if self.use_flash_sdp:
            # flash attn need a extra bmm
            self._compute_info.bwd_grad_act_flops += base_flops
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        q_size = batch_size * self.head_num * seq_len * self.head_size
        k_size = q_size
        v_size = q_size
        # output_grad_size = batch_size * seq_len * hidden_size
        output_grad_size = batch_size * seq_len * self.head_num * self.head_size
        lse_size = batch_size * self.head_num * seq_len
        if self.use_flash_sdp:
            self._compute_info.fwd_accessed_mem = (
                q_size + k_size + v_size + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_act_accessed_mem = (
                2 * q_size + 2 * k_size + 2 * v_size + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_w_accessed_mem = 0
            self._compute_info.recompute_accessed_mem = (
                self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
            )
            return
        softmax_size = batch_size * self.head_num * seq_len * seq_len
        self._compute_info.fwd_accessed_mem += (
            q_size + k_size + softmax_size
        ) * self.element_size
        self._compute_info.fwd_accessed_mem += 2 * softmax_size * self.element_size
        self._compute_info.fwd_accessed_mem += (
            softmax_size + v_size + output_grad_size
        ) * self.element_size
        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (softmax_size + v_size + output_grad_size) * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * softmax_size * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (q_size + k_size + softmax_size) * self.element_size
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="sdp_fwd",
            bwd_grad_act_op="sdp_bwd",
            bwd_grad_w_op="sdp_bwd",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        repr_info = (
            f"head_size={self.head_size},"
            f"head_num={self.head_num},"
            f"kv_head_num={self.kv_head_num},"
            f"use_math_sdp={self.use_math_sdp},"
            f"use_flash_sdp={self.use_flash_sdp},"
            f"enable_recompute={self.enable_recompute}"
        )
        return repr_info
    
class MLACoreAttention(CoreAttention):
    """Scaled Dot-Product Attention for MLA, which v_dim != k_dim"""

    def __init__(
        self,
        head_size: int,
        head_num: int,
        kv_head_num: int,
        use_math_sdp: bool,
        use_flash_sdp: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name='DotProductAttention',
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
        v_head_dim: int=None,
    ) -> None:
        super().__init__(head_size, head_num,kv_head_num,use_math_sdp,use_flash_sdp,
                         has_cached_inputs,enable_recompute,strategy,system,specific_name, is_last_recompute,
                         use_variance_tail_model=use_variance_tail_model)
        self.v_head_dim = v_head_dim
    
    
    #TODO: memory and net usage need to specify while the implement is different from general core attention
    def _comp_leaf_flops_info(self):
        # query: [s, b, n, 192]
        # key: [s, b, n, 192]
        # key: [s, b, n, 128]
        
        seq_len = self.input_info.tensors[0].size(1) # b
        if self.strategy.cp_size > 1:
            assert self.head_num % self.strategy.cp_size == 0, (
                f"head_num {self.head_num} must be divisible by cp_size {self.strategy.cp_size}"
            )
            seq_len = seq_len * self.strategy.cp_size
            head_num = self.head_num // self.strategy.cp_size
        else:
            head_num = self.head_num
        batch_size = self.input_info.tensors[0].size(0) # s
        base_flops = (
            batch_size * head_num * self.head_size * seq_len * seq_len + 
            batch_size * head_num * self.v_head_dim * seq_len * seq_len
        )  # 1 bmm
        base_flops *= 1 - self.attention_sparse_ratio
        # fwd is calculated by 2 bmm, bwd is calculated by 4 bmm
        self._compute_info.fwd_flops = 2 * base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 4 * base_flops
        if self.use_flash_sdp:
            # flash attn need a extra bmm  #TODO: remove from mfu compute
            self._compute_info.bwd_grad_act_flops += base_flops
        self._compute_info.bwd_grad_w_flops = 0

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.head_num * self.v_head_dim
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, hidden_size))]
        )
        return output_info
    
    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert self.head_size * (self.kv_head_num + self.head_num) +self.kv_head_num*self.v_head_dim  == hidden_size, f"{self.head_size * (self.kv_head_num + self.head_num) +self.kv_head_num*self.v_head_dim} vs {hidden_size}"
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)

        q_size = batch_size * self.head_num * seq_len * self.head_size
        k_size = q_size 
        v_size = batch_size * self.head_num * seq_len * self.v_head_dim
        # k_size = batch_size * self.kv_head_num * seq_len * self.head_size
        # v_size = batch_size * self.kv_head_num * seq_len * self.head_size

        lse_size = batch_size * self.head_num * seq_len
        # output_grad_size = batch_size * seq_len * hidden_size  
        output_grad_size = batch_size * seq_len * self.v_head_dim * self.head_num #same as _comp_leaf_mem_accessed_info
        if self.use_flash_sdp:
            qkv_lse_cache_mem = (
                q_size + k_size + v_size + lse_size
            ) * self.element_size
            out_mem = output_grad_size * self.element_size
            saved_output_cache_mem = self._saved_output_cache_mem(out_mem)
            self._act_info.activation_mem_cache = (
                qkv_lse_cache_mem
                + saved_output_cache_mem
            )
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= (
                    q_size + k_size + v_size
                ) * self.element_size

            q_mem = q_size * self.element_size
            k_mem = k_size * self.element_size
            v_mem = v_size * self.element_size
            qkv_mem = q_mem + k_mem + v_mem
            lse_mem = lse_size * self.element_size
            self._act_info.fwd_peak_mem_no_cache = qkv_mem + lse_mem + out_mem
            self._act_info.bwd_peak_mem_no_cache = (
                2 * q_size + 2 * k_size + 2 * v_size + lse_size + output_grad_size
            ) * self.element_size - saved_output_cache_mem

            if self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "a2a":
                peak_info = self._comp_cp_a2a_flash_peak_info(
                    q_mem,
                    k_mem,
                    v_mem,
                    out_mem,
                )
                peak_fwd_save = qkv_mem + self._act_info.activation_mem_cache
                self._act_info.fwd_peak_mem_no_cache = max(
                    peak_info["peak_fwd_prea2a"],
                    peak_info["peak_fwd_fa"],
                    peak_info["peak_fwd_posta2a"],
                    peak_fwd_save,
                )

                self._act_info.bwd_peak_mem_no_cache = max(
                    peak_info["peak_bwd_prea2a"],
                    peak_info["peak_bwd_fa"],
                    peak_info["peak_bwd_posta2a"],
                ) - saved_output_cache_mem
            return
        # kv repeat
        softmax_size = batch_size * self.head_num * seq_len * seq_len
        self._act_info.activation_mem_cache = (
            q_size + k_size + v_size + softmax_size
        ) * self.element_size
        if self.has_cached_inputs and self.head_num == self.kv_head_num:
            # repeat activation
            self._act_info.activation_mem_cache -= (
                q_size + k_size + v_size
            ) * self.element_size
        self._act_info.fwd_peak_mem_no_cache = 2 * softmax_size * self.element_size
        bwd_soft_factor = 3  # naive impl: softmax output + output grad + input grad
        # TODO: mask and dropout will be added later
        if self.system.accelerator.backend == "cuda":
            # The sdp interface of a certain pytorch will use an extra memory,
            # not sure if it is fixed now
            bwd_soft_factor += 1
        elif self.use_math_sdp and self.system.accelerator.backend == "musa":
            # torch_musa sdp kernel reuse the output grad memory
            bwd_soft_factor -= 1
        self._act_info.bwd_peak_mem_no_cache = (
            bwd_soft_factor * softmax_size * self.element_size
        )

    def _comp_leaf_mem_accessed_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        q_size = batch_size * self.head_num * seq_len * self.head_size
        k_size = q_size
        v_size = batch_size * self.head_num * seq_len * self.v_head_dim
        # output_grad_size = batch_size * seq_len * hidden_size   
        ##output_grad_size seems to be the result of A@V, so shape[2] should be n_head*d_v. TODO: check this, and verify corresponding part in CoreAttention
        output_grad_size = batch_size * seq_len * self.v_head_dim * self.head_num 
        lse_size = batch_size * self.head_num * seq_len
        if self.use_flash_sdp:
            self._compute_info.fwd_accessed_mem = (
                q_size + k_size + v_size + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_act_accessed_mem = (
                2 * q_size + 2 * k_size + 2 * v_size + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_w_accessed_mem = 0
            self._compute_info.recompute_accessed_mem = (
                self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
            )
            return
        softmax_size = batch_size * self.head_num * seq_len * seq_len
        self._compute_info.fwd_accessed_mem += (
            q_size + k_size + softmax_size
        ) * self.element_size
        self._compute_info.fwd_accessed_mem += 2 * softmax_size * self.element_size
        self._compute_info.fwd_accessed_mem += (
            softmax_size + v_size + output_grad_size
        ) * self.element_size
        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (softmax_size + v_size + output_grad_size) * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * softmax_size * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (q_size + k_size + softmax_size) * self.element_size
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0
        
class RotaryEmbedding(MetaModule):
    """TODO(sherry): Rotary Positional Embedding"""
    def __init__(
        self,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name:str = "RotaryEmbedding"
    ):
        super().__init__(strategy, system, specific_name)
        self.seq_len = strategy.seq_len
        self.enable_recompute = enable_recompute
        self.has_cached_inputs = has_cached_inputs
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size
    
    def create_output_info(self):
        output_info = InputOutputInfo(
            tensors=[t.new() for t in self.input_info.tensors]
        )
        return output_info
    
    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_act_info_impl(self):
        """
        Mainly layout operators, ignore for now
        """
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_mem_no_cache = 0

    def _comp_leaf_flops_info(self):
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        self._compute_info.fwd_accessed_mem = 0
        self._compute_info.bwd_grad_act_accessed_mem = 0
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    def _comp_leaf_intra_net_info(self):
        pass
    
    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )
    def extra_repr(self) -> str:
        repr_info = f"enable_recompute={self.enable_recompute}"
        return repr_info
    
class Swiglu(MetaModule):
    """Activation function Swish-Gated Linear Unit"""

    def __init__(
        self,
        is_fused: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_weighted_silu: bool = False,
    ) -> None:
        super().__init__(strategy, system)
        self.is_fused = is_fused
        self.enable_recompute = enable_recompute
        self.has_cached_inputs = has_cached_inputs
        self.is_weighted_silu = is_weighted_silu

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H] or [S, H]
        input_numel = self.input_info.tensors[0].numel()
        return input_numel

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        input_numel = self.output_info_.tensors[0].numel()
        return input_numel

    def create_output_info(self):
        hidden_size = self.input_info.tensors[0].size(-1)
        assert hidden_size % 2 == 0, "hidden size should be even"
        output_size_list = list(self.input_info.tensors[0].shape[:-1]) + [
            hidden_size // 2,
        ]
        output_tensor = [TensorSize(shape=tuple(output_size_list))]
        # if len(self.input_info.tensors) > 1:
        #     output_tensor += self.input_info.tensors[1:]
        output_info = InputOutputInfo(tensors=output_tensor)
        return output_info

    def _pre_op(self):
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_leaf_act_info_impl(self):
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        # silu cache 1, mul cache 2
        if self.is_fused:
            self._act_info.activation_mem_cache = 2 * output_size
        else:
            self._act_info.activation_mem_cache = 3 * output_size
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache -= 2 * output_size
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size

        if self.is_weighted_silu:
            probs_mem = self.input_info.tensors[1].numel() * 8
            # self._act_info.activation_mem_cache += probs_mem # probs are cached in the Permutation
            self._act_info.fwd_peak_mem_no_cache += probs_mem
            self._act_info.bwd_peak_mem_no_cache += probs_mem

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        # ignore memory bound kernel flops for now
        # base_flops = self.micro_hidden_state_size  # silu + mul
        self._compute_info.fwd_flops = 0  # base_flops
        self._compute_info.recompute_flops = (
            0  # self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 0  # base_flops / 2 * 3
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        if self.is_fused:
            self._compute_info.fwd_accessed_mem = input_size + output_size
            self._compute_info.bwd_grad_act_accessed_mem = input_size + output_size
            self._compute_info.bwd_grad_w_accessed_mem = 0
        else:
            self._compute_info.fwd_accessed_mem = (
                2 * output_size + 3 * output_size
            )  # silu 2, mul 3
            self._compute_info.bwd_grad_act_accessed_mem = (
                2 * output_size + 2 * 3 * output_size
            )
            self._compute_info.bwd_grad_w_accessed_mem = 0

        if self.is_weighted_silu:
            probs_mem = self.input_info.tensors[1].numel() * self.dtype_to_element_size[self.strategy.dtype]
            self._compute_info.fwd_accessed_mem += probs_mem
            self._compute_info.bwd_grad_act_accessed_mem += probs_mem

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        repr_info = f"is_fused={self.is_fused},enable_recompute={self.enable_recompute}"
        return repr_info

class Gelu(MetaModule):
    """Activation function Gelu"""

    def __init__(
        self,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.enable_recompute = enable_recompute
        self.has_cached_inputs = has_cached_inputs

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H] or [B * S, H]
        input_numel = self.input_info.tensors[0].numel()
        return input_numel

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        input_numel = self.input_info.tensors[0].numel()
        return input_numel

    def create_output_info(self):
        output_tensor = [deepcopy(self.input_info.tensors[0])]
        if len(self.input_info.tensors) > 1:
            output_tensor += self.input_info.tensors[1:]
        output_info = InputOutputInfo(tensors=output_tensor)
        return output_info

    def _pre_op(self):
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_leaf_act_info_impl(self):
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        # silu cache input, mul cache input1 and input2
        self._act_info.activation_mem_cache = 3 * output_size
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache -= input_size

        self._act_info.fwd_peak_mem_no_cache = input_size + output_size
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        # ignore memory bound kernel flops for now
        # base_flops = self.micro_hidden_state_size  # silu + mul
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        input_size = self.micro_hidden_state_size * self.element_size
        # output_size = self.micro_output_grad_size * self.element_size

        self._compute_info.fwd_accessed_mem = 2 * input_size
        self._compute_info.bwd_grad_act_accessed_mem = 2 * input_size
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        repr_info = f"enable_recompute={self.enable_recompute}"
        return repr_info

class ParallelCE(MetaModule):
    """
    input_parallel  -> VocabParallelCrossEntropy(fp32/fp16)
    """

    def __init__(self, strategy:StrategyConfig, system, specific_name='') -> None:
        super().__init__(strategy, system, specific_name)
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time))
        model_info = f"{format_model_info_microbatch_tag(args)}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        vocab_size = self.input_info.tensors[0].size(2)
        comm_size1 = (
            batch_size*seq_len*
            self.dtype_to_element_size["fp32"]
        )
        # Megatron only all-reduces the target-token predicted logit, not the
        # full local-vocab logits tensor, so this payload is also [B, S] fp32.
        comm_size2 = (
            batch_size*seq_len*
            self.dtype_to_element_size["fp32"]
        )
        # comm_size1=comm_size2=1
        cost1 = self.system.compute_net_op_time(
            "all_reduce",
            comm_size1,
            comm_num=self.strategy.tp_size,
            net=self.strategy.tp_net,
        )

        cost2 = self.system.compute_net_op_time(
            "all_reduce",
            comm_size2,
            comm_num=self.strategy.tp_size,
            net=self.strategy.tp_net,
        )
        # comm_size = "B*S"
        self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                fwd_cost=cost1, bwd_cost=0, global_rank=args.rank))
        state.comm_order += 1
        if self.strategy.cross_entropy_loss_fusion:
            # Fused CE batches predicted_logits and sum_exp_logits into one
            # collective. Payload is unchanged, but one launch/latency is saved.
            cost2 = self.system.compute_net_op_time(
                "all_reduce",
                comm_size2 * 2,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                    rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                    fwd_cost=cost2, bwd_cost=0, global_rank=args.rank))
        else:
            # comm_size = "B*S*V"
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                    rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                    fwd_cost=cost2, bwd_cost=0, global_rank=args.rank))
            state.comm_order += 1
            # comm_size = "B*S"
            self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                    rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                    fwd_cost=cost1, bwd_cost=0, global_rank=args.rank))
        state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
        
    def create_output_info(self):
        output_info = InputOutputInfo(tensors=[TensorSize(shape=(1,))])
        return output_info

    def _comp_leaf_intra_net_info(self):
        # FWD
        if self.strategy.tp_size > 1:
            batch_size = self.input_info.tensors[0].size(0)
            seq_len = self.input_info.tensors[0].size(1)
            vocab_size = self.input_info.tensors[0].size(2)
            scalar_comm_size = batch_size * seq_len * self.dtype_to_element_size["fp32"]
            logits_comm_size = scalar_comm_size
            # all_reduce for logits_max [B x S]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                scalar_comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="ParallelCE_FWD_TP"
            )
            if self.strategy.cross_entropy_loss_fusion:
                # Fused CE batches predicted_logits and sum_exp_logits into one
                # collective. Payload is still two [B x S] fp32 tensors.
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all_reduce",
                    logits_comm_size + scalar_comm_size,
                    comm_num=self.strategy.tp_size,
                    net=self.strategy.tp_net,
                    comm_stage="ParallelCE_FWD_TP"
                )
            else:
                # all_reduce for predicted_logits [B x S]
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all_reduce",
                    logits_comm_size,
                    comm_num=self.strategy.tp_size,
                    net=self.strategy.tp_net,
                    comm_stage="ParallelCE_FWD_TP"
                )
                # all_reduce for sum_exp_logits [B x S]
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all_reduce",
                    scalar_comm_size,
                    comm_num=self.strategy.tp_size,
                    net=self.strategy.tp_net,
                    comm_stage="ParallelCE_FWD_TP"
                )

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        vocab_size = self.input_info.tensors[0].size(2)

        if self.strategy.cross_entropy_loss_fusion:
            # TE fused CE saves the input logits shard for backward and uses
            # O(B*S) fp32 work buffers in forward (`loss_1d`, `m_d_X_y`, and
            # optionally gathered `m_d_X_y` when TP > 1). It does not keep the
            # full unfused fp32 CE pipeline as persistent cache.
            logits_cache = (
                batch_size
                * seq_len
                * vocab_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            loss_buf = batch_size * seq_len * self.dtype_to_element_size["fp32"]
            mdxy_local = 3 * batch_size * seq_len * self.dtype_to_element_size["fp32"]
            mdxy_gather = (
                3
                * batch_size
                * seq_len
                * self.strategy.tp_size
                * self.dtype_to_element_size["fp32"]
                if self.strategy.tp_size > 1
                else 0
            )
            self._act_info.activation_mem_cache = logits_cache
            self._act_info.fwd_peak_mem_no_cache = (
                logits_cache + loss_buf + mdxy_local + mdxy_gather
            )
            # Backward updates the saved logits shard in-place and may only
            # need a small grad_output view/contiguous buffer, which we treat
            # as negligible at module level.
            self._act_info.bwd_peak_mem_no_cache = 0
            self._act_info_with_recomp = self._act_info
            return

        # Unfused CE saves the fp32 CE pipeline intermediates.
        ce_cache = (
            batch_size * seq_len * vocab_size * self.dtype_to_element_size["fp32"]
        )
        self._act_info.activation_mem_cache = ce_cache
        self._act_info.fwd_peak_mem_no_cache = (
            ce_cache
            + batch_size
            * seq_len
            * vocab_size
            * self.dtype_to_element_size[self.strategy.dtype]
        )
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info_with_recomp = self._act_info

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        # ignore memory bound kernel flops for now
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        vocab_size = self.input_info.tensors[0].size(2)
        # fwd: 1.max(logits) 2.logits - max 3.exp 4.sum 5.exp/sum_exp_logits
        logtis_size = batch_size * seq_len * vocab_size
        if self.strategy.cross_entropy_loss_fusion:
            # TE fused CE does not materialize the full unfused fp32 pipeline.
            # Keep the model on a simple observable I/O basis: fwd reads bf16
            # logits, writes a saved/probability-sized buffer and loss; bwd
            # reads that saved buffer and writes bf16 grad_input.
            loss_size = batch_size * seq_len
            self._compute_info.fwd_accessed_mem = (
                2 * logtis_size * self.dtype_to_element_size[self.strategy.dtype]
                + loss_size * self.dtype_to_element_size["fp32"]
            )
            self._compute_info.bwd_grad_act_accessed_mem = (
                2 * logtis_size * self.dtype_to_element_size[self.strategy.dtype]
                + loss_size * self.dtype_to_element_size["fp32"]
            )
            self._compute_info.bwd_grad_w_accessed_mem = 0
            self._compute_info.recompute_accessed_mem = (
                self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
            )
            return

        self._compute_info.fwd_accessed_mem = (logtis_size * self.dtype_to_element_size['fp32'] + 
                                                logtis_size * self.dtype_to_element_size['bf16'])
        
        self._compute_info.fwd_accessed_mem += (
            logtis_size + batch_size * seq_len
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # max
        
        self._compute_info.fwd_accessed_mem += (
            logtis_size + batch_size * seq_len + logtis_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # logits - max

        self._compute_info.fwd_accessed_mem += (
            logtis_size * 2
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # exp

        self._compute_info.fwd_accessed_mem += (
            logtis_size + batch_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # sum

        self._compute_info.fwd_accessed_mem += (
            logtis_size + batch_size + logtis_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # exp/sum_exp_logits
        # bwd: p_i = p_i - 1 or q_i
        self._compute_info.bwd_grad_act_accessed_mem = (
            logtis_size + batch_size + logtis_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # only one subtraction operation
        self._compute_info.bwd_grad_act_accessed_mem += (logtis_size * self.dtype_to_element_size['fp32'] + 
                                                        logtis_size * self.dtype_to_element_size['bf16'])
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        ce_op = "ce_fusion" if self.strategy.cross_entropy_loss_fusion else "ce"
        self._comp_cost_info_impl(
            fwd_op=ce_op,
            bwd_grad_act_op=ce_op,
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )

class Float8Quantizer(MetaModule):
    def __init__(self, enable_recompute, strategy, system, specific_name='', parent_module=None):
        super().__init__(strategy, system, specific_name, parent_module)
        self.enable_recompute = enable_recompute 
        self.cache_inputs = False 
        self.cache_outputs = False

    def create_output_info(self):
        if isinstance(self.input_info, TensorSize):
            output_info = InputOutputInfo([Float8Tensor(deepcopy(self.input_info.shape))])
        else:
            output_info = InputOutputInfo([Float8Tensor(deepcopy(t.shape)) for t in self.input_info.tensors])
        return output_info

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = 0.0
        self._act_info.fwd_peak_mem_no_cache = self.all_input_element_num() + self.all_output_element_num()
        self._act_info.bwd_peak_mem_no_cache = 0.0

    def _comp_leaf_mem_accessed_info(self):
        self._compute_info.fwd_accessed_mem = self.all_input_element_num() + self.all_output_element_num()
        self._compute_info.bwd_grad_act_accessed_mem = 0
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    def extra_repr(self):
        return f"enable_recompute={self.enable_recompute}"


#endregion 

#region ----------------- Composite module ----------------
class QuantizedColLinear(MetaModule):
    def __init__(self, 
                 layer_idx, 
                 input_size, 
                 output_size, 
                 use_bias, 
                 has_cached_inputs, 
                 enable_recompute, 
                 strategy, 
                 system, 
                 is_last_recompute = False,
                 use_variance_tail_model: bool = False,
                 disable_tensor_parallel = False,
                 specific_name='QuantizedColLinear'):
        super().__init__(strategy, system, specific_name, parent_module=None)
        assert self.strategy.fp8, 'QuantizedColLinear only support fp8'
        self.quntizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = LinearCol(layer_idx, input_size, output_size, use_bias, has_cached_inputs, enable_recompute, strategy, system, 
                                is_last_recompute = is_last_recompute, 
                                use_variance_tail_model=use_variance_tail_model,
                                disable_tensor_parallel = disable_tensor_parallel)
    
    def set_breakpoints(self, status):
        self.linear.set_breakpoints(status)

    def forward(self, input_info:InputOutputInfo, path_debug_config:PathDebugContext):
        quntized_input_info = self.quntizer(input_info, path_debug_config)
        output = self.linear(quntized_input_info, path_debug_config)
        return output
class QuantizedRowLinear(MetaModule):
    def __init__(self, 
                 layer_idx, 
                 input_size, 
                 output_size, 
                 use_bias, 
                 has_cached_inputs, 
                 enable_recompute, 
                 strategy, 
                 system, 
                 is_last_recompute = False,
                 use_variance_tail_model: bool = False,
                 specific_name='QuantizedColLinear'):
        super().__init__(strategy, system, specific_name, parent_module=None)
        assert self.strategy.fp8, 'QuantizedColLinear only support fp8'
        self.quntizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = LinearRow(layer_idx, input_size, output_size, use_bias, has_cached_inputs, enable_recompute, strategy, system, 
                                is_last_recompute = is_last_recompute,
                                use_variance_tail_model=use_variance_tail_model)

    def set_breakpoints(self, status):
        self.linear.set_breakpoints(status)

    def forward(self, input_info:InputOutputInfo, path_debug_config:PathDebugContext):
        quntized_input_info = self.quntizer(input_info, path_debug_config)
        output = self.linear(quntized_input_info, path_debug_config)
        return output

class Attention(MetaModule):
    """Full Attention Layer"""

    def __init__(
        self,
        layer_idx,
        config: ModelConfig,
        enable_recompute: bool,
        attention_recompute_conf: AttentionRecomputeConfig,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name='',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.layer_idx = layer_idx
        self.config = config
        self.strategy = strategy
        self.system = system
        self.attention_recompute_conf = attention_recompute_conf
        

        self.enable_recompute = enable_recompute   # for old version 
        output_size = (
            self.config.head_num * self.config.head_size
            + 2 * self.config.kv_head_num * self.config.head_size
        )
        if self.strategy.recompute_granularity == "sdp_only":
            self.recompute_granularity = "submodule"
        # if not (attention_recompute.qkv_recompute and
        #         attention_recompute.attn_recompute and
        #         attention_recompute.out_recompute):
        #     self.recompute_granularity = "submodule"

        
        # support selective recompute
        LinearCol_ = QuantizedColLinear if self.strategy.fp8 else LinearCol
        LinearRow_ = QuantizedRowLinear if self.strategy.fp8 else LinearRow
        megatron_layernorm_tail = attention_recompute_conf.megatron_layernorm

        self.linear_qkv = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=output_size,
                use_bias=False,
                has_cached_inputs=megatron_layernorm_tail,
                enable_recompute=attention_recompute_conf.q_up_recompute or megatron_layernorm_tail,
                is_last_recompute=megatron_layernorm_tail,
                use_variance_tail_model=megatron_layernorm_tail,
                strategy=strategy,
                system=system
            )
        
        self.attention = CoreAttention(
                head_size=config.head_size,
                head_num=config.head_num,
                kv_head_num=config.kv_head_num,
                use_math_sdp=self.strategy.use_math_sdp,
                use_flash_sdp=self.strategy.use_flash_sdp,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.core_attn_recompute,
                strategy=strategy,
                system=system,
                is_last_recompute = True
            )
        linear_out_input_size = self.config.head_num * self.config.head_size
        self.linear_out = LinearRow_(
                layer_idx=layer_idx,
                input_size=linear_out_input_size,
                output_size=self.config.hidden_size,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.out_recompute,
                use_bias=False,
                strategy=strategy,
                system=system
            )
        self.attention.cache_outputs = self.strategy.use_flash_sdp and self.strategy.fp8

    def forward(self, input_info:InputOutputInfo, path_debug_context:PathDebugContext):
        qkv = self.linear_qkv(input_info, path_debug_context)
        attn = self.attention(qkv, path_debug_context)
        out = self.linear_out(attn, path_debug_context)
        return out
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        output_info = deepcopy(self.input_info)
        output_info.tensors = [TensorSize(shape=(batch_size, seq_len, hidden_size))]
        return output_info


class MLAAttention(MetaModule):
    """Multi latent Attention Layer"""

    def __init__(
        self,
        layer_idx,
        config: ModelConfig,
        enable_recompute: bool,
        attention_recompute_conf: AttentionRecomputeConfig,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name='',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        assert strategy.tp_size==1, "MLA do not support Tensor Parallel"
        self.layer_idx = layer_idx
        self.config = config
        self.strategy = strategy
        self.system = system
        self.attention_recompute_conf = attention_recompute_conf  # for old version 
        self.enable_recompute = enable_recompute 
        megatron_layernorm_tail = attention_recompute_conf.megatron_layernorm
        # Under CP A2A, TE keeps reordered attention tensors for backward.
        # Treating core_attention as an output-discard tail in this case is too aggressive.
        cp_a2a_mla_tail_bypass = (
            attention_recompute_conf.megatron_mla_up_proj
            and self.strategy.cp_size > 1
            and self.strategy.cp_comm_type == "a2a"
        )
        megatron_mla_up_proj_tail = (
            attention_recompute_conf.megatron_mla_up_proj
            and not cp_a2a_mla_tail_bypass
        )
        megatron_mla_up_proj_core_attn_recompute = (
            attention_recompute_conf.core_attn_recompute
            and not cp_a2a_mla_tail_bypass
        )
        query_projection_size = self.config.v_head_dim * self.config.head_num
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim
        self.num_attention_heads_per_partition = self.config.head_num // self.strategy.tp_size
        # only_enable_sdp = False
        if self.strategy.recompute_granularity == "sdp_only":
            self.recompute_granularity = "submodule"
        #     only_enable_sdp = True
        # if not (attention_recompute.qkv_recompute and
        #          attention_recompute.qkv_norm_recompute and 
        #          attention_recompute.attn_recompute and 
        #          attention_recompute.out_recompute):
        #     self.recompute_granularity = "submodule"

        
        LinearCol_ = QuantizedColLinear if self.strategy.fp8 else LinearCol
        
        if self.config.q_lora_rank is None:
            self.linear_q_proj = LinearCol_(
                    layer_idx=layer_idx,
                    input_size=self.config.hidden_size,
                    output_size=self.config.head_num * self.q_head_dim,
                    use_bias=False,
                    has_cached_inputs=False,
                    enable_recompute=attention_recompute_conf.q_up_recompute,
                    strategy=strategy,
                    system=system
                )
        else:
            self.linear_q_down_proj = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=self.config.q_lora_rank,
                use_bias=False,
                has_cached_inputs=megatron_layernorm_tail,
                enable_recompute=attention_recompute_conf.q_down_recompute,
                is_last_recompute = True,
                use_variance_tail_model=megatron_layernorm_tail,
                # disable_tensor_parallel=True,
                strategy=strategy,
                system=system
            )

            self.q_layernorm = LayerNorm(
                    norm_size=self.config.q_lora_rank,
                    norm_type="rms_norm",
                    use_fused_norm=self.strategy.use_fused_norm,
                    has_cached_inputs=False,
                    enable_recompute=attention_recompute_conf.q_layernorm_recompute,
                    strategy=strategy,
                    system=system
                )
            self.linear_q_up_proj = LinearCol_(
                    layer_idx=layer_idx,
                    input_size=self.config.q_lora_rank,
                    output_size=self.config.head_num * self.q_head_dim, 
                    use_bias=False,
                    has_cached_inputs=False,
                    enable_recompute=attention_recompute_conf.q_up_recompute,
                    strategy=strategy,
                    system=system
                )
        
        self.linear_kv_down_proj = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
                use_bias=False,
                has_cached_inputs=True,
                enable_recompute=attention_recompute_conf.kv_down_recompute,
                is_last_recompute = True,
                use_variance_tail_model=megatron_layernorm_tail,
                strategy=strategy,
                system=system
            )
    
        self.kv_layernorm = LayerNorm(
                norm_size=self.config.kv_lora_rank,
                norm_type="rms_norm",
                use_fused_norm=self.strategy.use_fused_norm,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.kv_layernorm_recompute,
                strategy=strategy,
                system=system
            )
        self.linear_kv_up_proj = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.kv_lora_rank,
                output_size=self.config.head_num * (self.config.qk_head_dim +
                                                                self.config.v_head_dim),
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.kv_up_recompute,
                strategy=strategy,
                system=system,
            )
        self.rotary_pos_emb = RotaryEmbedding(
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.rope_recompute,
                # enable_recompute=False,
                strategy=strategy,
                system=system
            )

        self.core_attention = MLACoreAttention(
                head_size=self.q_head_dim,
                head_num=config.head_num,
                kv_head_num=config.kv_head_num,
                use_math_sdp=self.strategy.use_math_sdp,
                use_flash_sdp=self.strategy.use_flash_sdp,
                has_cached_inputs=megatron_mla_up_proj_tail,
                enable_recompute=megatron_mla_up_proj_core_attn_recompute,
                is_last_recompute = True,
                use_variance_tail_model=megatron_mla_up_proj_tail,
                strategy=strategy,
                system=system,
                v_head_dim=config.v_head_dim
            )
        # TODO(sherry): in selective recompute, linear_out does not recompute yet
        self.linear_out_proj = LinearCol_(
                layer_idx=layer_idx,
                input_size=query_projection_size,
                output_size=self.config.hidden_size,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.out_recompute, # TODO: enable_out_recompute
                use_bias=False,
                strategy=strategy,
                system=system
            )
        if ENABLE_SIMU_GRAPH:
            ...
        else:
            if (
                (self.strategy.mla_rms_recompute or attention_recompute_conf.megatron_layernorm)
                and self.strategy.recompute_granularity == "selective_recompute"
            ):
                if self.config.q_lora_rank is not None:
                    self.linear_q_down_proj.set_breakpoints(True)
                self.linear_kv_down_proj.set_breakpoints(True)
            if self.linear_out_proj.enable_recompute and self.strategy.recompute_granularity == "selective_recompute":
                self.linear_out_proj.is_breakpoints = True
            
            self.core_attention.cache_outputs = (
                self.strategy.use_flash_sdp and self.strategy.fp8
            )

    def forward(self, hidden_states:TensorSize, path_debug_context:PathDebugContext):
        # ref: Megatron-LM/megatron/core/transformer/multi_latent_attention.py:306
        # hidden_states = input_info[0]
        if isinstance(hidden_states, InputOutputInfo):
            hidden_states = hidden_states[0]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        if self.config.q_lora_rank is not None:
            q_compressed = self.linear_q_down_proj(hidden_states, path_debug_context)
            # q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
            # if self.config.sequence_parallel:
                # q_compressed = scatter_to_sequence_parallel_region(q_compressed)
            q = self.linear_q_up_proj(self.q_layernorm(q_compressed, path_debug_context), path_debug_context)
        else:
            # hidden_states:[s, b, 2048], q: [s, b, n * 192]
            q = self.linear_q_proj(hidden_states, path_debug_context)

        q_len, bsz, _ = q.size()

        # q: [s, b, n, 192]
        query = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)
        # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
        # q_no_pe, q_pos_emb = simu_ops.split( # TODO(sherry): add simu_ops.split
        #     q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        # )

        # kv_combined: [s, b, 576]
        kv_combined = self.linear_kv_down_proj(hidden_states, path_debug_context)
        # kv_combined = gather_from_tensor_model_parallel_region(kv_combined)

        # kv_compressed:[s, b, 512], k_pos_emb: [s, b, 64]
        # kv_compressed, k_pos_emb = simu_ops.split(
        #     kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        # )
        kv_compressed, k_pos_emb = SplitFunction.apply(parent_model=self, 
                                             enable_recompute=self.attention_recompute_conf.core_attn_recompute,
                                            tensor_size = kv_combined, 
                                            split_size_or_sections=[self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], split_dim=-1,
                                            path_debug_context=path_debug_context,
                                            name='kv_combined_Split'
                                        )


        # if self.config.sequence_parallel:
        #     kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        # kv: [s, b, 2048]
        kv = self.linear_kv_up_proj(self.kv_layernorm(kv_compressed, path_debug_context), path_debug_context)

        # kv: [s, b, n, 256]
        kv = kv.view(
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
        # k_no_pe, value = simu_ops.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)
        k_no_pe, value = SplitFunction.apply(parent_model=self, 
                                             enable_recompute=self.attention_recompute_conf.core_attn_recompute, 
                                             tensor_size=kv, 
                                             split_size_or_sections=[self.config.qk_head_dim, self.config.v_head_dim], split_dim=-1, 
                                             path_debug_context=path_debug_context,
                                             name='KV_Split')
        

        # [s, b, 64] -> [s, b, 1, 64]
        k_pos_emb:TensorSize = simu_ops.unsqueeze(k_pos_emb, 2)

        k_pos_emb = self.rotary_pos_emb(
            k_pos_emb, path_debug_context
        )


        # key: [s, b, n, 192]
        k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
        key = ConcatFunction.apply(self, self.attention_recompute_conf.core_attn_recompute, 
                                   [k_no_pe, k_pos_emb], 
                                   dim=-1, path_debug_context=path_debug_context,
                                   name='K_pos_emb_Concat') # There is storage overhead, cat is defined as Module to manage statistics

        # TODO(sherry): contiguous
        # query = query.contiguous()
        # key = key.contiguous()
        # value = value.contiguous()
        s, b, n, d = query.size()
        d2 = value.size(-1)
        query = query.view(s, b, n*d)
        key = key.view(s, b, n*d)
        value = value.view(s, b, n*d2)
        # attn_input = simu_ops.cat([query, key, value], dim = -1) # for simuxmax attention input
        attn_input = ConcatFunction.apply(parent_model=self, enable_recompute=self.attention_recompute_conf.core_attn_recompute,
                                          tensor_sizes=[query, key, value], 
                                          dim = -1,
                                          path_debug_context=path_debug_context,
                                          name='QKV_Concat') # for simuxmax attention input
        
        attention_out = self.core_attention(attn_input, path_debug_context) # atomic module

        out = self.linear_out_proj(attention_out, path_debug_context)
        return out
   

    def prefill(self, args, call_stk='', com_buff=None): 
        self.call_stk = call_stk + self.call_stk
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        output_info = deepcopy(self.input_info)
        output_info.tensors = [TensorSize(shape=(batch_size, seq_len, hidden_size))]
        return output_info


class MLP(MetaModule):
    """normal mlp layers"""

    def __init__(self, layer_idx, 
        config:ModelConfig, 
        enable_recompute:bool, 
        mlp_recompute_conf:MLPRecomputeConfig,
        strategy:StrategyConfig, 
        system:SystemConfig, 
        intermediate_size=None
        ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute # for old version 
        is_shared_expert = isinstance(layer_idx, str) and ('shareExpert' in layer_idx)
        dense_mlp_checkpoint = mlp_recompute_conf.linear_recompute or (
            mlp_recompute_conf.megatron_mlp and not is_shared_expert
        )
        shared_expert_checkpoint = mlp_recompute_conf.shared_linear_recompute or (
            mlp_recompute_conf.megatron_moe and is_shared_expert
        )
        if not (dense_mlp_checkpoint or shared_expert_checkpoint):
            self.recompute_granularity = "submodule"
        
        local_intermediate_size = (intermediate_size if intermediate_size is not None 
                             else self.config.intermediate_size)
        intermediate_size = (
            2 * local_intermediate_size
            if self.config.use_swiglu
            else local_intermediate_size
        )
        LinearCol_ = QuantizedColLinear if self.strategy.fp8 else LinearCol
        LinearRow_ = QuantizedRowLinear if self.strategy.fp8 else LinearRow

        # support selective recompute
        enable_recompute = shared_expert_checkpoint if is_shared_expert else dense_mlp_checkpoint
        megatron_layernorm_tail = mlp_recompute_conf.megatron_layernorm and not is_shared_expert
        self.linear_fc1 = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
                use_bias=False,
                has_cached_inputs=megatron_layernorm_tail,
                enable_recompute=enable_recompute or megatron_layernorm_tail,
                is_last_recompute=megatron_layernorm_tail,
                use_variance_tail_model=megatron_layernorm_tail,
                strategy=strategy,
                system=system,
            )
        self.linear_fc2 = LinearRow_(
                layer_idx=layer_idx,
                input_size=local_intermediate_size,
                output_size=self.config.hidden_size,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                is_last_recompute=True,
                use_bias=False,
                strategy=strategy,
                system=system,
            )
        if self.config.use_swiglu:
            self.activation_layer = Swiglu(
                    is_fused=self.strategy.use_fused_swiglu,
                    has_cached_inputs=False,
                    enable_recompute=enable_recompute,
                    strategy=strategy,
                    system=system,
                )
        else:
            self.activation_layer = Gelu(
                    has_cached_inputs=False,
                    enable_recompute=enable_recompute,
                    strategy=strategy,
                    system=system,
                )
        if (
            self.strategy.recompute_granularity == "selective_recompute"
            and mlp_recompute_conf.megatron_layernorm
            and enable_recompute
        ):
            self.linear_fc1.set_breakpoints(True)

    def forward(self, input_info:InputOutputInfo, path_debug_context: PathDebugContext):
        x = self.activation_layer(self.linear_fc1(input_info, path_debug_context), path_debug_context)
        out = self.linear_fc2(x, path_debug_context)
        return out

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff=com_buff)

#endregion 

#region ---------------- optim and schedule ----------------
from simumax.core.transformer.pipeline_schedule import OptimizerSimulator, PpSchedule
#endregion
