"""basic dense transformer module"""

from copy import deepcopy
from simumax.core.tensor import TensorSize, Float8Tensor
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext, LinearBase
from simumax.core.base_struct import (all_gather, reduce_scatter, all_reduce, all_gather_bwd,
                           AtomModel, LeafModel, FwdQue,
                           send_next, send_prev, recv_next, recv_prev,
                           COM_BUFF)
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig, AttentionRecomputeConfig, MLPRecomputeConfig, ENABLE_SIMU_GRAPH
from simumax.core.utils import get_rank_group
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
        model_info = f"microbatch:{args.microbatch}-name:{self.__class__.__name__}"
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
                                         fwd_cost=cost, bwd_cost=cost))
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
                                         fwd_cost=cost, bwd_cost=0))
            state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
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
        weight_size = self.vocab_size * self.hidden_size
        output_size = batch_size * seq_len * self.hidden_size * self.element_size
        # FIXME: Aggregation will cause some model weight to be added repeatedly,
        # resulting in an overestimation of the peak
        
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.vocab_size * self.hidden_size
        self._model_info.weight_numel = weight_numel * self.strategy.tp_size # Statistics the parameters of all tp ranks
        self._model_info.dense_weight_bytes = weight_numel * self.element_size
        self._model_info.dense_grad_bytes = (
            self._model_info.dense_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= self.strategy.dp_size

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
        model_info = f"microbatch:{args.microbatch}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
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
                                         fwd_cost=cost, bwd_cost=cost))#'comm all_gather input/ bwd:rs'
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
                                         fwd_cost=0, bwd_cost=cost))
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
                                         fwd_cost=0, bwd_cost=cost))  #gather again in bwd to save memory
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
        # self._act_info.activation_mem_cache = self._comp_input_cache_size() # 拿到输入的cache信息, all-gather before
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
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size
        self._model_info.weight_numel = weight_numel * self.strategy.tp_size # Statistics the parameters of all tp ranks
        self._model_info.dense_weight_bytes = weight_numel * self.w_element_size # fp8_enabled = True, w_element_size=1
        self._model_info.dense_grad_bytes = (
            self._model_info.dense_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel # w/m/v
        )
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= self.strategy.dp_size

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
        model_info = f"microbatch:{args.microbatch}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
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
                                         fwd_cost=cost, bwd_cost=cost))
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
                                         fwd_cost=cost, bwd_cost=0))
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
        self._model_info.dense_grad_bytes = (
            self._model_info.dense_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= self.strategy.dp_size

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
            self._act_info.fwd_peak_prev_cache_mem = 0
            self._act_info.bwd_peak_mem_no_cache = input_size + output_size + rstd_size
            self._act_info.bwd_peak_prev_cache_mem = 0
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
            self._act_info.fwd_peak_prev_cache_mem = 0
            # same as fwd
            self._act_info.bwd_peak_mem_no_cache = self._act_info.fwd_peak_mem_no_cache
            self._act_info.bwd_peak_prev_cache_mem = 0

        self._act_info_with_recomp = self._act_info

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.norm_size
        self._model_info.weight_numel = weight_numel
        self._model_info.dense_weight_bytes = weight_numel * self.element_size
        self._model_info.dense_grad_bytes = (
            self._model_info.dense_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= self.strategy.dp_size

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
            # 3 kernel (dx grad、dw part grad、dw sum)
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
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

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
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
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
        qkv_contiguous = True # TODO(sherry): get qkv_contiguous by input stride
        shape_str = f'batch={int(batch)}, seq_len={int(seq_len)}, head_num={int(self.head_num)}, kv_head_num={int(self.kv_head_num)}, qk_head_dim={int(self.head_size)}, v_head_dim={int(self.v_head_dim)}, qkv_contiguous={qkv_contiguous}'
        return shape_str


    def _comp_leaf_intra_net_info(self):
        pass

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
            
            self._act_info.activation_mem_cache = qkv_mem + lse_size * self.element_size
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= qkv_mem
            self._act_info.fwd_peak_mem_no_cache = qkv_mem  + (lse_size + output_grad_size) * self.element_size
            self._act_info.fwd_peak_prev_cache_mem = 0
            self._act_info.bwd_peak_mem_no_cache = (
                2 * q_size + 2 * k_size + 2 * v_size + lse_size + output_grad_size
            ) * self.element_size
            self._act_info.bwd_peak_prev_cache_mem = 0
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
        self._act_info.fwd_peak_prev_cache_mem = (q_size + k_size) * self.element_size
        bwd_soft_factor = 3  # naive impl: softmax output + output grad + input grad
        # TODO: mask and dropout will be added later
        if self.system.accelerator.backend == "cuda":
            # The sdp interface of a certain pytorch will use an extra memory,
            # not sure if it is fixed now
            bwd_soft_factor += 1
        self._act_info.bwd_peak_prev_cache_mem = (q_size + k_size) * self.element_size
        self._act_info.bwd_peak_mem_no_cache = (
            bwd_soft_factor * softmax_size * self.element_size
        )

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        seq_len = self.input_info.tensors[0].size(1)
        batch_size = self.input_info.tensors[0].size(0)
        base_flops = (
            2 * batch_size * self.head_num * self.head_size * seq_len * seq_len
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
        v_head_dim: int=None,
    ) -> None:
        super().__init__(head_size,head_num,kv_head_num,use_math_sdp,use_flash_sdp,
                         has_cached_inputs,enable_recompute,strategy,system,specific_name, is_last_recompute)
        self.v_head_dim = v_head_dim
    
    
    #TODO: memory and net usage need to specify while the implement is different from general core attention
    def _comp_leaf_flops_info(self):
        # query: [s, b, n, 192]
        # key: [s, b, n, 192]
        # key: [s, b, n, 128]
        seq_len = self.input_info.tensors[0].size(1) # b
        batch_size = self.input_info.tensors[0].size(0) # s
        base_flops = (
            batch_size * self.head_num * self.head_size * seq_len * seq_len + 
            batch_size * self.head_num * self.v_head_dim * seq_len * seq_len
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
            self._act_info.activation_mem_cache = (
                q_size + k_size + v_size + lse_size
            ) * self.element_size
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= (
                    q_size + k_size + v_size
                ) * self.element_size
            self._act_info.fwd_peak_mem_no_cache = (
                q_size + k_size + v_size + lse_size + output_grad_size
            ) * self.element_size
            self._act_info.fwd_peak_prev_cache_mem = 0
            self._act_info.bwd_peak_mem_no_cache = (
                2 * q_size + 2 * k_size + 2 * v_size + lse_size + output_grad_size
            ) * self.element_size
            self._act_info.bwd_peak_prev_cache_mem = 0
            # TODO(sherry): add cache outputs
            # if self.cache_outputs:
            #     print(f"~~~~~~~ CoreAttention, FA cache output! mem={output_grad_size * self.element_size/1024/1024}MB, self.element_size={self.element_size}")
            #     self._act_info.activation_mem_cache += output_grad_size* self.element_size # TODO(sherry)：check this, fa，需要cache output
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
        self._act_info.fwd_peak_prev_cache_mem = (q_size + k_size) * self.element_size
        bwd_soft_factor = 3  # naive impl: softmax output + output grad + input grad
        # TODO: mask and dropout will be added later
        if self.system.accelerator.backend == "cuda":
            # The sdp interface of a certain pytorch will use an extra memory,
            # not sure if it is fixed now
            bwd_soft_factor += 1
        self._act_info.bwd_peak_prev_cache_mem = (q_size + k_size) * self.element_size
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
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

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
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size
        self._act_info.bwd_peak_prev_cache_mem = 0

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
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size
        self._act_info.bwd_peak_prev_cache_mem = 0

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
        model_info = f"microbatch:{args.microbatch}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        vocab_size = self.input_info.tensors[0].size(2)
        comm_size1 = (
            batch_size*seq_len*
            self.dtype_to_element_size["fp32"]
        )
        comm_size2 = (
            batch_size*seq_len*vocab_size*
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
                                fwd_cost=cost1, bwd_cost=0))
        state.comm_order += 1
        # comm_size = "B*S*V"
        self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                fwd_cost=cost2, bwd_cost=0))
        state.comm_order += 1
        # comm_size = "B*S"
        self.layers.append(all_reduce(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                fwd_cost=cost1, bwd_cost=0))
        state.comm_order += 1
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
        
    def create_output_info(self):
        output_info = InputOutputInfo(tensors=[TensorSize(shape=(1,))])
        return output_info

    def _comp_leaf_intra_net_info(self):
        # FWD
        if self.strategy.tp_size > 1:
            comm_size = 1
            # logits_max = 0  # all_reduce
            # all_reduce for logits_max [b x s]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="ParallelCE_FWD_TP"
            )
            # all_reduce for predicted_logits [b x s]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="ParallelCE_FWD_TP"
            )
            # all reduce for sum_exp_logits [b x s]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="ParallelCE_FWD_TP"
            )

    def _comp_leaf_act_info_impl(self):
        # save exp_logits and mask
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        vocab_size = self.input_info.tensors[0].size(2)

        ce_cache = (
            batch_size * seq_len * vocab_size * self.dtype_to_element_size["fp32"]
        )
        self._act_info.activation_mem_cache = ce_cache
        ce_fwd_peak_no_cache = 2 * ce_cache
        # self._act_info.fwd_peak_mem_no_cache = ce_fwd_peak_no_cache # FIXME(sherry): why double?
        self._act_info.fwd_peak_mem_no_cache = ce_cache 
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = ce_cache # FIXME(sherry)：需要double吗？
        self._act_info.bwd_peak_prev_cache_mem = 0
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
        self._comp_cost_info_impl(
            fwd_op="ce",
            bwd_grad_act_op="default",
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
        self._act_info.fwd_peak_prev_cache_mem = 0.0
        self._act_info.bwd_peak_mem_no_cache = 0.0
        self._act_info.bwd_peak_prev_cache_mem = 0.0

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
                 disable_tensor_parallel = False,
                 specific_name='QuantizedColLinear'):
        super().__init__(strategy, system, specific_name, parent_module=None)
        assert self.strategy.fp8, 'QuantizedColLinear only support fp8'
        self.quntizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = LinearCol(layer_idx, input_size, output_size, use_bias, has_cached_inputs, enable_recompute, strategy, system, 
                                is_last_recompute = is_last_recompute, 
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
                 specific_name='QuantizedColLinear'):
        super().__init__(strategy, system, specific_name, parent_module=None)
        assert self.strategy.fp8, 'QuantizedColLinear only support fp8'
        self.quntizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = LinearRow(layer_idx, input_size, output_size, use_bias, has_cached_inputs, enable_recompute, strategy, system, 
                                is_last_recompute = is_last_recompute)

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

        self.linear_qkv = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=output_size,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.q_up_recompute,
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

        is_bf16_and_next_recompute = (not self.strategy.fp8 and self.linear_out.enable_recompute)
        self.attention.cache_outputs = self.strategy.use_flash_sdp and (self.strategy.fp8 or is_bf16_and_next_recompute) # FIXME(sherry)：bf16

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
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.q_down_recompute,
                is_last_recompute = True,
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
                has_cached_inputs=False,
                enable_recompute=attention_recompute_conf.core_attn_recompute,
                is_last_recompute = True,
                strategy=strategy,
                system=system,
                v_head_dim=config.v_head_dim
            )
        # TODO(sherry)：selective_recompute中，linear_out不做recompute
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
            if self.strategy.mla_rms_recompute and self.strategy.recompute_granularity == "selective_recompute":
                if self.config.q_lora_rank is not None:
                    self.linear_q_down_proj.set_breakpoints(True)
                self.linear_kv_down_proj.set_breakpoints(True)
            if self.linear_out_proj.enable_recompute and self.strategy.recompute_granularity == "selective_recompute":
                self.linear_out_proj.is_breakpoints = True
            
            is_bf16_and_next_recompute = (not self.strategy.fp8 and self.linear_out_proj.enable_recompute)
            self.core_attention.cache_outputs = self.strategy.use_flash_sdp and (self.strategy.fp8 or is_bf16_and_next_recompute) # FIXME(sherry)：bf16

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
        # q_no_pe, q_pos_emb = simu_ops.split( # TODO(sherry): 新增simu_ops.split
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
        if not mlp_recompute_conf.linear_recompute:
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
        enable_recompute = mlp_recompute_conf.shared_linear_recompute if isinstance(layer_idx, str) and ('shareExpert' in layer_idx) else mlp_recompute_conf.linear_recompute
        self.linear_fc1 = LinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
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
class OptimizerSimulator(MetaModule):
    """normal mlp layers"""

    def __init__(self, strategy:StrategyConfig, system, model_info) -> None:
        super().__init__(strategy, system)
        state_weight_bytes = model_info.state_bytes
        chunk_weight_accessed_time = 3 * state_weight_bytes
        self.optim_time = self.system.compute_mem_access_time(chunk_weight_accessed_time)
        self.model_info = model_info

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = f"rank{args.rank}-microbatch{args.microbatch}{call_stk}{self.call_stk}"
        self.layers.append(AtomModel(fwd_cost=self.optim_time, bwd_cost=0))
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        
        comm_size = self.model_info.grad_bytes  # FIXME: moe need to be handled separately
        dp_net = self.strategy.dp_net
        bucket_size = (
            max(40000000, 1000000 * self.strategy.dp_size) * 4
        )  # consider bucket size
        num_bucket = (comm_size - 1) // bucket_size + 1
        if self.strategy.zero_state >= 1:
            cost = num_bucket * self.system.compute_net_op_time(
                "all_gather", bucket_size, comm_num=self.strategy.dp_size, net=dp_net
            )
            self.layers.append(all_gather(f"{state.comm_order}-dp_group:{rank_info['dp_group_id']}", 
                                rank_info['dp_rank'], self.strategy.dp_size,  com_buff=com_buff,
                                fwd_cost=cost))
            state.comm_order += 1
            cost = num_bucket * self.system.compute_net_op_time(
                "reduce_scatter",
                bucket_size,
                comm_num=self.strategy.dp_size,
                net=dp_net,
            )
            self.layers.append(reduce_scatter(f"{state.comm_order}-dp_group:{rank_info['dp_group_id']}", 
                                rank_info['dp_rank'], self.strategy.dp_size,  com_buff=com_buff,
                                fwd_cost=cost))
            state.comm_order += 1
        else:
            cost = num_bucket * self.system.compute_net_op_time(
                "all_reduce", bucket_size, comm_num=self.strategy.dp_size, net=dp_net
            )
            self.layers.append(all_reduce(f"{state.comm_order}-dp_group:{rank_info['dp_group_id']}", 
                                rank_info['dp_rank'], self.strategy.dp_size,  com_buff=com_buff,
                                fwd_cost=cost))
            state.comm_order += 1

        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

class PpSchedule(MetaModule):
    """normal mlp layers"""
    def __init__(self, strategy:StrategyConfig, system, model) -> None:
        super().__init__(strategy, system)
        self.model=model

    def prefill_batch(self, args, com_buff=None):
        job = []
        rank_info = get_rank_group(args.rank, self.strategy)
        pp_size = self.strategy.pp_size
        pp_rank = rank_info['pp_rank']
        pp_group = rank_info['pp_group_id']
        hidden_states_size = (
            self.strategy.micro_batch_size
            * self.strategy.seq_len
            * self.model.model_config.hidden_size
        )
        pp_comm_size = (
            hidden_states_size
            * self.dtype_to_element_size[self.strategy.dtype]
        )
        pp_comm_size = (
            pp_comm_size / self.strategy.tp_size
            if self.strategy.enable_sequence_parallel
            else pp_comm_size
        )
        pp_cost = self.system.compute_net_op_time(
            "p2p", pp_comm_size, 2, net=self.strategy.pp_net
        )  # p2p
        
        num_warmup_microbatches = (
            self.strategy.pp_size
            - rank_info['pp_rank']
            - 1
        )
        num_warmup_microbatches = min(num_warmup_microbatches, self.strategy.micro_batch_num)
        num_microbatches_remaining = self.strategy.micro_batch_num - num_warmup_microbatches
        fwd_stk = []
        fwd_idx = 0  #increase happened immediately after send forward, even no need to send in rank pp-1
        bwd_idx = 0  #increase happened immediately after send bwd, even no need to send in rank 0
        args.microbatch = 0
        for i in range(num_warmup_microbatches):
            if not pp_rank==0:
                cost = pp_cost
                comm = recv_prev(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                
                comm_que = FwdQue(call_stk=f"{args.rank}",que=[comm])
                job.append(comm_que)
                #recv_forward(recv_tensor_shapes, config)
            model = deepcopy(self.model)
            model.prefill(args, com_buff=com_buff)
            args.microbatch += 1
            job.append(model.prefill_fwd())
            fwd_stk.append(model)

            if not pp_rank==pp_size-1:
                #send_forward(output_tensor, send_tensor_shapes, config)
                cost = pp_cost
                comm = send_next(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                comm_que = FwdQue(que=[comm])
                job.append(comm_que)
            fwd_idx += 1
        if num_microbatches_remaining > 0:
            if not pp_rank==0:
                cost = pp_cost
                comm = recv_prev(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                comm_que = FwdQue(que=[comm])
                job.append(comm_que)
                #input_tensor = recv_forward(recv_tensor_shapes, config)

        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            model = deepcopy(self.model)
            model.prefill(args, com_buff=com_buff)
            args.microbatch += 1
            job.append(model.prefill_fwd())
            fwd_stk.append(model)

            # send_forward_recv_backward(output_tensor, send_tensor_shapes, config)
            cost = pp_cost
            que = []
            if not pp_rank==pp_size-1:
                comm1 = send_next(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size,global_rank=args.rank, call_stk=f"rank{args.rank}")
                que.append(comm1)
                comm2 = recv_next(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                que.append(comm2)
                if pp_rank%2:
                    # send then recv
                    comm_que = FwdQue(que=que)
                else:
                    # recv then send
                    comm_que = FwdQue(que=que[::-1])
                job.append(comm_que)
            fwd_idx += 1

            model=fwd_stk.pop()
            job.append(model.prefill_bwd())


            cost = pp_cost
            if last_iteration:
                # send_backward(input_tensor_grad, recv_tensor_shapes, config)
                if not pp_rank==0:
                    comm1 = send_prev(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                    comm_que = FwdQue(que=[comm1])
                    job.append(comm_que)
                bwd_idx += 1
            else:
                # send_backward_recv_forward(input_tensor_grad, recv_tensor_shapes, config)
                que = []
                if not pp_rank==0:
                    comm1 = send_prev(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                    que.append(comm1)
                    comm2 = recv_prev(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                    que.append(comm2)
                    if pp_rank%2:
                        # send then recv
                        comm_que = FwdQue(que=que)
                    else:
                        # recv then send
                        comm_que = FwdQue(que=que[::-1])
                    job.append(comm_que)
                bwd_idx += 1
                
        for i in range(num_warmup_microbatches):
            #recv_backward
            if not pp_rank==pp_size-1:
                cost = pp_cost
                comm = recv_next(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                comm_que = FwdQue(que=[comm])
                job.append(comm_que)

            model=fwd_stk.pop()
            job.append(model.prefill_bwd())

            # send_backward
            if not pp_rank==0:
                cost = pp_cost
                comm = send_prev(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                comm_que = FwdQue(que=[comm])
                job.append(comm_que)

            bwd_idx += 1
        return job
#endregion