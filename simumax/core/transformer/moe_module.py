"""basic moe transformer module"""
import math
from simumax.core.base_struct import (
    MetaModule,
    TensorSize,
    InputOutputInfo,
    GroupLinearBase
)
from simumax.core.base_struct import (all_gather, reduce_scatter, all_reduce, all_gather_bwd, all2all,
                           AtomModel, LeafModel, FwdQue,PathDebugContext,
                           COM_BUFF)
from simumax.core.config import StrategyConfig, SystemConfig, ModelConfig, MLPRecomputeConfig
import simumax.core.transformer.simu_ops as simu_ops
from simumax.core.transformer.dense_module import Swiglu, Gelu, MLP, Float8Quantizer, LinearCol, LinearRow
from simumax.core.utils import get_rank_group
from simumax.core.transformer.function import AddFunction
Input = InputOutputInfo
#region ------------------ Atomic module ------------------
class Router(MetaModule):
    """
    Megatron alltoall-sep impl (fwd)
    1.apply jitter
    2.linear gating
    3.rounting:
      - z_loss for local logits
      - aux loss: input logits, output scores and indexs
        - topk_softmax_with_capacity
        - softmax
        - apply_load_balancing_loss
    """

    def __init__(
        self,
        layer_idx,
        hidden_size: int,
        expert_num: int,
        topk: int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = expert_num // self.strategy.ep_size
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.hidden_size = hidden_size
        self.moe_dispatcher_policy = moe_dispatcher_policy
        # TODO: consider z-loss、aux-loss etc.

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"microbatch:{args.microbatch}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,))
        # routing get full logits
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.local_logits_size
                * self.strategy.tp_size
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


        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def local_logits_size(self):
        assert self.input_info is not None, "Please set input info"
        b = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        ep_num = self.expert_num
        return b * seq_len * ep_num

    def create_output_info(self):
        # FIXME(sherry): check this, return [hidden_states, scores, routting_map]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        # hidden_size = self.input_info.tensors[0].size(2)
        hidden_states = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.expert_num), dtype="int32")]
        )
        return hidden_states
    
    @property
    def weight(self):
        return TensorSize(shape=(self.hidden_size, self.expert_num))
    
    def _pre_op(self): 
        assert self.hidden_size == self.input_info.tensors[0].size(2)

    def _comp_leaf_intra_net_info(self):
        """
        all_gather on seq dim to get full logits(fwd:ag bwd:rs)
        """
        # routing get full logits
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.local_logits_size
                * self.strategy.tp_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Router_FWD_TP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Router_BWD_TP"
            )

    def _comp_leaf_act_info_impl(self):
        """
        activation_mem_cache = input(linear), scores([S,K], softmax)
        """
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size
        self._act_info.activation_mem_cache = input_size * self.element_size
        
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        # Gating, The tensor processed by the softmax is relatively small,
        # so the gating here is used as the operator that appears peak in this module
        gating_weight_size = self.hidden_size * self.expert_num * self.element_size
        input_size = input_size * self.element_size
        output_size = self.local_logits_size * self.element_size
        self._act_info.fwd_peak_mem_no_cache = (
            input_size + output_size + gating_weight_size
        )
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = (
            input_size + output_size + gating_weight_size
        )
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        """
        weight = input(linear), scores([S,K], softmax)
        """
        weight_numel = self.hidden_size * self.expert_num
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
            self._model_info.dense_state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= self.strategy.edp_size

    def _comp_leaf_flops_info(self):
        # Count Gating
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size

        base_flops = 2 * input_size * self.expert_num
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        """
        linear + softmax
        """
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size
        # linear
        gating_weight_size = self.hidden_size * self.expert_num * self.element_size
        linear_input_size = input_size * self.element_size
        linear_output_size = self.local_logits_size * self.element_size
        linear_mem_accessed = (
            gating_weight_size + linear_input_size + linear_output_size
        )
        # softmax
        softmax_input_size = linear_output_size
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            softmax_input_size *= self.strategy.tp_size
        # output_size = self.local_logits_size * self.element_size
        softmax_fwd_mem_accessed = 2 * softmax_input_size
        softmax_bwd_mem_accessed = 3 * softmax_input_size

        self._compute_info.fwd_accessed_mem = (
            linear_mem_accessed + softmax_fwd_mem_accessed
        )
        self._compute_info.bwd_grad_act_accessed_mem = (
            linear_mem_accessed + softmax_bwd_mem_accessed
        )
        self._compute_info.bwd_grad_w_accessed_mem = linear_mem_accessed

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="matmul",
            bwd_grad_act_op="matmul",
            bwd_grad_w_op="matmul",
            enable_recompute=self.enable_recompute,
        )

class Permutation(MetaModule):
    """
    Permutation Impl
    1.all2all on tp group(fwd: alltoall, bwd: all2all) for old policy(all2all-seq)
    2.permute1([S, M] -> [E, C, M] or unbalance [(E1T1, E1T2, ..., E2T1, ...)])
    3.all2all on ep group
    4.permutate2: when local_expert_num > 1,
      arranged according to the batch that the expert needs to process
    5.all_gather feat-dim on tp group or token-dim etp group (fwd: all gather, bwd: reduce_scatter)
    """

    def __init__(
        self,
        layer_idx,
        expert_num: int,
        local_expert_num: int,
        topk: int,
        moe_pad_expert_input_to_capacity:bool,
        capacity:int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy
        self.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity
        self.capacity = capacity

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"microbatch:{args.microbatch}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        

    
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.input_act_size * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )   
            self.layers.append(all2all(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))
            state.comm_order += 1
        
        #permutate1 for ep all2all
        permutate1_mem_accessed = (
            self.input_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype]
        mem_time = self.system.compute_mem_access_time(
            permutate1_mem_accessed
        )
        fwd_compute_time = self.system.compute_end2end_time(0, mem_time)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = mem_time
        bwd_grad_act_time = self.system.compute_end2end_time(0, bwd_grad_act_accessed_mem)
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='permute1'))
        


        if self.strategy.ep_size > 1:
            comm_size = (
                self.permuted_act_size * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
            )   
            self.layers.append(all2all(f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}", 
                                         rank_info['ep_rank'], self.strategy.ep_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))      
            state.comm_order += 1
        if self.strategy.etp_size > 1:
            comm_size = (
                self.permuted_act_size
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            cost = self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            ) 
            self.layers.append(all_gather(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))
            state.comm_order += 1

        #permutate2 after ep all2all and tp
        permutate2_mem_accessed = (
            self.permuted_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype]
        mem_time = self.system.compute_mem_access_time(
            permutate2_mem_accessed
        )
        fwd_compute_time = self.system.compute_end2end_time(0, mem_time)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = mem_time
        bwd_grad_act_time = self.system.compute_end2end_time(0, bwd_grad_act_accessed_mem)
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='permute2'))
        
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def permuted_act_size(self):
        # only consider balanced case for now
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        token_num = self.topk * batch_size * seq_len
        if self.moe_pad_expert_input_to_capacity:
            token_num = math.ceil(token_num/self.expert_num) * self.expert_num * self.capacity
        return token_num * hidden_size

    @property
    def input_act_size(self):
        # only consider balanced case for now
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def _pre_op(self):
        super()._pre_op()
        # if self.strategy.dispatch_probs:
        #     assert len(self.input_info.tensors) == 2, "dispatch_probs=True requires two inputs in Permutation, [x, probs]"
        # else:
        #     assert len(self.input_info.tensors) == 1, "dispatch_probs=False requires one inputs in Permutation, [x]"

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        part_seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel and self.strategy.etp_size > 1:  
            seq_len = part_seq_len * self.strategy.etp_size
            # part_hidden_size = hidden_size // self.strategy.tp_size
        else:
            seq_len = part_seq_len
            # part_hidden_size = hidden_size
        balance_token_num = batch_size * seq_len * self.topk
        if self.moe_pad_expert_input_to_capacity:
            balance_token_num = math.ceil(balance_token_num/self.expert_num) * self.expert_num * self.capacity
        output_info = InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(balance_token_num, hidden_size)
                ),  # permuted moe input
            ]
        )
        return output_info

    def _comp_leaf_intra_net_info(self):
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.input_act_size * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Dispatch_FWD_EP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="Dispatch_BWD_EP"
            )
        if self.strategy.ep_size > 1:
            comm_size = (
                self.permuted_act_size * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                comm_stage="Dispatch_FWD_EP"
            )
        
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                comm_stage="Dispatch_BWD_EP"
            )

            # HACK(sherry): all2all the router probs to expert, and fused combined probs to SiluOp in ExpertMLP, to avoid the activation_mem_cache in Unpermutaion
            if self.strategy.dispatch_probs:
                prob_comm_size = self.input_info.tensors[1].numel() * self.dtype_to_element_size[self.strategy.dtype]
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all2all",
                    prob_comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    comm_stage="Permutation_FWD_EP_PROB"
                )
                self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                    "all2all",
                    prob_comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    comm_stage="Permutation_BWD_EP_PROB"
                )
            # HACK(sherry)

        if self.strategy.etp_size > 1:
            comm_size = (
                self.permuted_act_size
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                comm_stage="Permutation_FWD_ETP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                comm_stage="Permutation_BWD_ETP"
            )
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        probs_mem = self.input_info.tensors[1].numel() * 8
        self._act_info.activation_mem_cache = probs_mem
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        """
        ignore memory bound operation's flops for now
        """
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        """
        permutate1 for ep all2all, scatter
        permutate2 for mlp compute, drop_and_pad=True: transpose + contiuous memory, drop_and_pad=False: sort_chunks_by_idx
        """
        permutate1_mem_accessed, permutate2_mem_accessed = 0, 0
        permutate1_mem_accessed = (
            self.input_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype] # fused: scatter
        permutate2_mem_accessed = (
            self.permuted_act_size + self.permuted_act_size # drop_and_pad=True:transpose + contiuous memory(drop_and_pad), drop_and_pad=False:sort_chunks_by_idx
        ) * self.dtype_to_element_size[self.strategy.dtype]

        self._compute_info.fwd_accessed_mem = (
            permutate1_mem_accessed + permutate2_mem_accessed
        )
        self._compute_info.bwd_grad_act_accessed_mem = (
            permutate1_mem_accessed + permutate2_mem_accessed
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="permute_fwd",
            bwd_grad_act_op="permute_bwd",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )


class UnPermutation(MetaModule):
    """
    Reverse permutation
    1.reduce_scatter feat-dim on tp group or token-dim etp group
    2.Unpermuation1: when local_expert_num > 1, rearraange for all2all
    3.all2all on ep group
    4.Unpermuation2:
        - no (padding and drop):
          - 通过argsort的sorted_indices反向unpermutate,然后根据probs进行combine，但是drop的没有残差连接
        - padding and drop：
          - 通过final_indices(token indices, [E, C])和scatter_add实现恢复和combine weight
    5.all2all on tp group for old policy(all2all-seq)
    """

    def __init__(
        self,
        layer_idx,
        expert_num: int,
        local_expert_num: int,
        topk: int,
        # moe_pad_expert_input_to_capacity:bool,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy
        self.ori_shape = None

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"microbatch:{args.microbatch}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        

    
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.act_size_after_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )   
            self.layers.append(all2all(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))
            state.comm_order += 1
        
        #unpermutate1 before tp and ep all2all
        unpermutate1_mem_accessed = (
            2 * self.act_size_before_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]
        mem_time = self.system.compute_mem_access_time(
            unpermutate1_mem_accessed
        )
        fwd_compute_time = self.system.compute_end2end_time(0, mem_time)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = mem_time
        bwd_grad_act_time = self.system.compute_end2end_time(0, bwd_grad_act_accessed_mem)
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='unpermute1'))
        
        if self.strategy.etp_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            cost = self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            ) 
            self.layers.append(reduce_scatter(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))
            state.comm_order += 1


        if self.strategy.ep_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
            )   
            self.layers.append(all2all(f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}", 
                                         rank_info['ep_rank'], self.strategy.ep_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))      
            state.comm_order += 1

        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.act_size_after_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )  
            self.layers.append(all2all(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost))
            state.comm_order += 1

        #permutate2 and combine
        unpermutate2_and_combine_mem_accessed = (
            self.act_size_before_combined + self.act_size_after_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]
        mem_time = self.system.compute_mem_access_time(
            unpermutate2_and_combine_mem_accessed
        )
        fwd_compute_time = self.system.compute_end2end_time(0, mem_time)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = mem_time
        bwd_grad_act_time = self.system.compute_end2end_time(0, bwd_grad_act_accessed_mem)
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='unpermutate2_and_combine'))
        
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def act_size_before_combined(self):
        # only consider balanced case
        act_size = self.input_info.tensors[0].numel()
        return act_size

    @property
    def act_size_after_combined(self):
        # only consider balanced case
        act_size = self.output_info_.tensors[0].numel()
        return act_size
    
    def _pre_op(self):
        super()._pre_op()
        if not self.strategy.dispatch_probs:
            assert len(self.input_info.tensors) == 2, "dispatch_probs=False requires two inputs in Permutation, [x, probs]"
        else:
            assert len(self.input_info.tensors) == 1, "dispatch_probs=True requires one inputs in Permutation, [x]"
    def set_ori_shape(self, shape):
        self.ori_shape = shape

    def create_output_info(self):
        # recover the original input
        assert self.output_info_ is None
        assert self.ori_shape is not None
        output_info = InputOutputInfo(tensors=[TensorSize(shape=self.ori_shape)])
        # print('-- unpermute output_info', output_info)
        return output_info

    def _comp_leaf_intra_net_info(self):
        if self.strategy.etp_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                comm_stage="UnPermutation_FWD_ETP"
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                comm_stage="UnPermutation_BWD_ETP"
            )

        # all2all on ep group
        if self.strategy.ep_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                comm_stage="UnPermutation_FWD_EP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                comm_stage="UnPermutation_BWD_EP"
            )
        # all2all on tp group
        if self.moe_dispatcher_policy == "all2all-seq" and self.strategy.tp_size > 1:
            comm_size = (
                self.act_size_after_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="UnPermutation_FWD_TP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                comm_stage="UnPermutation_BWD_TP"
            )
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        """
        Mainly layout operators, ignore for now
        """
        # HACK(sherry): the weighted_probs is fused in SiluOP.
        # if dispatch_probs=True, no cache.
        # if dispatch_probs=False, cache the unpermute_before_hidden_states and probs (for mul op).
        if self.strategy.dispatch_probs:
            self._act_info.activation_mem_cache = 0
            self._act_info.fwd_peak_mem_no_cache = max(self.act_size_before_combined, self.act_size_after_combined) * self.element_size
            self._act_info.bwd_peak_mem_no_cache = 0
        else:
            # mul
            self._act_info.activation_mem_cache =  self.act_size_before_combined * self.element_size # Cache hidden states, probs are cache in Permutation
            self._act_info.fwd_peak_mem_no_cache = self.act_size_before_combined * self.element_size + self.act_size_after_combined * self.element_size
            self._act_info.bwd_peak_mem_no_cache = self.act_size_before_combined * self.element_size + self.act_size_after_combined * self.element_size
        # HACK(sherry)
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        """
        Mainly layout operators, ignore for now
        """
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        """
        4.Unpermuation2:
        - no (padding and drop):
          - 通过argsort的sorted_indices反向unpermutate,然后根据probs进行combine，但是drop的没有残差连接
        - padding and drop：
          - 通过final_indices(token indices, [E, C])和scatter_add实现恢复和combine weight

        1.permutate1 for ep all2all
        2.permutate2 for mlp compute
        3.combine scores
        """
        # pylint: disable=invalid-name
        permutate1_mem_accessed = ( # none-fused: contiguous memory(drop_and_pad) or sort_chunks_by_idxs
            2 * self.act_size_before_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]
        
        permutate2_and_combine_mem_accessed = ( # fused-op: combine permuted_features by probs and scatter_add
            self.act_size_before_combined + self.act_size_after_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]

        self._compute_info.fwd_accessed_mem = (
            permutate1_mem_accessed + permutate2_and_combine_mem_accessed
        )
        self._compute_info.bwd_grad_act_accessed_mem = (
            permutate1_mem_accessed + permutate2_and_combine_mem_accessed
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )
        # pylint: enable=invalid-name

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="permute_fwd",
            bwd_grad_act_op="permute_bwd",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )


class GroupLinearCol(GroupLinearBase):
    """Multi Expert Linear Layer, Suport column parallelism"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False
    ) -> None:
        super().__init__(local_expert_num, input_size, output_size, strategy, system)
        assert mode in ['parallel', 'serial']
        assert output_size % self.strategy.etp_size == 0
        self.layer_idx = layer_idx
        self.local_expert_num = local_expert_num
        self.input_size = input_size
        self.output_size = output_size // self.strategy.etp_size
        self.use_bias = use_bias  # for now unless
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        if self.strategy.fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

        if mode == "serial":
            import types
            for i in range(self.local_expert_num):
                setattr(self, f"linear_{i}", LinearCol(layer_idx=layer_idx,
                                                    input_size=input_size, 
                                                    output_size=output_size,
                                                    use_bias=use_bias,
                                                    has_cached_inputs=False,
                                                    enable_recompute=enable_recompute,
                                                    strategy=strategy,
                                                    system=system)
                )   
            def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
                input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
                out = []
                for i in range(self.local_expert_num):
                    linear_i = getattr(self, f"linear_{i}")
                    x = simu_ops.unsqueeze(input[i], 0)
                    x = linear_i(x, path_debug_context)
                    out.append(simu_ops.squeeze(x, 0))
                out = simu_ops.cat(out, 0)
                return out
            # Methods to bind functions as instances
            self.forward = types.MethodType(forward, self)

            
    def prefill(self, args, call_stk='', com_buff=None):
        # tp comm is in Permuation
        self.call_stk = call_stk + self.call_stk
        #linear
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                                 specific_name='Linear'))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return TensorSize(shape = [token_num, hidden_size], dtype=self.input_info.tensors[0].dtype)
    
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        # token_num = self.input_info.tensors[1].size(0)
        # hidden_size = self.input_info.tensors[1].size(1)
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        # if self.strategy.enable_sequence_parallel:
        #     hidden_size *= self.strategy.etp_size
        return token_num * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        token_num = self.output_info_.tensors[0].size(0)
        return token_num * self.output_size

    def create_output_info(self):
        token_num = self.input_info.tensors[0].size(0)
        origin_input_info = self.input_info.tensors[1:]
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(token_num, self.output_size))]
            + origin_input_info
        )
        return output_info
    
    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(1)
        assert self.input_size == hidden_size

    def _comp_leaf_intra_net_info(self):
        # tp comm is in Permuation
        pass

    def _comp_leaf_act_info_impl(self):
        
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size # fp8
        )
        if self.has_cached_inputs or self.offload_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.local_expert_num
            * self.input_size
            * self.output_size
            * self.w_element_size # fp8
        )
        grad_size = (
            self.local_expert_num
            * self.input_size
            * self.output_size
            * self.dtype_to_element_size['fp32'] # fp8
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size   # bf16
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + (grad_size if self.strategy.fp8 else 0) + (input_size if self.offload_inputs else 0)
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.local_expert_num * self.input_size * self.output_size
        self._model_info.moe_weight_numel = weight_numel * self.strategy.ep_size * self.strategy.etp_size # Statistics the parameters of all etp ranks and ep ranks
        self._model_info.moe_weight_bytes = weight_numel * self.w_element_size # fp8
        self._model_info.moe_grad_bytes = (
            self._model_info.moe_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.moe_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.moe_state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.moe_grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.moe_weight_bytes /= self.strategy.edp_size

    def _comp_leaf_flops_info(self):
        token_num = self.input_info.tensors[0].size(0)
        base_flops = 2 * token_num * self.input_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = (
            self.input_size
            * self.output_size
            * self.w_element_size  # fp8
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size # bf16

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size = self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0)
        )

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_group_matmul",
                bwd_grad_act_op="fp8_group_matmul",
                bwd_grad_w_op="fp8_group_matmul",
                enable_recompute=self.enable_recompute,
            )
        else:
            self._comp_cost_info_impl(
                fwd_op="group_matmul",
                bwd_grad_act_op="group_matmul",
                bwd_grad_w_op="group_matmul",
                enable_recompute=self.enable_recompute,
            )


    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"local_expert_num={self.local_expert_num},"
            f"use_bias={self.use_bias}"
        )
        return repr_info


class GroupLinearRow(GroupLinearBase):
    """Multi Expert Linear Layer, Suport row parallelism"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False
    ) -> None:
        super().__init__(local_expert_num, input_size, output_size, strategy, system)
        assert mode in ['parallel', 'serial']
        assert input_size % self.strategy.etp_size == 0
        self.layer_idx = layer_idx
        self.local_expert_num = local_expert_num
        self.input_size = input_size // self.strategy.etp_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        if self.strategy.fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

        if mode == "serial":
            import types
            for i in range(self.local_expert_num):
                setattr(self, f"linear_{i}", LinearCol(layer_idx=layer_idx,
                                                    input_size=input_size, 
                                                    output_size=output_size,
                                                    use_bias=use_bias,
                                                    has_cached_inputs=False,
                                                    enable_recompute=enable_recompute,
                                                    strategy=strategy,
                                                    system=system)
                )   
            def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
                input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
                out = []
                for i in range(self.local_expert_num):
                    linear_i = getattr(self, f"linear_{i}")
                    x = simu_ops.unsqueeze(input[i], 0)
                    x = linear_i(x, path_debug_context)
                    out.append(simu_ops.squeeze(x, 0))
                out = simu_ops.cat(out, 0)
                return out
            # Methods to bind functions as instances
            self.forward = types.MethodType(forward, self)
            
    def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
        input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
        out = []
        for i in range(self.local_expert_num):
            linear_i = getattr(self, f"linear_{i}")
            x = simu_ops.unsqueeze(input[i], 0)
            x = linear_i(x, path_debug_context)
            out.append(simu_ops.squeeze(x, 0))
        out = simu_ops.cat(out, 0)
        return out
    
    def prefill(self, args, call_stk='', com_buff=None):
        # tp comm is in UnPermuation
        self.call_stk = call_stk + self.call_stk
        #linear
        self.layers.append(AtomModel(fwd_cost=self._cost_info.fwd_compute_time,
                                 bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                                 specific_name='Linear'))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return TensorSize(shape = [token_num, hidden_size], dtype=self.input_info.tensors[0].dtype)
    
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return token_num * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        token_num = self.output_info_.tensors[0].size(0)
        hidden_size = self.output_info_.tensors[0].size(1)
        # hidden_size = self.output_info.tensors[0].size(2)
        return token_num * hidden_size

    def create_output_info(self):
        token_num = self.input_info.tensors[0].size(0)
        origin_input_info = self.input_info.tensors[1:]

        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(token_num, self.output_size))]
            + origin_input_info
        )
        return output_info

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(1)
        assert self.input_size == hidden_size, f"input_size {self.input_size} != hidden_size {hidden_size}"

    def _comp_leaf_intra_net_info(self):
        # tp comm is in UnPermuation
        pass

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size # fp8
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.input_size
            * self.output_size
            * self.local_expert_num
            * self.w_element_size # fp8
        )
        grad_size =  (
            self.input_size
            * self.output_size
            * self.local_expert_num
            * self.dtype_to_element_size['fp32'] # fp8
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size # bf16
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size +  (grad_size if self.strategy.fp8 else 0)
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size * self.local_expert_num
        self._model_info.moe_weight_numel = weight_numel * self.strategy.ep_size * self.strategy.etp_size # Statistics the parameters of all etp ranks and ep ranks
        self._model_info.moe_weight_bytes = weight_numel * self.w_element_size # fp8
        self._model_info.moe_grad_bytes = (
            self._model_info.moe_weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self.dtype_to_element_size["fp32"] * weight_numel
        )
        self._model_info.moe_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.moe_state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.moe_grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.moe_weight_bytes /= self.strategy.edp_size

    def _comp_leaf_flops_info(self):
        token_num = self.input_info.tensors[0].size(0)
        base_flops = 2 * token_num * self.input_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = (
            self.input_size
            * self.output_size
            * self.w_element_size # fp8
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.a_element_size  # fp8
        output_size = self.micro_output_grad_size * self.element_size   # bf16

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size = self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0)
        )

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_group_matmul",
                bwd_grad_act_op="fp8_group_matmul",
                bwd_grad_w_op="fp8_group_matmul",
                enable_recompute=self.enable_recompute,
            )
        else:
            self._comp_cost_info_impl(
                fwd_op="group_matmul",
                bwd_grad_act_op="group_matmul",
                bwd_grad_w_op="group_matmul",
                enable_recompute=self.enable_recompute,
            )

    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"local_expert_num={self.local_expert_num},"
            f"use_bias={self.use_bias}"
        )
        return repr_info
#endregion 

#region ----------------- Composite module ----------------
class QuantizedGroupLinearCol(MetaModule):
    def __init__(self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False
        ):
        super().__init__(strategy, system)
        quantizer_recompute = False if strategy.cache_groupgemm_col_fp8_inputs else enable_recompute
        self.quantizer = Float8Quantizer(enable_recompute=quantizer_recompute, strategy=strategy, system=system)
        enable_cahce_bf16_inputs = not self.strategy.cache_groupgemm_col_fp8_inputs
        if enable_cahce_bf16_inputs:
            self.quantizer.offload_inputs = self.strategy.offload_groupgemm_col_inputs  # the quantizer can perform offload When the input of bf16 needs to be cached
       
        self.linear = GroupLinearCol(
            layer_idx,
            input_size,
            output_size,
            local_expert_num,
            use_bias,
            has_cached_inputs,
            enable_recompute,
            mode,
            strategy,
            system,
            is_last_recompute
        )
    def forward(self, hidden_states, path_debug_context=None):
        hidden_states = self.quantizer(hidden_states, path_debug_context)
        hidden_states = self.linear(hidden_states, path_debug_context)
        return hidden_states
    

class QuantizedGroupLinearRow(MetaModule):
    def __init__(self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        if_first_recompute: bool = False,
        is_last_recompute: bool = False
    ):
        super().__init__(strategy, system)
        self.quantizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = GroupLinearRow(
                    layer_idx,
                    input_size,
                    output_size,
                    local_expert_num,
                    use_bias,
                    has_cached_inputs,
                    enable_recompute,
                    mode,
                    strategy,
                    system,
                    is_last_recompute
                )

    def forward(self, hidden_states, path_debug_context=None):
        hidden_states = self.quantizer(hidden_states, path_debug_context)
        hidden_states = self.linear(hidden_states, path_debug_context)
        return hidden_states

class ExpertMLP(MetaModule):
    """Expert MLP Layer"""

    def __init__(self, 
                 layer_idx, 
                 config:ModelConfig, 
                 enable_recompute, 
                 mlp_recompute:MLPRecomputeConfig,
                 strategy:StrategyConfig, 
                 system:SystemConfig, 
                 specific_name='') -> None:
        super().__init__(strategy, system, specific_name)
        self.layer_idx = layer_idx
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute  # for old version 
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        self.local_expert_num = self.config.expert_num // self.strategy.ep_size
        ffn_hidden_size = (self.config.moe_ffn_hidden_size if self.config.moe_ffn_hidden_size is not None 
                        else self.config.intermediate_size)
        intermediate_size = (
            2 * ffn_hidden_size
            if self.config.use_swiglu
            else ffn_hidden_size
        )
        self.mlp_recompute = mlp_recompute
        self.shared_expert = None
        if getattr(self.config, "moe_shared_expert_intermediate_size", None) is not None:
            self.shared_expert = MLP(
                    layer_idx=f"{layer_idx}-shareExpert",
                    config=self.config,
                    enable_recompute=enable_recompute, # for old version 
                    mlp_recompute_conf=mlp_recompute,
                    strategy=strategy,
                    system=system,
                    intermediate_size=self.config.moe_shared_expert_intermediate_size
                )

        GroupLinearCol_ = QuantizedGroupLinearCol if self.strategy.fp8 else GroupLinearCol
        GroupLinearRow_ = QuantizedGroupLinearRow if self.strategy.fp8 else GroupLinearRow
        
        self.router = Router(
                layer_idx=layer_idx,
                hidden_size=self.config.hidden_size,
                expert_num=self.config.expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.router_recompute,
                strategy=strategy,
                system=system,
            )
        self.permutation = Permutation(
                layer_idx=layer_idx,
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_pad_expert_input_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                capacity=self.config.capacity,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.permutation_recompute,
                strategy=strategy,
                system=system,
            )
        self.group_linear1 = GroupLinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
                local_expert_num=self.local_expert_num,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.linear_recompute,
                mode=self.config.group_linear_mode,
                strategy=strategy,
                system=system,
            )
        if self.strategy.fp8:
            # fp8
            if self.strategy.cache_groupgemm_col_fp8_inputs:
                self.group_linear1.linear.offload_inputs = self.strategy.offload_groupgemm_col_inputs
            else:
                self.group_linear1.quantizer.offload_inputs = self.strategy.offload_groupgemm_col_inputs
        else:
            # bf16
            self.group_linear1.offload_inputs = self.strategy.offload_groupgemm_col_inputs
        
        if self.config.use_swiglu:
            self.expert_activation_layer = Swiglu(
                    is_fused=self.strategy.use_fused_swiglu,
                    has_cached_inputs=False,
                    enable_recompute=mlp_recompute.linear_recompute,
                    strategy=strategy,
                    system=system,
                    is_weighted_silu= self.strategy.dispatch_probs
                )
        else:
            self.expert_activation_layer =Gelu(
                    has_cached_inputs=False,
                    enable_recompute=mlp_recompute.linear_recompute,
                    strategy=strategy,
                    system=system,
                )
        self.group_linear2 = GroupLinearRow_(
                layer_idx=layer_idx,
                input_size=ffn_hidden_size,
                output_size=self.config.hidden_size,
                local_expert_num=self.local_expert_num,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.linear_recompute,
                is_last_recompute = True,
                mode=self.config.group_linear_mode,
                use_bias=False,
                strategy=strategy,
                system=system,
            )
        self.unpermutation = UnPermutation(
                layer_idx=layer_idx,
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.permutation_recompute,
                strategy=strategy,
                system=system,
            )
        if self.unpermutation.enable_recompute and self.strategy.recompute_granularity == "selective_recompute":
            self.unpermutation.is_breakpoints = True

        if not (mlp_recompute.router_recompute and
                mlp_recompute.permutation_recompute and 
                mlp_recompute.linear_recompute and
                self.shared_expert.recompute_granularity == "full" if self.shared_expert else True):
            self.recompute_granularity = "submodule"
    
    def preprocess(self, input_info:InputOutputInfo):
        self.unpermutation.set_ori_shape(input_info.tensors[0].shape.copy())

    def forward(self, input_info:InputOutputInfo, path_debug_context:PathDebugContext):
        self.preprocess(input_info) 
        if self.shared_expert:
            shared_out = self.shared_expert(input_info, path_debug_context)
        probs = self.router(input_info, path_debug_context) # add router scores

        if self.strategy.dispatch_probs:
            permute_hidden_states = self.permutation(Input(tensors=[input_info.tensors[0], 
                                                                              probs.tensors[0]]),
                                                    path_debug_context) 
            grou1_out = self.group_linear1(permute_hidden_states, path_debug_context)
            act_out = self.expert_activation_layer(Input(tensors=[grou1_out.tensors[0], probs.tensors[0]]), path_debug_context)
            group2_out = self.group_linear2(act_out, path_debug_context)
            out = self.unpermutation(group2_out, path_debug_context)

        else:
            permute_hidden_states = self.permutation(Input(tensors=[input_info.tensors[0], 
                                                                probs.tensors[0]]),path_debug_context) # pass probs to Permutation to cache.
            grou1_out = self.group_linear1(permute_hidden_states, path_debug_context)
            act_out = self.expert_activation_layer(grou1_out, path_debug_context)
            group2_out = self.group_linear2(act_out, path_debug_context)
            out = self.unpermutation(Input(tensors=[group2_out.tensors[0], probs.tensors[0]]), path_debug_context)
           
        # FIXME(sherry):  add mul, routed_expert hidden_states * router scores
        if self.shared_expert:
            # return out + shared_out
            return AddFunction.apply(parent_model=self,
                                     enable_recompute=self.recompute_granularity == 'full_block',
                                     tensor_size1=out,
                                     tensor_size2=shared_out,
                                     path_debug_context=path_debug_context,
                                     name='SharedExpertAddFunction')
        return out 
       
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff)
#endregion 
