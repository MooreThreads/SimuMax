"""basic moe transformer module"""

from ..base_struct import (
    MetaModule,
    TensorSize,
    InputOutputInfo,
)
from ..config import StrategyConfig, SystemConfig
from .dense_module import Swiglu, Gelu


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
        self.expert_num = expert_num
        self.local_expert_num = expert_num // self.strategy.ep_size
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.hidden_size = hidden_size
        self.moe_dispatcher_policy = moe_dispatcher_policy
        # TODO: consider z-loss、aux-loss etc.

    @property
    def local_logits_size(self):
        assert self.input_info is not None, "Please set input info"
        b = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        ep_num = self.expert_num
        return b * seq_len * ep_num

    @property
    def output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, hidden_size))]
        )
        return output_info

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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
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
        self._model_info.weight_bytes = weight_numel * self.element_size
        self._model_info.grad_bytes = (
            self._model_info.weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self._model_info.weight_bytes * 2
        )
        self._model_info.state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.edp_size

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
    5.all_gahter feat-dim on tp group or token-dim etp group (fwd: all gather, bwd: reduce_scatter)
    """

    def __init__(
        self,
        expert_num: int,
        local_expert_num: int,
        topk: int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy

    @property
    def permuted_act_size(self):
        # only consider balanced case for now
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        token_num = self.topk * batch_size * seq_len
        return token_num * hidden_size

    @property
    def input_act_size(self):
        # only consider balanced case for now
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        part_seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel and self.strategy.etp_size > 1:
            seq_len = part_seq_len * self.strategy.etp_size
            # part_hidden_size = hidden_size // self.strategy.tp_size
        else:
            seq_len = part_seq_len
            # part_hidden_size = hidden_size

        output_info = InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(batch_size * seq_len * self.topk, hidden_size)
                ),  # permuted moe input
                TensorSize(
                    shape=(batch_size, part_seq_len, hidden_size)
                ),  # original moe input
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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
        elif self.moe_dispatcher_policy == "all2all" and (
            self.strategy.etp_size > 1 or self.strategy.ep_size > 1
        ):
            # gather the global distribution of tokens across all experts
            comm_size = (
                self.input_act_size * self.dtype_to_element_size[self.strategy.dtype]
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
            )
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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
            )
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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
        permutate1 for ep all2all
        permutate2 for mlp compute
        """
        permutate1_mem_accessed = (
            self.input_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype]
        permutate2_mem_accessed = (
            self.permuted_act_size + self.permuted_act_size
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
            fwd_op="default",
            bwd_grad_act_op="default",
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
        expert_num: int,
        local_expert_num: int,
        topk: int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy

    @property
    def act_size_before_combined(self):
        # only consider balanced case
        act_size = self.input_info.tensors[0].numel()
        return act_size

    @property
    def act_size_after_combined(self):
        # only consider balanced case
        act_size = self.output_info.tensors[0].numel()
        return act_size

    @property
    def output_info(self):
        # recover the original input
        output_info = InputOutputInfo(tensors=[self.input_info.tensors[1]])
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
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
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
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        """
        Mainly layout operators, ignore for now
        """
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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
        1.permutate1 for ep all2all
        2.permutate2 for mlp compute
        3.combine scores
        """
        # pylint: disable=invalid-name
        permutate1_mem_accessed = (
            2 * self.act_size_before_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]
        permutate2_and_combine_mem_accessed = (
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
            fwd_op="default",
            bwd_grad_act_op="default",
            bwd_grad_w_op="default",
            enable_recompute=self.enable_recompute,
        )


class GroupLinearCol(MetaModule):
    """Multi Expert Linear Layer, Suport column parallelism"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        assert output_size % self.strategy.etp_size == 0
        self.local_expert_num = local_expert_num
        self.input_size = input_size
        self.output_size = output_size // self.strategy.etp_size
        self.use_bias = use_bias  # for now unless
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[1].size(0)
        hidden_size = self.input_info.tensors[1].size(1)
        if self.strategy.enable_sequence_parallel:
            hidden_size *= self.strategy.etp_size
        return token_num * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        token_num = self.output_info.tensors[0].size(0)
        return token_num * self.output_size

    @property
    def output_info(self):
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
            self.micro_hidden_state_size * self.element_size
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.local_expert_num
            * self.input_size
            * self.output_size
            * self.element_size
        )
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.local_expert_num * self.input_size * self.output_size
        self._model_info.weight_bytes = weight_numel * self.element_size
        self._model_info.grad_bytes = (
            self._model_info.weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self._model_info.weight_bytes * 2
        )
        self._model_info.state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.edp_size

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
            * self.element_size
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size
        )

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
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


class GroupLinearRow(MetaModule):
    """Multi Expert Linear Layer, Suport row parallelism"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        assert input_size % self.strategy.etp_size == 0
        self.local_expert_num = local_expert_num
        self.input_size = input_size // self.strategy.etp_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute

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
        token_num = self.output_info.tensors[0].size(0)
        hidden_size = self.output_info.tensors[0].size(1)
        # hidden_size = self.output_info.tensors[0].size(2)
        return token_num * hidden_size

    @property
    def output_info(self):
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
        # tp comm is in UnPermuation
        pass

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.element_size
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.input_size
            * self.output_size
            * self.local_expert_num
            * self.element_size
        )
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size * self.local_expert_num
        self._model_info.weight_bytes = weight_numel * self.element_size
        self._model_info.grad_bytes = (
            self._model_info.weight_bytes
            if not self.strategy.use_fp32_accum_grad
            else self._model_info.weight_bytes * 2
        )
        self._model_info.state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        if self.strategy.zero_state >= 1:
            self._model_info.state_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.edp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.edp_size

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
            * self.element_size
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size
        )

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

    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"local_expert_num={self.local_expert_num},"
            f"use_bias={self.use_bias}"
        )
        return repr_info


class ExpertMLP(MetaModule):
    """Expert MLP Layer"""

    def __init__(self, config, enable_recompute, strategy, system) -> None:
        super().__init__(strategy, system)
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        self.local_expert_num = self.config.expert_num // self.strategy.ep_size
        intermediate_size = (
            2 * self.config.intermediate_size
            if self.config.use_swiglu
            else self.config.intermediate_size
        )
        self.register_module(
            Router(
                hidden_size=self.config.hidden_size,
                expert_num=self.config.expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                strategy=strategy,
                system=system,
            )
        )
        self.register_module(
            Permutation(
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                strategy=strategy,
                system=system,
            )
        )
        self.register_module(
            GroupLinearCol(
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
                local_expert_num=self.local_expert_num,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                strategy=strategy,
                system=system,
            )
        )
        if self.config.use_swiglu:
            self.register_module(
                Swiglu(
                    is_fused=self.strategy.use_fused_swiglu,
                    has_cached_inputs=False,
                    enable_recompute=enable_recompute,
                    strategy=strategy,
                    system=system,
                )
            )
        else:
            self.register_module(
                Gelu(
                    has_cached_inputs=False,
                    enable_recompute=enable_recompute,
                    strategy=strategy,
                    system=system,
                )
            )  # TODO: sub module enable recompute

        self.register_module(
            GroupLinearRow(
                input_size=self.config.intermediate_size,
                output_size=self.config.hidden_size,
                local_expert_num=self.local_expert_num,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                use_bias=False,
                strategy=strategy,
                system=system,
            )
        )
        self.register_module(
            UnPermutation(
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                strategy=strategy,
                system=system,
            )
        )
