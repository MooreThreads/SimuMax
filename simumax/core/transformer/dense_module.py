"""basic dense transformer module"""

from copy import deepcopy
from ..base_struct import MetaModule, TensorSize, InputOutputInfo
from ..config import ModelConfig, StrategyConfig, SystemConfig


class LinearCol(MetaModule):
    """Support for column parallel linear layer"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        assert output_size % self.strategy.tp_size == 0
        self.input_size = input_size
        self.output_size = output_size // self.strategy.tp_size
        self.use_bias = use_bias  # FIXME(for now unless)
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute

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
            seq_len *= self.strategy.tp_size
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        # hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * self.output_size

    @property
    def output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.enable_sequence_parallel:
            seq_len *= self.strategy.tp_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.output_size))]
        )
        return output_info

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert self.input_size == hidden_size

    def _comp_leaf_intra_net_info(self):
        # 1.FWD
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            # fwd compute with sp
            comm_size = (
                self.micro_hidden_state_size
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
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
            )

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.element_size
        )
        if self.strategy.enable_sequence_parallel:
            # Note: sp only cache the activation slice
            self._act_info.activation_mem_cache /= self.strategy.tp_size
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = self.input_size * self.output_size * self.element_size
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size
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
            self._model_info.state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.dp_size

    def _comp_leaf_flops_info(self):
        base_flops = 2 * self.micro_hidden_state_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = self.input_size * self.output_size * self.element_size
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

    def extra_repr(self):
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"use_bias={self.use_bias},"
            f"enable_recompute={self.enable_recompute}"
        )
        return repr_info


class LinearRow(MetaModule):
    """support row parallel linear layer"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        assert input_size % self.strategy.tp_size == 0
        self.input_size = input_size // self.strategy.tp_size
        self.output_size = output_size
        self.use_bias = use_bias  # FIXME(for now unless)
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

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        hidden_size = self.output_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel:
            seq_len *= self.strategy.tp_size
        return batch_size * seq_len * hidden_size

    @property
    def output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.enable_sequence_parallel:
            seq_len /= self.strategy.tp_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.output_size))]
        )
        return output_info

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(2)
        assert (
            self.input_size == hidden_size
        ), f"input_size: {self.input_size} vs hidden_size: {hidden_size}"
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    def _comp_leaf_intra_net_info(self):
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
            )
        else:
            # identity operation
            pass

    def _comp_leaf_act_info_impl(self):
        weight_size = self.input_size * self.output_size * self.element_size
        input_size = self.micro_hidden_state_size * self.element_size
        output_size = self.micro_output_grad_size * self.element_size

        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.element_size
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache -= (
                self.micro_hidden_state_size * self.element_size
            )
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size
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
            self._model_info.state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.dp_size

    def _comp_leaf_flops_info(self):
        base_flops = 2 * self.micro_hidden_state_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = self.input_size * self.output_size * self.element_size
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
            f"use_bias={self.use_bias},"
            f"enable_recompute={self.enable_recompute}"
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
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

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
            self._model_info.state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.dp_size

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
    ) -> None:
        super().__init__(strategy, system)
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
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute

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
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def output_info(self):
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

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)

        q_size = batch_size * self.head_num * seq_len * self.head_size
        # repeat kv
        k_size = q_size  # batch_size * self.kv_head_num * seq_len * self.head_size
        v_size = q_size  # batch_size * self.kv_head_num * seq_len * self.head_size
        lse_size = batch_size * self.head_num * seq_len
        output_grad_size = batch_size * seq_len * hidden_size
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
        elif self.use_math_sdp and self.system.accelerator.backend == "musa":
            # torch_musa sdp kernel reuse the output grad memory
            bwd_soft_factor -= 1
        self._act_info.bwd_peak_prev_cache_mem = (q_size + k_size) * self.element_size
        self._act_info.bwd_peak_mem_no_cache = (
            bwd_soft_factor * softmax_size * self.element_size
        )

    def _comp_leaf_model_info_impl(self):
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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
        output_grad_size = batch_size * seq_len * hidden_size
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


class Swiglu(MetaModule):
    """Activation function Swish-Gated Linear Unit"""

    def __init__(
        self,
        is_fused: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.is_fused = is_fused
        self.enable_recompute = enable_recompute
        self.has_cached_inputs = has_cached_inputs

    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H] or [S, H]
        input_numel = self.input_info.tensors[0].numel()
        return input_numel

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        input_numel = self.output_info.tensors[0].numel()
        return input_numel

    @property
    def output_info(self):
        hidden_size = self.input_info.tensors[0].size(-1)
        assert hidden_size % 2 == 0, "hidden size should be even"
        output_size_list = list(self.input_info.tensors[0].shape[:-1]) + [
            hidden_size // 2,
        ]
        output_tensor = [TensorSize(shape=tuple(output_size_list))]
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

    def _comp_leaf_model_info_impl(self):
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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

    @property
    def output_info(self):
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
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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


class Attention(MetaModule):
    """Full Attention Layer"""

    def __init__(
        self,
        config: ModelConfig,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute
        output_size = (
            self.config.head_num * self.config.head_size
            + 2 * self.config.kv_head_num * self.config.head_size
        )
        only_enable_sdp = False
        if self.strategy.recompute_granularity == "sdp_only":
            self.recompute_granularity = "submodule"
            only_enable_sdp = True

        self.register_module(
            LinearCol(
                input_size=self.config.hidden_size,
                output_size=output_size,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=(self.enable_recompute and not only_enable_sdp),
                strategy=strategy,
                system=system,
            )
        )
        self.register_module(
            CoreAttention(
                head_size=config.head_size,
                head_num=config.head_num,
                kv_head_num=config.kv_head_num,
                use_math_sdp=self.strategy.use_math_sdp,
                use_flash_sdp=self.strategy.use_flash_sdp,
                has_cached_inputs=False,
                enable_recompute=self.enable_recompute,
                strategy=strategy,
                system=system,
            )
        )
        self.register_module(
            LinearRow(
                input_size=self.config.hidden_size,
                output_size=self.config.hidden_size,
                has_cached_inputs=False,
                enable_recompute=(self.enable_recompute and not only_enable_sdp),
                use_bias=False,
                strategy=strategy,
                system=system,
            )
        )

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
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        output_info = deepcopy(self.input_info)
        output_info.tensors = [TensorSize(shape=(batch_size, seq_len, hidden_size))]
        return output_info


class MLP(MetaModule):
    """normal mlp layers"""

    def __init__(self, config, enable_recompute, strategy, system) -> None:
        super().__init__(strategy, system)
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute
        intermediate_size = (
            2 * self.config.intermediate_size
            if self.config.use_swiglu
            else self.config.intermediate_size
        )
        self.register_module(
            LinearCol(
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
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
            )

        self.register_module(
            LinearRow(
                input_size=self.config.intermediate_size,
                output_size=self.config.hidden_size,
                has_cached_inputs=False,
                enable_recompute=enable_recompute,
                use_bias=False,
                strategy=strategy,
                system=system,
            )
        )


class Embedding(MetaModule):
    """
    Parallel Embedding Layer
    """

    def __init__(self, hidden_size, vocab_size, strategy, system) -> None:
        super().__init__(strategy, system)
        assert vocab_size % self.strategy.tp_size == 0
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size // self.strategy.tp_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        batch_size = self.output_info.tensors[0].size(0)
        seq_len = self.output_info.tensors[0].size(1)
        hidden_size = self.output_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def output_info(self):
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
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
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
            )

    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        input_size = batch_size * seq_len * 4  # int32?
        weight_size = self.vocab_size * self.hidden_size
        output_size = batch_size * seq_len * self.hidden_size * self.element_size
        # FIXME: Aggregation will cause some model weight to be added repeatedly,
        # resulting in an overestimation of the peak
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + weight_size
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = weight_size
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.vocab_size * self.hidden_size
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
            self._model_info.state_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 2:
            self._model_info.grad_bytes /= self.strategy.dp_size
        if self.strategy.zero_state >= 3:
            self._model_info.weight_bytes /= self.strategy.dp_size

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

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = 0
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
        repr_info = f"hidden_size={self.hidden_size}," f"vocab_size={self.vocab_size}"
        return repr_info


class ParallelCE(MetaModule):
    """
    input_parallel  -> VocabParallelCrossEntropy(fp32/fp16)
    """

    # def __init__(self, strategy, system) -> None:
    #     super().__init__(strategy, system)

    @property
    def output_info(self):
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
            )
            # all_reduce for predicted_logits [b x s]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
            )
            # all reduce for sum_exp_logits [b x s]
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
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
        self._act_info.fwd_peak_mem_no_cache = ce_fwd_peak_no_cache
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = ce_fwd_peak_no_cache
        self._act_info.bwd_peak_prev_cache_mem = 0
        self._act_info_with_recomp = self._act_info

    def _comp_leaf_model_info_impl(self):
        self._model_info.weight_bytes = 0
        self._model_info.grad_bytes = 0
        self._model_info.state_bytes = 0

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

        self._compute_info.fwd_accessed_mem = (
            logtis_size + batch_size * seq_len
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # max
        self._compute_info.fwd_accessed_mem += (
            logtis_size + batch_size * seq_len + logtis_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # logits - max
        self._compute_info.fwd_accessed_mem = (
            logtis_size * 2
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # exp
        self._compute_info.fwd_accessed_mem = (
            logtis_size + batch_size
        ) * self.dtype_to_element_size[
            "fp32"
        ]  # sum
        self._compute_info.fwd_accessed_mem = (
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
