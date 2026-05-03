"""Shared model/debug/result data structures for SimuMax core."""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Tuple

from simumax.core.tensor import TensorSize
from simumax.core.utils import (
    human_readable_bytes,
    human_readable_nums,
    human_readable_times,
    path_convert_to_str,
)


class RecomputeStatus:
    NO_RECOMPUTE = "no_recompute"
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass
class InputOutputInfo:
    tensors: List[TensorSize]

    def __repr__(self) -> str:
        tensor_info = ",".join(
            [f"Tensor {i}: {str(tensor)}" for i, tensor in enumerate(self.tensors)]
        )
        return f"InputInfo: {tensor_info}"

    @property
    def shapes(self):
        return [tensor.shape for tensor in self.tensors]

    def __getitem__(self, index: int) -> TensorSize:
        return self.tensors[index]


@dataclass
class ModuleComputeInfo:
    """Record compute/flops/memory access information for a module."""

    fwd_flops: int = 0
    recompute_flops: int = 0
    bwd_grad_w_flops: int = 0
    bwd_grad_act_flops: int = 0

    fwd_accessed_mem: int = 0
    recompute_accessed_mem: int = 0
    bwd_grad_w_accessed_mem: int = 0
    bwd_grad_act_accessed_mem: int = 0

    @property
    def bwd_flops(self):
        return self.bwd_grad_w_flops + self.bwd_grad_act_flops

    @property
    def bwd_accessed_mem(self):
        return self.bwd_grad_w_accessed_mem + self.bwd_grad_act_accessed_mem

    def get_all_flops(self):
        return [self.fwd_flops, self.bwd_grad_act_flops, self.bwd_grad_w_flops]

    def get_all_accessed_mem(self):
        return [self.fwd_accessed_mem, self.bwd_grad_act_accessed_mem, self.bwd_grad_w_accessed_mem]

    def _format_repr_info(self):
        attributes = {
            "fwd_flops": self.fwd_flops,
            "recompute_flops": self.recompute_flops,
            "bwd_flops": self.bwd_flops,
            "bwd_grad_w_flops": self.bwd_grad_w_flops,
            "bwd_grad_act_flops": self.bwd_grad_act_flops,
            "fwd_accessed_mem": self.fwd_accessed_mem,
            "recompute_accessed_mem": self.recompute_accessed_mem,
            "bwd_accessed_mem": self.bwd_accessed_mem,
            "bwd_grad_w_accessed_mem": self.bwd_grad_w_accessed_mem,
            "bwd_grad_act_accessed_mem": self.bwd_grad_act_accessed_mem,
        }
        repr_info = []
        for key, value in attributes.items():
            if "flops" in key:
                formatted_value = human_readable_nums(value)
            elif "mem" in key:
                formatted_value = human_readable_bytes(value)
            else:
                formatted_value = f"{value:.4f}"
            repr_info.append(f"\t{key}={formatted_value};")
        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ModuleComputeInfo(\n{self._format_repr_info()}\n)"

    def __add__(self, other):
        if not isinstance(other, ModuleComputeInfo):
            raise ValueError(
                f"Unsupported operand type for +: ModuleComputeInfo and {type(other)}"
            )
        return ModuleComputeInfo(
            fwd_flops=self.fwd_flops + other.fwd_flops,
            recompute_flops=self.recompute_flops + other.recompute_flops,
            bwd_grad_w_flops=self.bwd_grad_w_flops + other.bwd_grad_w_flops,
            bwd_grad_act_flops=self.bwd_grad_act_flops + other.bwd_grad_act_flops,
            fwd_accessed_mem=self.fwd_accessed_mem + other.fwd_accessed_mem,
            recompute_accessed_mem=self.recompute_accessed_mem + other.recompute_accessed_mem,
            bwd_grad_w_accessed_mem=self.bwd_grad_w_accessed_mem + other.bwd_grad_w_accessed_mem,
            bwd_grad_act_accessed_mem=self.bwd_grad_act_accessed_mem + other.bwd_grad_act_accessed_mem,
        )


@dataclass
class ActivationInfo:
    """Record activation/cache/peak memory information for a module."""

    activation_mem_cache: int = 0
    fwd_peak_mem_no_cache: int = 0
    fwd_peak_point = ""

    bwd_peak_mem_no_cache = 0
    bwd_peak_point = ""

    cache_for_bwd_mem: int = 0
    fwd_idx = 0
    fwd_total_activation_mem_cache: int = 0

    @property
    def fwd_peak_mem(self):
        return self.fwd_peak_mem_no_cache

    @property
    def total_activation_mem_cache(self):
        return self.activation_mem_cache

    @property
    def bwd_peak_mem(self):
        return self.bwd_peak_mem_no_cache

    def to_dict(self):
        data_dict = asdict(self)
        data_dict["fwd_peak_mem"] = self.fwd_peak_mem
        data_dict["bwd_peak_mem"] = self.bwd_peak_mem
        data_dict["peak_stage"] = "forward" if self.fwd_peak_mem > self.bwd_peak_mem else "backward"
        data_dict["peak_path"] = self.fwd_peak_point if self.fwd_peak_mem > self.bwd_peak_mem else self.bwd_peak_point
        data_dict["peak_mem"] = max(self.fwd_peak_mem, self.bwd_peak_mem)
        return data_dict

    def _format_repr_info(self):
        attributes = {
            "activation_mem_cache": self.activation_mem_cache,
            "fwd_peak_point": self.fwd_peak_point,
            "fwd_peak_mem_no_cache": self.fwd_peak_mem_no_cache,
            "fwd_peak_mem": self.fwd_peak_mem,
            "bwd_peak_point": self.bwd_peak_point,
            "bwd_peak_mem_no_cache": self.bwd_peak_mem_no_cache,
            "bwd_peak_mem": self.bwd_peak_mem,
        }
        repr_info = []
        for key, value in attributes.items():
            formatted_value = human_readable_bytes(value) if any(x in key for x in ["mem", "bytes", "cache"]) else value
            repr_info.append(f"\t{key}={formatted_value};")
        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ActivationInfo(\n{self._format_repr_info()}\n)"


@dataclass
class PointDebugInfo:
    """Debug info for one memory-debug collection point."""

    point: str = ""
    parent_path_list: List[str] = None
    next_parent_path_to_collect: List[str] = None
    prev_cache_mem: int = 0
    fwd_peak_no_cache_mem: int = 0
    bwd_peak_no_cache_mem: int = 0

    @property
    def fwd_peak_mem(self):
        return self.fwd_peak_no_cache_mem + self.prev_cache_mem

    @property
    def bwd_peak_mem(self):
        return self.bwd_peak_no_cache_mem + self.prev_cache_mem

    @property
    def parent_path(self):
        return path_convert_to_str(self.parent_path_list)

    @property
    def next_parent_path(self):
        return path_convert_to_str(self.next_parent_path_to_collect)

    def valid_debug_info(self):
        return


@dataclass
class PathDebugContext:
    """Manage memory-debug context across a whole workflow path."""

    point_datas: Dict[str, PointDebugInfo] = None
    point_datas_with_recomp: Dict[str, PointDebugInfo] = None
    target_point: List[str] = None
    path_list: list = None

    def get_point_datas(self, enable_recompute=False):
        return self.point_datas if not enable_recompute else self.point_datas_with_recomp

    def get_next_parent_to_point(self, enable_recompute=False):
        res = {}
        data = self.get_point_datas(enable_recompute=enable_recompute)
        if not data:
            return res
        for _, v in data.items():
            if v.next_parent_path not in res:
                res[v.next_parent_path] = []
            res[v.next_parent_path].append(v)
        return res

    @property
    def parent(self):
        path_name = ""
        if len(self.path_list) > 1:
            path_name = path_convert_to_str(self.path_list[:-1])
        return path_name

    @property
    def current(self):
        if len(self.path_list) == 0:
            return ""
        return self.path_list[-1]

    @property
    def path(self):
        return path_convert_to_str(self.path_list)


@dataclass
class ModuleMemoryInfo:
    """Record weight/grad/state memory usage of a module."""

    weight_numel: int = 0
    dense_weight_bytes: int = 0
    dense_grad_bytes: int = 0
    dense_state_bytes: int = 0
    moe_weight_numel: int = 0
    moe_weight_bytes: int = 0
    moe_grad_bytes: int = 0
    moe_state_bytes: int = 0
    te_dummy_wgrad_shapes: Set[Tuple[int, int, int]] = field(default_factory=set)

    @property
    def all(self):
        return (
            self.dense_weight_bytes
            + self.dense_grad_bytes
            + self.dense_state_bytes
            + self.moe_weight_bytes
            + self.moe_grad_bytes
            + self.moe_state_bytes
            + self.te_dummy_wgrad_bytes
        )

    @property
    def te_dummy_wgrad_bytes(self):
        return sum(rows * cols * elem_size for rows, cols, elem_size in self.te_dummy_wgrad_shapes)

    @property
    def all_state_bytes(self):
        return self.dense_state_bytes + self.moe_state_bytes

    @property
    def all_weight_bytes(self):
        return self.dense_weight_bytes + self.moe_weight_bytes

    @property
    def all_weight_numel(self):
        return self.weight_numel + self.moe_weight_numel

    @property
    def all_grad_bytes(self):
        return self.dense_grad_bytes + self.moe_grad_bytes

    def _format_repr_info(self):
        attributes = {
            "all": self.all,
            "weight_bytes": self.dense_weight_bytes,
            "grad_bytes": self.dense_grad_bytes,
            "state_bytes": self.dense_state_bytes,
            "moe_weight_bytes": self.moe_weight_bytes,
            "moe_grad_bytes": self.moe_grad_bytes,
            "moe_state_bytes": self.moe_state_bytes,
            "te_dummy_wgrad_bytes": self.te_dummy_wgrad_bytes,
        }
        repr_info = []
        for key, value in attributes.items():
            repr_info.append(f"\t{key}={human_readable_bytes(value)};")
        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ModuleMemoryInfo(\n{self._format_repr_info()}\n)"

    def __add__(self, other):
        if not isinstance(other, ModuleMemoryInfo):
            raise ValueError(
                f"Unsupported operand type for +: ModuleMemoryInfo and {type(other)}"
            )
        return ModuleMemoryInfo(
            weight_numel=self.weight_numel + other.weight_numel,
            dense_weight_bytes=self.dense_weight_bytes + other.dense_weight_bytes,
            dense_grad_bytes=self.dense_grad_bytes + other.dense_grad_bytes,
            dense_state_bytes=self.dense_state_bytes + other.dense_state_bytes,
            moe_weight_numel=self.moe_weight_numel + other.moe_weight_numel,
            moe_weight_bytes=self.moe_weight_bytes + other.moe_weight_bytes,
            moe_grad_bytes=self.moe_grad_bytes + other.moe_grad_bytes,
            moe_state_bytes=self.moe_state_bytes + other.moe_state_bytes,
            te_dummy_wgrad_shapes=self.te_dummy_wgrad_shapes | other.te_dummy_wgrad_shapes,
        )


@dataclass
class ModuleCostInfo:
    """Record time-cost information of a module."""

    fwd_compute_time: int = 0
    recompute_compute_time: int = 0
    bwd_grad_w_time: int = 0
    bwd_grad_act_time: int = 0

    fwd_net_time: int = 0
    recompute_net_time: int = 0
    bwd_grad_w_net_time: int = 0
    bwd_grad_act_net_time: int = 0

    fwd_net_exposed_time: int = 0
    recompute_net_exposed_time: int = 0
    bwd_net_exposed_time: int = 0

    @property
    def fwd_time(self):
        return self.fwd_compute_time + self.fwd_net_exposed_time

    @property
    def all_time(self):
        return self.fwd_time + self.fwd_net_time + self.bwd_time + self.bwd_net_time

    @property
    def recompute_time(self):
        return self.recompute_compute_time + self.recompute_net_exposed_time

    @property
    def bwd_compute_time(self):
        return self.bwd_grad_w_time + self.bwd_grad_act_time

    @property
    def bwd_time(self):
        return self.bwd_grad_w_time + self.bwd_grad_act_time + self.bwd_net_exposed_time

    @property
    def bwd_net_time(self):
        return self.bwd_grad_w_net_time + self.bwd_grad_act_net_time

    @property
    def net_time(self):
        return self.fwd_net_time + self.bwd_net_time + self.recompute_net_time

    def get_all_costs(self):
        return [self.fwd_time, self.bwd_grad_act_time, self.bwd_grad_w_time]

    def _format_repr_info(self):
        attributes = {
            "fwd_compute_time": self.fwd_compute_time,
            "fwd_net_time": self.fwd_net_time,
            "fwd_net_exposed_time": self.fwd_net_exposed_time,
            "recompute_compute_time": self.recompute_compute_time,
            "recompute_net_time": self.recompute_net_time,
            "recompute_net_exposed_time": self.recompute_net_exposed_time,
            "bwd_compute_time": self.bwd_compute_time,
            "bwd_grad_w_time": self.bwd_grad_w_time,
            "bwd_grad_act_time": self.bwd_grad_act_time,
            "bwd_net_time": self.bwd_net_time,
            "bwd_net_exposed_time": self.bwd_net_exposed_time,
            "total": self.fwd_time + self.recompute_time + self.bwd_time,
        }
        repr_info = []
        for key, value in attributes.items():
            repr_info.append(f"\t{key}={human_readable_times(value)};")
        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ModuleCostInfo(\n{self._format_repr_info()}\n)"

    def __add__(self, other):
        if not isinstance(other, ModuleCostInfo):
            raise ValueError(
                f"Unsupported operand type for +: ModuleCostInfo and {type(other)}"
            )
        return ModuleCostInfo(
            fwd_compute_time=self.fwd_compute_time + other.fwd_compute_time,
            recompute_compute_time=self.recompute_compute_time + other.recompute_compute_time,
            bwd_grad_w_time=self.bwd_grad_w_time + other.bwd_grad_w_time,
            bwd_grad_act_time=self.bwd_grad_act_time + other.bwd_grad_act_time,
            fwd_net_time=self.fwd_net_time + other.fwd_net_time,
            recompute_net_time=self.recompute_net_time + other.recompute_net_time,
            bwd_grad_w_net_time=self.bwd_grad_w_net_time + other.bwd_grad_w_net_time,
            bwd_grad_act_net_time=self.bwd_grad_act_net_time + other.bwd_grad_act_net_time,
            fwd_net_exposed_time=self.fwd_net_exposed_time + other.fwd_net_exposed_time,
            recompute_net_exposed_time=self.recompute_net_exposed_time + other.recompute_net_exposed_time,
            bwd_net_exposed_time=self.bwd_net_exposed_time + other.bwd_net_exposed_time,
        )


class Result:
    """A simple wrapper around a result dict."""

    def __init__(self, result: dict) -> None:
        self.data = result

    def get(self, key: str):
        return self.data.get(key, None)
