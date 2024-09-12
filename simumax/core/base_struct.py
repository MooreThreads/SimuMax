"""Basic data structures for simumax"""

from copy import deepcopy
from dataclasses import dataclass
from abc import ABC
from typing import List, Tuple, Dict

from .utils import get_point_name, path_convert_to_str, to_json_string
from .utils import human_readable_bytes, human_readable_nums, human_readable_times


@dataclass
class TensorSize:
    """record the shape of the tensor"""

    shape: Tuple[int, ...]

    def size(self, index: int) -> int:
        """
        Get the size at the specified index in the tuple.
        """
        if index < 0:
            index = len(self.shape) + index

        if 0 <= index < len(self.shape) or index == -1:
            shape = self.shape[index]
        else:
            raise IndexError(
                f"Index {index} is out of range for size tuple {self.shape}"
            )
        return shape

    def numel(self) -> int:
        if len(self.shape) == 0:
            return 0
        res = 1
        for x in self.shape:
            res *= x
        return res


@dataclass
class InputOutputInfo:
    tensors: List[TensorSize]

    def __repr__(self) -> str:
        tensor_info = "\n".join(
            [f"Tensor {i}: {tensor.shape}" for i, tensor in enumerate(self.tensors)]
        )
        return f"InputInfo:\n{tensor_info}"


@dataclass
class ModuleComputeInfo:
    """
    ModuleComputeInfo is used to record the compute usage of the module.
    """

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
            recompute_accessed_mem=self.recompute_accessed_mem
            + other.recompute_accessed_mem,
            bwd_grad_w_accessed_mem=self.bwd_grad_w_accessed_mem
            + other.bwd_grad_w_accessed_mem,
            bwd_grad_act_accessed_mem=self.bwd_grad_act_accessed_mem
            + other.bwd_grad_act_accessed_mem,
        )


@dataclass
class ActivationInfo:
    """
    ActivationInfo is used to record the memory usage of the activation.
    """

    activation_mem_cache: int = 0  # memory need to cached for backward
    fwd_peak_mem_no_cache: int = 0  # Operation need memory without cache at peak point
    fwd_peak_prev_cache_mem: int = 0  # before peak point, the memory need to cache
    fwd_peak_point = ""  # the peak point of the forward memory usage

    bwd_peak_mem_no_cache = 0
    bwd_peak_prev_cache_mem: int = 0
    bwd_peak_point = ""

    @property
    def fwd_peak_mem(self):
        return self.fwd_peak_prev_cache_mem + self.fwd_peak_mem_no_cache

    @property
    def bwd_peak_mem(self):
        return self.bwd_peak_prev_cache_mem + self.bwd_peak_mem_no_cache

    def _format_repr_info(self):
        attributes = {
            "activation_mem_cache": self.activation_mem_cache,
            "fwd_peak_point": self.fwd_peak_point,
            "fwd_peak_mem_no_cache": self.fwd_peak_mem_no_cache,
            "fwd_peak_prev_cache_mem": self.fwd_peak_prev_cache_mem,
            "fwd_peak_mem": self.fwd_peak_mem,
            "bwd_peak_point": self.bwd_peak_point,
            "bwd_peak_mem_no_cache": self.bwd_peak_mem_no_cache,
            "bwd_peak_prev_cache_mem": self.bwd_peak_prev_cache_mem,
            "bwd_peak_mem": self.bwd_peak_mem,
        }

        repr_info = []
        for key, value in attributes.items():
            if any(x in key for x in ["mem", "bytes", "cache"]):
                formatted_value = human_readable_bytes(value)
            else:
                formatted_value = value

            repr_info.append(f"\t{key}={formatted_value};")

        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ActivationInfo(\n{self._format_repr_info()}\n)"


@dataclass
class PointDebugInfo:
    """
    PointDebugInfo is used to record the debug info of the point for memory model.
    """

    point: str = ""
    parent_path_list: List[str] = None  # Parent path as list
    next_parent_path_to_collect: List[str] = None  # Next parent path to collect

    # before peak point, the memory need to cache on parent module compute path
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
        """
        next parent path need to collect memory info
        """
        return path_convert_to_str(self.next_parent_path_to_collect)

    def valid_debug_info(self):
        # assert not self.parent_path_list, "parent_list should be empty"
        return


@dataclass
class PathDebugContext:
    """
    PathDebugContext is used to manage the debug infos for memory in whole workflow.
    """

    point_datas: Dict[str, PointDebugInfo] = None  # path -> PointDebugInfo
    point_datas_with_recomp: Dict[str, PointDebugInfo] = None
    target_point: List[str] = None  # target point to collect mem info
    path_list: list = None  # current path list

    def get_point_datas(self, enable_recompute=False):
        data = (
            self.point_datas if not enable_recompute else self.point_datas_with_recomp
        )
        return data

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
        path_name = path_convert_to_str(self.path_list)
        return path_name


@dataclass
class ModuleMemoryInfo:
    """
    ModuleMemoryInfo is used to record the memory usage of the module.
    The memory usage is divided into three parts:
    1. Weight memory usage
    2. Gradient memory usage
    3. State memory usage
    """

    weight_bytes: int = 0
    grad_bytes: int = 0
    state_bytes: int = 0

    @property
    def all(self):
        return self.weight_bytes + self.grad_bytes + self.state_bytes

    def _format_repr_info(self):
        attributes = {
            "all": self.all,
            "weight_bytes": self.weight_bytes,
            "grad_bytes": self.grad_bytes,
            "state_bytes": self.state_bytes,
        }

        repr_info = []
        for key, value in attributes.items():
            formatted_value = human_readable_bytes(value)

            repr_info.append(f"\t{key}={formatted_value};")

        return "\n".join(repr_info)

    def __repr__(self) -> str:
        return f"ModuleMemoryInfo(\n{self._format_repr_info()}\n)"

    def __add__(self, other):
        if not isinstance(other, ModuleMemoryInfo):
            raise ValueError(
                f"Unsupported operand type for +: ModuleMemoryInfo and {type(other)}"
            )
        return ModuleMemoryInfo(
            weight_bytes=self.weight_bytes + other.weight_bytes,
            grad_bytes=self.grad_bytes + other.grad_bytes,
            state_bytes=self.state_bytes + other.state_bytes,
        )


@dataclass
class ModuleCostInfo:
    """
    ModuleCostInfo is used to record the time cost of the module.
    """

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
            formatted_value = human_readable_times(value)

            repr_info.append(f"\t{key}={formatted_value};")

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
            recompute_compute_time=self.recompute_compute_time
            + other.recompute_compute_time,
            bwd_grad_w_time=self.bwd_grad_w_time + other.bwd_grad_w_time,
            bwd_grad_act_time=self.bwd_grad_act_time + other.bwd_grad_act_time,
            fwd_net_time=self.fwd_net_time + other.fwd_net_time,
            recompute_net_time=self.recompute_net_time + other.recompute_net_time,
            bwd_grad_w_net_time=self.bwd_grad_w_net_time + other.bwd_grad_w_net_time,
            bwd_grad_act_net_time=self.bwd_grad_act_net_time
            + other.bwd_grad_act_net_time,
            fwd_net_exposed_time=self.fwd_net_exposed_time + other.fwd_net_exposed_time,
            recompute_net_exposed_time=self.recompute_net_exposed_time
            + other.recompute_net_exposed_time,
            bwd_net_exposed_time=self.bwd_net_exposed_time + other.bwd_net_exposed_time,
        )


class MetaModule(ABC):
    """
    Assume that there are two types of modules:
    1. The most basic module that does not have children modules
    2. A module composed of children modules, except for children modules,
       there are no other calculations
    """

    dtype_to_element_size = {"fp32": 4, "fp16": 2, "bf16": 2}

    def __init__(self, strategy, system) -> None:
        super().__init__()
        self.strategy = strategy
        self.system = system

        self.children_ordered_module = []
        self.default_dtype = strategy.dtype
        self._init_strategy = False
        self.input_info = None
        self.enable_recompute = False
        self.recompute_granularity = "full"
        self._reset_infos()

    def _reset_infos(self):
        self._act_info = ActivationInfo()
        self._act_info_with_recomp = ActivationInfo()
        self._model_info = ModuleMemoryInfo()
        self._compute_info = ModuleComputeInfo()
        self._cost_info = ModuleCostInfo()
        self.path_debug_context = None
        self.parent = None
        self.current = None
        self._info_ready = False

    def register_module(self, sub_module):
        self.children_ordered_module.append(sub_module)

    def set_dtype(self, dtype: str):
        assert dtype in ["fp32", "fp16", "bf16"]
        self.dtype = dtype

    @property
    def element_size(self):
        dtype = self.default_dtype
        if getattr(self, "dtype", False):
            dtype = self.dtype
        return self.dtype_to_element_size[dtype]

    # =========================
    # Basic Compute Related
    # =========================
    def compute_end2end_time(self, compute_time, mem_time):
        return self.system.compute_end2end_time(compute_time, mem_time)

    def all_input_element_num(self):
        res = 0
        for x in self.input_info.tensors:
            res += x.numel() * self.element_size
        return res

    def set_input_state_info(self, input_info: InputOutputInfo):
        self.input_info = deepcopy(input_info)

    def set_path_debug_context(self, path_debug_context: PathDebugContext):
        self.path_debug_context = path_debug_context

    @property
    def output_info(self):
        raise NotImplementedError

    # =========================
    # Pre/Post Porcess Related
    # =========================
    def _pre_op(self):
        pass

    def _post_op(self):
        pass

    # =========================
    # Memory Related
    # =========================
    def _comp_submod_act_info_impl(
        self, act_info: ActivationInfo, enable_recompute=False
    ):
        # module peak mem = prev_cache + cur_peak
        cur_cache_mem = 0
        cur_fwd_peak_prev_cache_mem = 0
        cur_bwd_peak_prev_cache_mem = 0
        cur_fwd_peak_mem_no_cache = 0
        cur_bwd_cache_mem_no_cache = 0
        max_fwd_peak_mem = 0
        max_bwd_peak_mem = 0
        fwd_peak_point = None
        bwd_peak_point = None
        for idx, module in enumerate(self.children_ordered_module):
            cur_act_info = (
                module.get_act_info()
                if not enable_recompute
                else module.get_act_info_with_recomp()
            )
            cur_fwd_peak_mem = cur_cache_mem + cur_act_info.fwd_peak_mem
            cur_bwd_peak_mem = cur_cache_mem + cur_act_info.bwd_peak_mem

            current_module = "(" + str(idx) + ")" + module.__class__.__name__
            if cur_fwd_peak_mem > max_fwd_peak_mem:
                cur_fwd_peak_mem_no_cache = cur_act_info.fwd_peak_mem_no_cache
                cur_fwd_peak_prev_cache_mem = (
                    cur_cache_mem + cur_act_info.fwd_peak_prev_cache_mem
                )
                max_fwd_peak_mem = cur_fwd_peak_mem
                # update the fwd peak point
                fwd_peak_point = (
                    current_module
                    if not cur_act_info.fwd_peak_point
                    else (current_module + " -> " + cur_act_info.fwd_peak_point)
                )
            if cur_bwd_peak_mem > max_bwd_peak_mem:
                cur_bwd_cache_mem_no_cache = cur_act_info.bwd_peak_mem_no_cache
                cur_bwd_peak_prev_cache_mem = (
                    cur_cache_mem + cur_act_info.bwd_peak_prev_cache_mem
                )
                max_bwd_peak_mem = cur_bwd_peak_mem
                # update the bwd peak point
                bwd_peak_point = (
                    current_module
                    if not cur_act_info.bwd_peak_point
                    else (current_module + " -> " + cur_act_info.bwd_peak_point)
                )

            # For debug memory
            if (
                self.path_debug_context
                and self.path_debug_context.target_point is not None
            ):
                # get the parent path of the current module
                parent_path = get_point_name(
                    parent=self.parent, current=self.current, sep=" -> "
                )
                # convet to list
                parent_path_list = parent_path.split(" -> ")
                # get the full path of the current module
                current_point = get_point_name(
                    parent=parent_path, current=current_module, sep=" -> "
                )
                data = self.path_debug_context.get_point_datas(
                    enable_recompute=enable_recompute
                )
                next_parent_to_data = self.path_debug_context.get_next_parent_to_point(
                    enable_recompute=enable_recompute
                )
                if current_point in self.path_debug_context.target_point:
                    # if hit the target point, record the point info，
                    assert current_point not in data, "point should not be duplicated"

                    point_info = PointDebugInfo(
                        point=current_point,
                        parent_path_list=parent_path_list,
                        next_parent_path_to_collect=parent_path_list,
                        prev_cache_mem=cur_cache_mem,
                        fwd_peak_no_cache_mem=cur_act_info.fwd_peak_mem_no_cache,
                        bwd_peak_no_cache_mem=cur_act_info.bwd_peak_mem_no_cache,
                    )
                    data[current_point] = point_info
                if current_point in next_parent_to_data.keys():
                    # We use post-order traversal to collect point information.
                    # To collect point information, we need to gather the memory of the parent path
                    # and add it to the current point’s information.

                    for v in next_parent_to_data[current_point]:
                        v.prev_cache_mem += cur_cache_mem
                        # update the next parent path
                        v.next_parent_path_to_collect.pop(-1)
            # add activation cached
            if (
                enable_recompute
                and module.enable_recompute
                and module.recompute_granularity == "full"
            ):
                cur_cache_mem += module.all_input_element_num()
            else:
                cur_cache_mem += cur_act_info.activation_mem_cache

        act_info.fwd_peak_mem_no_cache = cur_fwd_peak_mem_no_cache
        act_info.fwd_peak_prev_cache_mem = cur_fwd_peak_prev_cache_mem
        act_info.fwd_peak_point = fwd_peak_point
        act_info.bwd_peak_mem_no_cache = cur_bwd_cache_mem_no_cache
        act_info.bwd_peak_prev_cache_mem = cur_bwd_peak_prev_cache_mem
        act_info.activation_mem_cache = cur_cache_mem
        act_info.bwd_peak_point = bwd_peak_point
        assert max_fwd_peak_mem == act_info.fwd_peak_mem
        assert max_bwd_peak_mem == act_info.bwd_peak_mem

    def _comp_leaf_act_info_impl(self):
        raise NotImplementedError

    def _comp_act_info(self):
        if len(self.children_ordered_module) > 0:
            self._comp_submod_act_info_impl(self._act_info)
            self._comp_submod_act_info_impl(
                self._act_info_with_recomp, enable_recompute=True
            )
        else:
            self._comp_leaf_act_info_impl()
            # leaf module act info is the same with recompute,
            # because _act_info_with_recomp is used to distinguish
            # the case of recompute in the combined module
            self._act_info_with_recomp = deepcopy(self._act_info)

    def _comp_leaf_model_info_impl(self):
        raise NotImplementedError

    def _comp_model_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._model_info = self._model_info + module.get_model_info()
        else:
            self._comp_leaf_model_info_impl()

    # =========================
    # Compute or Communicate Related
    # =========================

    def _comp_leaf_flops_info(self):
        raise NotImplementedError

    def _comp_leaf_mem_accessed_info(self):
        raise NotImplementedError

    def _comp_leaf_intra_net_info(self):
        raise NotImplementedError

    def _comp_compute_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._compute_info = self._compute_info + module.get_compute_info()
        else:
            self._comp_leaf_flops_info()
            self._comp_leaf_mem_accessed_info()
            self._comp_leaf_intra_net_info()

    def _comp_cost_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._cost_info = self._cost_info + module.get_cost_info()
        else:
            raise NotImplementedError

    def _comp_cost_info_impl(
        self,
        fwd_op="default",
        bwd_grad_act_op="default",
        bwd_grad_w_op="default",
        enable_recompute=False,
    ):
        compute_time = self.system.compute_op_time(fwd_op, self._compute_info.fwd_flops)
        mem_time = self.system.compute_mem_access_time(
            self._compute_info.fwd_accessed_mem
        )
        self._cost_info.fwd_compute_time = self.compute_end2end_time(
            compute_time=compute_time, mem_time=mem_time
        )
        if enable_recompute:
            self._cost_info.recompute_compute_time = self._cost_info.fwd_time

        compute_time = self.system.compute_op_time(
            bwd_grad_act_op, self._compute_info.bwd_grad_act_flops
        )
        mem_time = self.system.compute_mem_access_time(
            self._compute_info.bwd_grad_act_accessed_mem
        )
        self._cost_info.bwd_grad_act_time = self.compute_end2end_time(
            compute_time=compute_time, mem_time=mem_time
        )

        compute_time = self.system.compute_op_time(
            bwd_grad_w_op, self._compute_info.bwd_grad_w_flops
        )
        mem_time = self.system.compute_mem_access_time(
            self._compute_info.bwd_grad_w_accessed_mem
        )
        self._cost_info.bwd_grad_w_time = self.compute_end2end_time(
            compute_time=compute_time, mem_time=mem_time
        )

    # =========================
    # Agg Related
    # =========================
    def get_compute_info(self) -> ModuleComputeInfo:
        assert (
            self._info_ready
        ), "flops/mem info not ready, please call the module to compute info"
        return self._compute_info

    def get_act_info(self) -> ActivationInfo:
        assert (
            self._info_ready
        ), "act info not ready, please call the module to compute info"
        return self._act_info

    def get_act_info_with_recomp(self) -> ActivationInfo:
        assert (
            self._info_ready
        ), "act info with recompute not ready, please call the module to compute info"
        return self._act_info_with_recomp

    def get_model_info(self) -> ModuleMemoryInfo:
        assert (
            self._info_ready
        ), "model info not ready, please call the module to compute info"
        return self._model_info

    def get_cost_info(self) -> ModuleCostInfo:
        assert (
            self._info_ready
        ), "cost info not ready, please call the module to compute info"
        return self._cost_info

    def __call__(
        self, input_info: InputOutputInfo, path_debug_context: PathDebugContext = None
    ) -> InputOutputInfo:
        # reset last result info
        self._reset_infos()

        self.set_input_state_info(input_info)
        self.set_path_debug_context(path_debug_context)
        # Debug, record the parent module and
        if self.path_debug_context:
            self.parent = self.path_debug_context.parent
            self.current = self.path_debug_context.current

        # call once, return all fwd, bwd info
        self._pre_op()
        output_info = None
        if len(self.children_ordered_module) > 0:
            for idx, module in enumerate(self.children_ordered_module):
                current_repr = "(" + str(idx) + ")" + module.__class__.__name__
                if self.path_debug_context:
                    self.path_debug_context.path_list.append(current_repr)
                output_info = module(input_info, path_debug_context)
                input_info = output_info
                if self.path_debug_context:
                    self.path_debug_context.path_list.pop(-1)
        # aggregate the info or compute the leaf info
        self._comp_model_info()
        self._comp_act_info()
        self._comp_compute_info()
        self._post_op()
        self._comp_cost_info()
        self._info_ready = True
        output_info = output_info if output_info else self.output_info
        return output_info

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module
        """
        return ""

    # modified from
    # https://github.com/pytorch/pytorch/blob/08b5e07/torch/ao/nn/quantized/modules/utils.py#L114  # pylint: disable=line-too-long
    def __repr__(self) -> str:
        # pylint: disable=invalid-name
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        prev_mod_str = None
        prev_start_idx = 0
        for idx, module in enumerate(self.children_ordered_module):
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)

            if prev_mod_str == mod_str:
                # merge
                if child_lines:
                    child_lines.pop()
                child_lines.append(
                    "(" + str(prev_start_idx) + "->" + str(idx) + "): " + mod_str
                )
            else:
                child_lines.append("(" + str(idx) + "): " + mod_str)
                prev_start_idx = idx
            prev_mod_str = mod_str

        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        # pylint: enable=invalid-name
        return main_str


class Result:
    """A simple class to wrap the result dict"""

    def __init__(self, result: dict) -> None:
        self.data = result

    def get(self, key: str):
        return self.data.get(key, None)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return to_json_string(self.data)

    def __str__(self):
        return self.to_json_string()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"
