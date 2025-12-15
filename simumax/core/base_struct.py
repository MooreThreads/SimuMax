"""Basic data structures for simumax"""

from copy import deepcopy
from dataclasses import dataclass, asdict
from abc import ABC
from typing import List, Tuple, Dict
import time
import types
import multiprocessing
try:
    from mpi4py import MPI
    enable_mpi = True
except ImportError:
    enable_mpi = False
import time
import json
import os
from simumax.core.tensor import TensorSize
from simumax.core.config import StrategyConfig, SystemConfig, get_capture_graph_only, SIMU_DEBUG, TMP_PATH
from simumax.core.utils import get_point_name, path_convert_to_str, to_json_string 
from simumax.core.utils import human_readable_bytes, human_readable_nums, human_readable_times
from simumax.core.utils import get_rank_group
from simumax.core.graph import SimuONNXGraphBuilder

class RecomputeStatus:
    NO_RECOMPUTE = "no_recompute"
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"   
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

    def get_all_flops(self):
        """Get all flops of the module in forward and backward(bwd_act and bwd_w) pass."""
        return [self.fwd_flops, self.bwd_grad_act_flops, self.bwd_grad_w_flops]
    
    def get_all_accessed_mem(self):
        """Get all accessed memory of the module in forward and backward(bwd_act and bwd_w) pass."""
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
    
    cache_for_bwd_mem:int = 0
    
    # TODO(sherry): delete this, for debug
    fwd_idx = 0
    fwd_total_activation_mem_cache:int = 0
    @property
    def fwd_peak_mem(self):
        return self.fwd_peak_prev_cache_mem + self.fwd_peak_mem_no_cache
    
    @property
    def total_activation_mem_cache(self):
        return self.activation_mem_cache
    @property
    def bwd_peak_mem(self):
        return self.bwd_peak_prev_cache_mem + self.bwd_peak_mem_no_cache

    def to_dict(self):
        data_dict =  asdict(self)
        data_dict["fwd_peak_mem"] = self.fwd_peak_mem
        data_dict["bwd_peak_mem"] = self.bwd_peak_mem
        data_dict["peak_stage"] = "forward" if self.fwd_peak_mem > self.bwd_peak_mem else "backward"
        data_dict["peak_path"] = self.fwd_peak_point if self.fwd_peak_mem > self.bwd_peak_mem else self.bwd_peak_point
        data_dict["peak_mem"] = max(self.fwd_peak_mem, self.bwd_peak_mem )
        return data_dict

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
    weight_numel: int = 0
    dense_weight_bytes: int = 0
    dense_grad_bytes: int = 0
    dense_state_bytes: int = 0
    moe_weight_numel: int = 0
    moe_weight_bytes: int = 0
    moe_grad_bytes: int = 0
    moe_state_bytes: int = 0

    @property
    def all(self):
        return self.dense_weight_bytes + self.dense_grad_bytes + self.dense_state_bytes + self.moe_weight_bytes + self.moe_grad_bytes + self.moe_state_bytes
    
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
            "moe_state_bytes": self.moe_state_bytes
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
            weight_numel=self.weight_numel + other.weight_numel,
            dense_weight_bytes=self.dense_weight_bytes + other.dense_weight_bytes,
            dense_grad_bytes=self.dense_grad_bytes + other.dense_grad_bytes,
            dense_state_bytes=self.dense_state_bytes + other.dense_state_bytes,
            moe_weight_numel = self.moe_weight_numel + other.moe_weight_numel,
            moe_weight_bytes=self.moe_weight_bytes + other.moe_weight_bytes,
            moe_grad_bytes=self.moe_grad_bytes + other.moe_grad_bytes,
            moe_state_bytes=self.moe_state_bytes + other.moe_state_bytes
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

    # def all_time(self):
    #     return self.fwd_time + self.fwd_net_time + self.recompute_compute_time + self.recompute_net_time + self.bwd_time + self.bwd_net_time
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
        """Get all cost of the model in forward and backward(bwd_act, bwd_w) pass"""
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


class FwdQue:
    log_file = './tmp/log.log'
    def __init__(self, call_stk='', que=None, ):
        self.que= que if que else []
        # self.call_stk = call_stk+f'-{self.__class__.__name__}'
        self.call_stk = call_stk
        self.st = None
    def step(self, t, manager):
        if self.st is None:
            self.st = t[0]
        if self._step(t, manager):
            info = f"{self.call_stk} fwd cost {t[0]-self.st:.6f} st {self.st:.6f} ed {t[0]:.6f}"
            with open(self.log_file, 'a') as f:
                f.write(info+'\n')
            return True
        return False

    def _step(self, t, manager):
        while self.que:
            if not self.que[0].step(t, manager):
                return False
            self.que.pop(0)
        t[0] += 2e-3  # just for tracing visialization
        return True
    
    def append(self, x):
        self.que.append(x)

    def __bool__(self):
        return bool(self.que)
    

class BwdStk:
    log_file = './tmp/log.log'
    def __init__(self, call_stk='',stk=None):
        self.stk =  stk if stk else []
        # self.call_stk = call_stk+f'-{self.__class__.__name__}'
        self.call_stk = call_stk
        self.st_bwd = None
    def bwd(self, t, manager):
        if self.st_bwd is None:
            self.st_bwd = t[0]
        if self._bwd(t, manager):
            info = f"{self.call_stk} fwd cost {t[0]-self.st_bwd:.6f} st {self.st_bwd:.6f} ed {t[0]:.6f}"
            with open(self.log_file, 'a') as f:
                f.write(info+'\n')
            return True
        return False
        
    def _bwd(self, t, manager):
        while self.stk:
            if not self.stk[-1].bwd(t, manager):
                return False
            self.stk.pop(-1)
        t[0] += 2e-3  # just for tracing visialization
        return True

    def append(self, x):
        self.stk.append(x)

    def __bool__(self):
        return  bool(self.stk)
    

class BaseModel: #templete for non-leaf model
    def __init__(self, specific_name=''):
        # self.call_stk = call_stk+f'-{self.__class__.__name__}'
        self.call_stk = f'-{self.__class__.__name__}'
        self.specific_name = specific_name
        if specific_name:
            self.call_stk =f'-{specific_name}'
        self.layers = [] #layer require prefill_fwd/bwd, could be (non-)leaf model

    def prefill(self, args, call_stk='', com_buff=None):
        #
        pass

    def prefill_fwd(self):
        # return a fwd job:  FwdQue or LeafModel
        fwd = FwdQue(call_stk=self.call_stk)
        for layer in self.layers:
            fwd.append(layer.prefill_fwd())
        return fwd
    
    def prefill_bwd(self):
        # return a bwd job:  BwdStk or LeafModel
        bwd = BwdStk(call_stk=self.call_stk)
        for layer in self.layers:
            bwd.append(layer.prefill_bwd())
        return bwd

class PostInitMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if hasattr(obj, '__post_init__'):
            obj.__post_init__()
        return obj

class MetaModule(BaseModel, metaclass = PostInitMeta):
    """
    Assume that there are two types of modules:
    1. The most basic module that does not have children modules
    2. A module composed of children modules, except for children modules,
       there are no other calculations
    """

    dtype_to_element_size = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
    id_counter = 0
    def __init__(self, strategy:StrategyConfig, system:SystemConfig, specific_name='', parent_module = None) -> None:
        super().__init__(specific_name)
        self.strategy = strategy
        self.system = system
        self.offload_inputs = False

        self.children_ordered_module:List[MetaModule] = []
        self.children_modules:List[MetaModule] = [] # children modules是所有子模块的列表（无序）
        self.children_modules_names:Dict[MetaModule, str] = {}
        self.default_dtype = strategy.dtype 
        self._init_strategy = False
        self.input_info = None
        self.output_info_ = None
        # self.cache_info = []
        self.enable_recompute = False
        self.recompute_granularity = "full"
        self.parent_module:MetaModule = parent_module
        self._reset_infos()
        self.is_leaf_module = False
        self.cache_inputs = False
        self.cache_outputs = False
        self.recompute_status:str = RecomputeStatus.NO_RECOMPUTE # "first", "middle", "last", default = middle
        self.is_breakpoints = False
        self.ordered_module_hooks:List[callable] = None
        self.forward_pre_hooks:List[callable] = None
        self.forward_post_hooks:List[callable] = None
        self.init_ready = False
        self.is_recompute_forward_finished = False
        self.full_name = "self"
        self.name = ''
        self.call_idx = -1

        # for Selective recompute strategy
        self.all_recompute_nodes:List[MetaModule] = []
        self.all_leaf_nodes:List[MetaModule] = []
        self.status_ready = False
        self.is_variance_node = False
        self.id = MetaModule.id_counter
        MetaModule.id_counter += 1

    def __post_init__(self):
        self.is_leaf_module = self.set_children_modules()
        self.cache_inputs = not self.enable_recompute 
        self.init_ready = True

    def set_children_modules(self):
        is_leaf = True
        for name, member in vars(self).items():
            if isinstance(member, MetaModule):
                is_leaf = False
                if member.parent_module is None:
                    member.parent_module = self
                    self.children_modules.append(member) 
                    self.children_modules_names[member] = name
        return is_leaf
    
    def set_variance_node(self, is_variance_node:bool):
        if self.strategy.recompute_variance:
            self.is_variance_node = is_variance_node        
    @property
    def output_info(self):
        if self.output_info_ is None:
            self.output_info_ = self.create_output_info()
        return self.output_info_
    
    def set_leaf_full_name(self, parent_name:str):
        for child, name in self.children_modules_names.items():
            child.full_name = parent_name + '.' + name
            child.name = name
            child.set_leaf_full_name(child.full_name)
    
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
        self.is_recompute_forward_finished = False
        self.children_ordered_module:List[MetaModule] = [] 
        self.children_modules:List[MetaModule] = [] 
        self.all_recompute_nodes:List[MetaModule] = []
        self.all_leaf_nodes:List[MetaModule] = []
        
    def get_all_leaf_modules(self):
        assert self.status_ready, f"{self.__class__.__name__} is not ready yet, please run set_first_last_recompute_status() first"
        return self.all_leaf_nodes
    
    def get_weight(self) -> TensorSize:
        return None
    
    def register_add_ordered_module_hooks(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_ordered_module_hooks(hook)
        for module in self.children_modules:
            module.register_add_ordered_module_hooks(hook)

    def register_add_forward_pre_hook(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_forward_pre_hooks(hook)
        for module in self.children_modules:
            module.register_add_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_forward_post_hooks(hook)
        for module in self.children_modules:
            module.register_forward_post_hook(hook)

    def add_ordered_module_hooks(self, hook):
        if self.ordered_module_hooks is None:
            self.ordered_module_hooks = []
        self.ordered_module_hooks.append(hook)
    def add_forward_pre_hooks(self, hook):
        if self.forward_pre_hooks is None:
            self.forward_pre_hooks = []
        self.forward_pre_hooks.append(hook)
    def add_forward_post_hooks(self, hook):
        if self.forward_post_hooks is None:
            self.forward_post_hooks = []
        self.forward_post_hooks.append(hook)
    def call_add_ordered_module_hooks(self, *args):
        if self.ordered_module_hooks is not None:
            for hook in self.ordered_module_hooks:
                hook(self, *args)
    def call_forward_pre_hook(self, *args):
        if self.forward_pre_hooks is not None:
            for hook in self.forward_pre_hooks:
                hook(self, *args)
    def call_forward_post_hook(self, *args):
        if self.forward_post_hooks is not None:
            for hook in self.forward_post_hooks:
                hook(self, *args)

    def register_module(self, sub_module):
        self.children_ordered_module.append(sub_module)
        # TODO(sherry): 支持register hook
        self.call_add_ordered_module_hooks(sub_module)
    
    def set_dtype(self, dtype: str):
        assert dtype in ["fp32", "fp16", "bf16"]
        self.dtype = dtype
    
    def parse_recompute_node(self):
        all_ordered_leaf_module:List[MetaModule] = []
        def dfs_traverse_leaf_module(module:MetaModule):
            if module.is_leaf_module:
                all_ordered_leaf_module.append(module)
            else:
                for module in self.children_ordered_module:
                    dfs_traverse_leaf_module(module)
        dfs_traverse_leaf_module(self)
      
        self.all_ordered_leaf_module = all_ordered_leaf_module

        fisrt_recomps:List[MetaModule] = []
        last_recomps:List[MetaModule] = []
        pre_enabled_recompute = False
        for leaf in all_ordered_leaf_module:
            if not pre_enabled_recompute and leaf.enable_recompute:
                leaf.recompute_status = "first"
                fisrt_recomps.append(leaf)
            if pre_enabled_recompute and not leaf.enable_recompute:
                leaf.recompute_status = "last"
                last_recomps.append(leaf)
            pre_enabled_recompute = leaf.enable_recompute

        # for i, p in enumerate(fisrt_recomps):
        #     print(f"{i}first recomputable module, path={p.}")

    @property
    def element_size(self):
        dtype = self.default_dtype
        if getattr(self, "dtype", False):
            dtype = self.dtype
        return self.dtype_to_element_size[dtype]

    @property
    def first_compute_module(self):
        return self.children_ordered_module[0] if len(self.children_ordered_module) > 0 else self
    # =========================
    # Basic Compute Related
    # =========================
    def compute_end2end_time(self, compute_time, mem_time):
        return self.system.compute_end2end_time(compute_time, mem_time)

    def all_input_element_num(self):
        res = 0

        if isinstance(self.input_info, InputOutputInfo):
            input_info = [self.input_info]
        else:
            input_info = self.input_info
        for ii in input_info:
            if isinstance(ii, InputOutputInfo):
                for x in ii.tensors:
                    res += x.get_memory_size()
            elif isinstance(ii, TensorSize):
                res += ii.get_memory_size()
        return res

    def all_output_element_num(self):
        res = 0
        # element_size = self.element_size
        if isinstance(self.output_info, InputOutputInfo):
            output_info = [self.output_info]
        else:
            output_info = self.output_info
        for oi in output_info:
            if isinstance(oi, InputOutputInfo):
                for x in oi.tensors:
                    res += x.get_memory_size()
            elif isinstance(oi, TensorSize):
                res += oi.get_memory_size()
        return res
    
    def set_input_state_info(self, input_info: InputOutputInfo):
        # self.input_info = deepcopy(input_info)
        self.input_info = input_info # reference assignments are allowed here

    def set_path_debug_context(self, path_debug_context: PathDebugContext):
        self.path_debug_context = deepcopy(path_debug_context)

    def create_output_info(self):
        return InputOutputInfo([])

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
    def _comp_submod_cache_info_impl(self):
        ...
    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.fwd_peak_prev_cache_mem = 0
        self._act_info.bwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_prev_cache_mem = 0

    def _comp_act_info(self):
        if len(self.children_ordered_module) == 0:
            self._comp_leaf_act_info_impl()
            # leaf module act info is the same with recompute,
            # because _act_info_with_recomp is used to distinguish
            # the case of recompute in the combined module
            if self.is_variance_node:
                # print("Warning: variance node change peak")
                # self._act_info.activation_mem_cache = 0
                # self._act_info.bwd_peak_mem_no_cache += self._act_info.activation_mem_cache
                pass
            self._act_info_with_recomp = deepcopy(self._act_info)
        else:
            for module in self.children_ordered_module:
                self._act_info.activation_mem_cache = self._act_info.activation_mem_cache + module._act_info.activation_mem_cache

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

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

    def _comp_compute_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._compute_info = self._compute_info + module.get_compute_info()
        else:
            self._comp_leaf_flops_info()
            self._comp_leaf_mem_accessed_info()
            self._comp_leaf_intra_net_info()
            if self.strategy.recompute_variance and self.is_variance_node:
                self._compute_info.recompute_accessed_mem = 0
                self._compute_info.recompute_flops = 0
                self._cost_info.recompute_net_time = 0
                self._cost_info.recompute_net_exposed_time = 0
                if SIMU_DEBUG:
                    print(f"- {self.full_name} is variance node, recompute_accessed_mem and recompute_flops are set to 0")

    def _comp_cost_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._cost_info = self._cost_info + module.get_cost_info()
        else:
            # raise NotImplementedError
            self._comp_cost_info_impl(
                fwd_op="default",
                bwd_grad_act_op="default",
                bwd_grad_w_op="default",
                enable_recompute=self.enable_recompute,
            )  
                 
        if (
            self.path_debug_context
            and self.path_debug_context.target_point is not None
        ):
            # get the parent path of the current module
            path = get_point_name(
                parent=self.parent, current=self.current, sep=" -> "
            )
            if path in self.path_debug_context.target_point:
                file_path = f'{TMP_PATH}/cost_log.json'
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        try:
                            existing_data = json.load(file) 
                        except json.JSONDecodeError:
                            existing_data = {}
                else:
                    existing_data = {}
                existing_data.update(
                    {path:{"cost_F": self._cost_info.fwd_compute_time,
                            "cost_B": self._cost_info.bwd_grad_act_time,
                            "cost_W": self._cost_info.bwd_grad_w_time,
                            "recompute_F": self._cost_info.recompute_compute_time,
                            "net_F": self._cost_info.fwd_net_time,
                            "net_B": self._cost_info.bwd_net_time,
                            }
                            }
                )
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(existing_data, file, indent=4, ensure_ascii=False)
    def set_details(
        self, stage, compute_details, io_details
    ):
        if not hasattr(self, 'details'):
            self.details = {}
        self.details[stage] = {
            "compute_details" : deepcopy(compute_details),
            "io_details" : deepcopy(io_details),
        }

    def get_input_shapes_desc(self, stage):
        if isinstance(self, LinearBase):
            bmnk_info = self.get_gemm_bmnk(stage)
            b, m, n, k = bmnk_info['B'], bmnk_info['M'], bmnk_info['N'], bmnk_info['K']
            layout = bmnk_info['layout']
            accumulate = bmnk_info['accumulate'] # TODO(sherry): in bwd_grad_w, accumulate is True
            out_dtype = bmnk_info['out_dtype']
            return f'b={b}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}'
        else:
            return ""
    def _comp_cost_info_impl(
        self,
        fwd_op="default",
        bwd_grad_act_op="default",
        bwd_grad_w_op="default",
        enable_recompute=False,
    ):
        def compute_details(op_name, stage, flops, accessed_mem):
            #compute_details include compute time, tflops of accelerator, flops of current op, etc.
            compute_details = self.system.compute_op_accuracy_time(op_name, flops, shape_desc= self.get_input_shapes_desc(stage), reture_detail=True)

            # io_details include io time, gbps of accelerator, io size of current op, etc.
            io_details = self.system.compute_mem_access_time(op_name,
                accessed_mem, reture_detail=True
            )

            # Get final time, we can set "roofline" or "compute_only" in accelerator config, default is roofline
            # if rooline, final time = max(compute_time, mem_time)
            # if compute_only, final time = compute_time
            end2end_time = self.compute_end2end_time(
                compute_time=compute_details['compute_only_time'], mem_time=io_details['io_time'],
            )

            # save details for each stage, for analysis
            self.set_details(stage, compute_details, io_details)
            return end2end_time
        # 1. forward   
        self._cost_info.fwd_compute_time = compute_details(fwd_op, 'fwd', self._compute_info.fwd_flops, self._compute_info.fwd_accessed_mem)
        self._cost_info.bwd_grad_act_time = compute_details(bwd_grad_act_op, 'bwd_grad_act', self._compute_info.bwd_grad_act_flops, self._compute_info.bwd_grad_act_accessed_mem)
        self._cost_info.bwd_grad_w_time = compute_details(bwd_grad_w_op, 'bwd_grad_w', self._compute_info.bwd_grad_w_flops, self._compute_info.bwd_grad_w_accessed_mem)

        self._cost_info.recompute_compute_time = self._cost_info.fwd_time if self.enable_recompute else 0

        if self.enable_recompute and self.is_variance_node:
            self._cost_info.recompute_compute_time = 0
            if SIMU_DEBUG:
            # if 1:
                print(f'%% {self.name} is variance node, recompute_compute_time is 0')

        # if (
        #     self.path_debug_context
        #     and self.path_debug_context.target_point is not None
        # ):
        #     # get the parent path of the current module
        #     path = get_point_name(
        #         parent=self.parent, current=self.current, sep=" -> "
        #     )
        #     if path in self.path_debug_context.target_point:
        #         file_path = f'{TMP_PATH}/cost_log.json'
        #         os.makedirs(TMP_PATH, exist_ok=True)
        #         if os.path.exists(file_path):
        #             with open(file_path, 'r', encoding='utf-8') as file:
        #                 try:
        #                     existing_data = json.load(file) 
        #                 except json.JSONDecodeError:
        #                     existing_data = {}
        #         else:
        #             existing_data = {}
        #         existing_data.update(
        #             {path:{"cost_F": self._cost_info.fwd_compute_time,
        #                     "cost_B": self._cost_info.bwd_grad_act_time,
        #                     "cost_W": self._cost_info.bwd_grad_w_time,
        #                     "recompute_F": self._cost_info.recompute_compute_time,
        #                     "net_F": self._cost_info.fwd_net_time,
        #                     "net_B": self._cost_info.bwd_net_time,
        #                     }
        #                     }
        #         )
        #         with open(file_path, 'w', encoding='utf-8') as file:
        #             json.dump(existing_data, file, indent=4, ensure_ascii=False)

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
        ), f"model {self.__class__.__name__} info not ready, please call the module to compute info"
        return self._model_info

    def get_cost_info(self) -> ModuleCostInfo:
        assert (
            self._info_ready
        ), "cost info not ready, please call the module to compute info"
        return self._cost_info
    
    def forward(self, input_info: InputOutputInfo, path_debug_context: PathDebugContext) -> InputOutputInfo:
        raise NotImplementedError   
    
    def __call__(
        self, input_info: InputOutputInfo, path_debug_context: PathDebugContext
    ) -> InputOutputInfo:
        is_capture_only = get_capture_graph_only()
        if isinstance(input_info, TensorSize):
            input_info = InputOutputInfo([input_info])

        self.call_forward_pre_hook(input_info)

        # reset last result info
        self._reset_infos()
        
        self.set_input_state_info(input_info) # record the input
        self.set_path_debug_context(path_debug_context) # copy path debug context
        
        if self.parent_module and self not in self.parent_module.children_ordered_module:
            self.parent_module.register_module(self) # Non-leaf nodes also register themselves in the children module on the previous layer.
        # Debug, record the parent module and
        if self.path_debug_context:
            idx = len(self.parent_module.children_ordered_module)-1 if self.parent_module else 0
            current_repr = "(" + str(idx) + ")" + self.__class__.__name__

            self.path_debug_context.path_list.append(current_repr)
            
            self.parent = get_point_name(
                parent=path_debug_context.parent, 
                current=path_debug_context.current, sep=" -> "
            ) 
            self.current = current_repr
            self.current_full_module_path = get_point_name(parent=self.parent, current=self.current, sep=" -> ") #FIXME(sherry): path_debug_context is deepcopy to module. How to modify the parent of the temporary variable and pass it to the next module?

        # call once, return all fwd, bwd info
        self._pre_op()
        output_info = None        

        if not self.is_leaf_module:
            output_info = self.forward(input_info, self.path_debug_context)
        else:
            output_info = output_info if output_info else self.output_info # output_info = None, return leaf output
            if is_capture_only:
                graph_builder = SimuONNXGraphBuilder()
                graph_builder.add_node(op = self,
                                    op_type = self.__class__.__name__, 
                                    inputs = input_info.tensors if isinstance(input_info, InputOutputInfo) else [input_info],
                                    outputs = output_info.tensors  if isinstance(output_info, InputOutputInfo) else [output_info]
                                    )
        
        if not is_capture_only:
            # aggregate the info or compute the leaf info
            self._comp_model_info()  #static model memory usage
            self._comp_act_info()  #activation
            self._comp_compute_info()
            self._post_op()
            self._comp_cost_info()
        
        self._info_ready = True
        
        if isinstance(output_info, InputOutputInfo) and len(output_info.tensors) == 1:
            output_info = output_info.tensors[0]

        # path = get_point_name(parent=self.parent, current=self.current, sep=" -> ")
        self.call_forward_post_hook(input_info, output_info)
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
        def get_variable_name(var, namespace):
            for name, value in namespace.items():
                if value is var:
                    return name
            return None

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
        show_full_name = False
        for idx, module in enumerate(self.children_ordered_module):
            if show_full_name:
                mod_str = module.full_name + " " + repr(module)
            else:
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
        

        module = self
        
        # TODO(sherry): delete this, for debug

        main_str += ")"

        show_details = True
        if show_details:
            cost_info = module._cost_info
            main_str += f"\n\t1. cost: (total_time={cost_info.all_time:.2f} ms, fwd_details=(sum={cost_info.fwd_time+cost_info.fwd_net_time:.2f} ms, compute={cost_info.fwd_compute_time*1000:.2f} us, net={cost_info.fwd_net_time*1000:.2f} us), bwd_details=(sum={cost_info.bwd_time+cost_info.bwd_net_time:.2f} ms, compute={cost_info.bwd_compute_time*1000:.2f} us, net={cost_info.bwd_net_time*1000:.2f} us), variance_node={self.is_variance_node} flops={sum(module._compute_info.get_all_flops())/1e12:.2f} T) "

            module_info = module._model_info
            main_str += f"\n\t2. memory: (d_w={module_info.dense_weight_bytes}, d_g={module_info.dense_grad_bytes}, m_w={module_info.moe_weight_bytes}, m_g={module_info.moe_grad_bytes})"

        return main_str

class RecomputeBreakModule(MetaModule):
    def __init__(self, strategy, system, specific_name='', parent_module=None):
        super().__init__(strategy, system, specific_name, parent_module=parent_module)
        self.enable_recompute = False
    
    # TODO(sherry): no memory and not cost. Need to be implemented
    def create_output_info(self):
        output_info = InputOutputInfo(tensors=[t.new() for t in self.input_info.tensors])
        return output_info
class LinearBase(MetaModule):
    def __init__(self, input_size, output_size, strategy, system, specific_name='', parent_module=None):
        super().__init__(strategy, system, specific_name, parent_module)
        self.input_size = input_size
        self.output_size = output_size

    @property
    def micro_input_tensor(self) -> TensorSize:
        return TensorSize(shape=[])
    
    def get_weight(self):
        return TensorSize(shape=(self.output_size, self.input_size), dtype='fp8' if self.strategy.fp8 else 'bf16')
    
    def get_gemm_mnk(self, stage, format=False):
        """Get the m, n, k of the gemm operation, include forward and backward(bwd_act, bwd_w) pass"""
        inp_tensor = self.micro_input_tensor
        if inp_tensor.ndim == 2:
            bs = inp_tensor.shape[0]
        else:
            bs = inp_tensor.shape[0] * inp_tensor.shape[1]
        print(self.input_info.tensors[0])
        inp = self.input_size
        out = self.output_size
        if stage == 'fwd':
            return [[bs, inp], [inp, out], [bs, out]] if format else bs, inp, out
        elif stage == 'bwd_act':
            return [[bs, out], [out, inp], [bs, inp]] if format else bs, out, inp
        elif stage == 'bwd_w':
            return [[out, bs], [bs, inp], [out, inp]] if format else out, bs, inp
        elif stage == 'all':
            # get ms, ns, ks for all stages, fwd, bwd_act, bwd_w
            return [bs, bs, out], [inp, out, bs], [out, inp, inp]
        
    def get_gemm_bmnk(self, stage, format=False):
        """Get the b, m, k, n of the gemm operation, include forward and backward(bwd_grad_act, bwd_grad_w) pass sequently"""
        inp_tensor = self.micro_input_tensor
        if inp_tensor.ndim == 2:
            bs, seq_len = 1, inp_tensor.shape[0]
        else:
            bs, seq_len = inp_tensor.shape[:2] 
        inp = self.input_size
        out = self.output_size
        bs, seq_len, inp, out = int(bs), int(seq_len), int(inp), int(out)
        if stage == 'fwd':
            return [[bs, seq_len, inp], [inp, out], [bs, out]] if format else dict(B=bs, M=seq_len, K=inp, N=out, layout='TN', accumulate=False, out_dtype='bf16')
        elif stage == 'bwd_grad_act':
            return [[bs, seq_len, out], [out, inp], [bs, inp]] if format else dict(B=bs, M=seq_len, K=out, N=inp, layout='NN', accumulate=False, out_dtype='bf16')
        elif stage == 'bwd_grad_w':
            return [[1, out, bs*seq_len], [bs*seq_len, inp], [out, inp]] if format else dict(B=1, M=out, K=bs*seq_len, N=inp, layout='NT', accumulate=True, out_dtype='bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32')
        elif stage == 'all':
            # get bs, ms,  ks, ns for all stages, fwd, bwd_grad_act, bwd_grad_w, sequently
            return dict(B=[bs, bs, 1], M=[seq_len, seq_len, out], K=[inp, out, bs*seq_len], N=[out, inp, inp], layout=['TN', 'NN', 'NT'], accumulate=[False, False, True], out_dtype=['bf16', 'bf16', 'fp32'])


    def parse_fwd_bwd_gemm_shape(self):
        x = self.input_info
        if x.tensors[0].ndim == 3:
            batch_size = int(x.tensors[0].shape[0] * x.tensors[0].shape[1])
        elif x.tensors[0].ndim == 2:
            batch_size = int(x.tensors[0].shape[0])
        else:
            raise NotImplementedError("Only support 2D and 3D tensors")
     
        fwd_lhs_shape, fwd_rhs_shape, fwd_out_shape = self.get_gemm_mnk('fwd', format=True)
        bwd_a_lhs_shape, bwd_a_rhs_shape, bwd_a_out_shape = self.get_gemm_mnk('bwd', format=True)
        bwd_w_lhs_shape, bwd_w_rhs_shape, bwd_w_out_shape = self.get_gemm_mnk('bwd_w', format=True)
        
        return {
            "fwd_lhs_shape": fwd_lhs_shape,
            "fwd_rhs_shape": fwd_rhs_shape,
            "fwd_out_shape": fwd_out_shape,
            "bwd_a_lhs_shape": bwd_a_lhs_shape,
            "bwd_a_rhs_shape": bwd_a_rhs_shape,
            "bwd_a_out_shape": bwd_a_out_shape,
            "bwd_w_lhs_shape": bwd_w_lhs_shape,
            "bwd_w_rhs_shape": bwd_w_rhs_shape,
            "bwd_w_out_shape": bwd_w_out_shape
        }  

class GroupLinearBase(LinearBase):
    """Base class for GroupGemm"""
    def __init__(self, local_expert_num, input_size: int, output_size: int,  strategy, system, specific_name='', parent_module=None) -> None:
        super().__init__(input_size, output_size, strategy, system, specific_name, parent_module)
        self.local_expert_num = local_expert_num    

    def get_input_shapes_desc(self, stage):
        assert self.input_info.tensors[0].size(0) % self.local_expert_num == 0, f'input size {self.input_info.tensors[0].size(0)} is not divisible by local_expert_num {self.local_expert_num} {self.strategy.parallelism}'
        num_tokens = self.input_info.tensors[0].size(0) // self.local_expert_num
        shape_str = f'ng={self.local_expert_num}, M={num_tokens}, N={self.output_size}, K={self.input_size}'

        dtype_str = f", dtype={'fp8' if self.strategy.fp8 else 'bf16'}, out_dtype=bf16, main_grad_dtype={'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}"
        # if self.strategy.fp8:
        shape_str += dtype_str
        if stage == 'fwd':
            shape_str += ', stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True'
        elif stage == 'bwd_grad_act':
            shape_str += ', stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False'
        elif stage == 'bwd_grad_w':
            shape_str += ', stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False'
        else:
            raise ValueError(f'Invalid stage: {stage}') 
        return shape_str
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


class State_Thread:
    def __init__(self):
        self.comm_order = 0    
    
class SimuThread:
    def __init__(self):
        self.job = []  #job require step and bool
        self.time = [0]
        self.thread_state = State_Thread()

    def step(self, manager):
        while self.job:
            if isinstance(self.job[0], FwdQue):
                if not self.job[0].step(self.time, manager):
                    return False #block
            else:
                if not self.job[0].bwd(self.time, manager):
                    return False #block              
            if not self.job[0]:
                self.job.pop(0)
        return True #finish

class SimuSystem:
    def __init__(self):
        self.threads = [] #item in self.threads require step
        # self.comm_buff = []

    def simu(self, manager):
        n = len(self.threads)
        idx = 0
        finish = [0]*n
        while True:
            if self.threads[idx].step(manager):
                finish[idx]=1
            idx+=1
            idx %= n
            if sum(finish)==n:
                break
        print(f'end in {self.threads[0].time[0]}')

    def simu_mt(self, manager):
        n = len(self.threads)
        processes = []
        def worker(thread):
            """每个进程的工作函数，负责执行 thread.step()"""
            while True:
                if thread.step(manager):  # 如果 step() 返回 True，表示任务完成
                    break
                # time.sleep(1)

        # 创建并启动多个进程
        for i in range(n):
            p = multiprocessing.Process(target=worker, args=(self.threads[i],))
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        print(f'end in {self.threads[0].time[0]}')


class LeafModel():
    log_file = './tmp/log.log'
    def __init__(self, specific_name=''):
        self.st = None
        self.st_bwd = None
        self.call_stk =f'-{self.__class__.__name__}'
        if specific_name:
            self.call_stk =f'-{specific_name}'
    def step(self, t, manager):
        if self.st is None:
            self.st = t[0]
        if self._step(t, manager):
            info = f"{self.call_stk} fwd cost {t[0]-self.st:.6f} st {self.st:.6f} ed {t[0]:.6f}"
            with open(self.log_file, 'a') as f:
                f.write(info+'\n')
            return True
        return False

    def _step(self, t, manager):
        return True
    
    def bwd(self, t, manager):
        if self.st_bwd is None:
            self.st_bwd = t[0]
        if self._bwd(t, manager):
            info = f"{self.call_stk} fwd cost {t[0]-self.st_bwd:.6f} st {self.st_bwd:.6f} ed {t[0]:.6f}"
            with open(self.log_file, 'a') as f:
                f.write(info+'\n')
            return True
        return False
    def _bwd(self, t, manager):
        return True

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
    
    def prefill_fwd(self):
        return self
    
    def prefill_bwd(self):
        return self
    
class AtomModel(LeafModel):
    #simplify LeafModel with cost information
    def __init__(self, fwd_cost, bwd_cost, specific_name=''):
        super().__init__(specific_name)
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        # self.fwd_cost = fwd_cost*(1+random.random()*0.6)
        # self.bwd_cost = bwd_cost*(1+random.random()*0.6)
    def _step(self, t, manager):
        t[0] += self.fwd_cost
        return True
    def _bwd(self, t, manager):
        t[0] += self.bwd_cost
        return True
    
class Com(LeafModel):
    def __init__(self, id,rank,group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', global_rank=None):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.rank=rank
        self.group_size=group_size
        self.com_buff = com_buff
        self.fwd_cost=fwd_cost
        self.bwd_cost=bwd_cost
        self.global_rank = global_rank
        if self.fwd_cost==0 or group_size<=1:
            self.step = lambda *args:True
        if self.bwd_cost==0 or group_size<=1:
            self.bwd = lambda *args:True
        groups = get_comm_group(com_buff)
        if 'tp_group' in self.id:
            self.com_group = groups['tp_group']
        elif 'ep_group' in self.id:
            self.com_group = groups['ep_group']
        elif 'dp_group' in self.id:
            self.com_group = groups['dp_group']
        elif 'edp_group' in self.id:
            self.com_group = groups['edp_group']
        elif 'pp_group' in self.id:
            self.com_group = groups['pp_group']
        else:
            self.com_group = None

    def _step(self, t, manager):
        id = self.id+'fwd'
        if id.startswith('send_recv'):
            info = id.split('send_recv')[1]
            info = info.split('-')[1:]
            print(info)
            source, dest = info[:2]
            source, dest = int(source), int(dest)
            print(f"source={source},dest={dest},group={self.com_group}, group_size={self.com_group.Get_size()}, id={id}")
            if self.rank == source:
                self.com_group.send(id, dest)
                return True
            elif self.rank == dest:
                _ = self.com_group.recv(source)
                return True
            else:
                raise ValueError('rank not match')
        else:
            for comm_tpye in  ["all_gather", "reduce_scatter", "all_reduce", "all2all"]:
                if id.startswith(comm_tpye):
                    self.com_group.allgather(id)
                    return True
        raise NotImplementedError(f"not support for {id}")
            
        if not id in self.com_buff:
            self.com_buff[id] = {'is_ready': [0]*self.group_size, 'ready_time': 0}
            # nested_dict = manager.dict()
            # nested_dict['is_ready'] = manager.list([0] * self.group_size)  # 使用 Manager().list() 共享列表
            # nested_dict['ready_time'] = 0
            # self.com_buff[id] = nested_dict  # 将嵌套的共享对象赋值给 self.com_buff[id]
        self.com_buff[id]['is_ready'][self.rank]=1
        self.com_buff[id]['ready_time'] = max(t[0], self.com_buff[id]['ready_time'])
        if sum(self.com_buff[id]['is_ready'])==self.group_size:
            t[0] = self.fwd_cost + self.com_buff[id]['ready_time']
            return True
        return False
    
    def _bwd(self, t, manager,):
        id = self.id+'bwd'
        if id.startswith('send_recv'):
            info = id.split('send_recv')[1]
            info = info.split('-')[1:]
            print(info)
            source, dest = info[:2]
            source, dest = int(source), int(dest)
            if self.rank == source:
                self.com_group.send(id, dest)
                return True
            elif self.rank == dest:
                _ = self.com_group.recv(source)
                return True
            else:
                raise ValueError('rank not match')
        else:
            for comm_tpye in  ["all_gather", "reduce_scatter", "all_reduce", "all2all"]:
                if id.startswith(comm_tpye):
                    self.com_group.allgather(id)
                    return True
        raise NotImplementedError(f"not support for {id}")
    
        if not id in self.com_buff:
            # nested_dict = manager.dict()
            # nested_dict['is_ready'] = manager.list([0] * self.group_size)  # 使用 Manager().list() 共享列表
            # nested_dict['ready_time'] = 0
            # self.com_buff[id] = nested_dict
            self.com_buff[id] = {'is_ready': [0]*self.group_size, 'ready_time': 0}
        self.com_buff[id]['is_ready'][self.rank]=1
        self.com_buff[id]['ready_time'] = max(t[0], self.com_buff[id]['ready_time'])
        if sum(self.com_buff[id]['is_ready'])==self.group_size:
            t[0] = self.bwd_cost + self.com_buff[id]['ready_time']
            return True
        return False
    
class all_gather(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)
        # self.call_stk = self.call_stk + '-all_gather'
class all_gather_fwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)
        # self.call_stk = self.call_stk + '-all_gather'
    def bwd(self, args):
        pass

class all_gather_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-all_gather'
    def fwd(self, args):
        pass

class reduce_scatter(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('reduce_scatter'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-reduce_scatter'
class all_reduce(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_reduce'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-all_reduce'
class all2all(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

class send(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==0 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

class recv(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==1 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

class recv_prev(recv):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank-1)%pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        local_rank = 1
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True

class send_next(send):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank+1)%pp_size
        id = f"send_recv-{rank}-{next_rank}-{id}"
        local_rank = 0
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True

class recv_next(recv):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank+1)%pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        local_rank = 1
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True
class send_prev(send):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank-1)%pp_size
        id = f"send_recv-{rank}-{prev_rank}-{id}"
        local_rank = 0
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True

COM_BUFF={}
COM_BUFF=None
# COM_BUFF = Manager.dict()

def get_comm_group(strategy):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rank_info = get_rank_group(rank, strategy)
    group_name  = [
        "tp_group_id",
        "pp_group_id",
        "dp_group_id",
        "ep_group_id",
        "edp_group_id",
    ]
    local_group_id = {k:rank_info[k] for k in group_name}
    local_group = {k:[] for k in group_name}
    for i in range(size):
        rank_info_i = get_rank_group(i, strategy)
        for name in group_name:
            if rank_info_i[name] == local_group_id[name]:
                local_group[name].append(i)
    group = comm.Get_group()
    comm_group = {k.split('_id')[0]:comm.Create(group.Incl(ranks)) for k, ranks in local_group.items()}
    # for k,v in comm_group.items():
    #     print(f"{k} group={v}, size={v.Get_size()}")
    # comm_group = {k.split('_id')[0]:sub_comm.Create_group(ranks) for k, ranks in local_group.items()}
    return comm_group
