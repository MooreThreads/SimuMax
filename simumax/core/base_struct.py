"""Basic data structures for simumax"""

from copy import deepcopy
from dataclasses import dataclass, field
from abc import ABC
from typing import List, Tuple, Dict
from collections import defaultdict, deque
import heapq
import time
import types
import multiprocessing
try:
    from mpi4py import MPI
    enable_mpi = True
except ImportError:
    enable_mpi = False
import json
import os
from simumax.core.tensor import TensorSize
from simumax.core.config import StrategyConfig, SystemConfig, get_capture_graph_only, SIMU_DEBUG, TMP_PATH
from simumax.core.model_struct import (
    ActivationInfo,
    InputOutputInfo,
    ModuleComputeInfo,
    ModuleCostInfo,
    ModuleMemoryInfo,
    PathDebugContext,
    RecomputeStatus,
)
from simumax.core.simu_memory import OpMemoryProfile
from simumax.core.utils import get_point_name, to_json_string
from simumax.core.utils import get_rank_group
from simumax.core.graph import SimuONNXGraphBuilder

class FwdQue:
    def __init__(
        self,
        call_stk='',
        que=None,
        mem_profile: OpMemoryProfile = None,
        phase: str = "fwd",
        batch_blocking_comm: bool = False,
    ):
        self.que = que if que else []
        self.call_stk = call_stk
        self.st = None
        self.mem_profile = mem_profile
        self.phase = phase
        self._mem_started = False
        self._mem_finished = False
        self.batch_blocking_comm = batch_blocking_comm

    def step(self, t, ctx):
        # t is dict: {"comp","comm","off"}
        if self.st is None:
            self.st = t["comp"]
        if (
            self.mem_profile is not None
            and not self._mem_started
            and getattr(ctx, "memory_tracker", None) is not None
        ):
            ctx.memory_tracker.phase_start(
                rank=ctx.current_rank,
                ts=self.st,
                profile=self.mem_profile,
                phase=self.phase,
            )
            self._mem_started = True

        ok, blk = self._step(t, ctx)
        if ok:
            if (
                self.mem_profile is not None
                and not self._mem_finished
                and getattr(ctx, "memory_tracker", None) is not None
            ):
                ctx.memory_tracker.phase_end(
                    rank=ctx.current_rank,
                    ts=t["comp"],
                    profile=self.mem_profile,
                    phase=self.phase,
                )
                self._mem_finished = True
            info = f"{self.call_stk} {self.phase} cost {t['comp']-self.st:.6f} st {self.st:.6f} ed {t['comp']:.6f}"
            with open(ctx.log_path, 'a') as f:
                f.write(info+'\n')
            return True, None
        return False, blk

    def _step(self, t, ctx):
        if self.batch_blocking_comm:
            batch_submit_t = max(t["comp"], t["comm"])
            blocked_key = None
            remaining = []
            for op in self.que:
                if hasattr(op, "_prime_batch_submit"):
                    op._prime_batch_submit(self.phase, batch_submit_t)
                ok, blk = op.step(t, ctx)
                if ok:
                    continue
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        continue
                    if blk[0] in ("yield_done", "yield_keep"):
                        self.que = [op] + list(self.que[len(remaining) + 1 :])
                        return False, blk
                remaining.append(op)
                if blocked_key is None:
                    blocked_key = blk
            self.que = remaining
            if self.que:
                return False, blocked_key
            t["comp"] += 2e-3  # tracing
            return True, None

        while self.que:
            ok, blk = self.que[0].step(t, ctx)   # LeafModel.step now returns (ok, blk)
            if not ok:
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        self.que.pop(0)
                    if blk[0] in ("yield_done", "yield_keep"):
                        return False, blk
                return False, blk
            self.que.pop(0)

        t["comp"] += 2e-3  # tracing
        return True, None

    def append(self, x):
        self.que.append(x)

    def __bool__(self):
        return bool(self.que)


class BwdStk:
    def __init__(self, call_stk='', stk=None, mem_profile: OpMemoryProfile = None):
        self.stk = stk if stk else []
        self.call_stk = call_stk
        self.st_bwd = None
        self.mem_profile = mem_profile
        self._mem_started = False
        self._mem_finished = False

    def bwd(self, t, ctx):
        if self.st_bwd is None:
            self.st_bwd = t["comp"]
        if (
            self.mem_profile is not None
            and not self._mem_started
            and getattr(ctx, "memory_tracker", None) is not None
        ):
            ctx.memory_tracker.phase_start(
                rank=ctx.current_rank,
                ts=self.st_bwd,
                profile=self.mem_profile,
                phase="bwd",
            )
            self._mem_started = True

        ok, blk = self._bwd(t, ctx)
        if ok:
            if (
                self.mem_profile is not None
                and not self._mem_finished
                and getattr(ctx, "memory_tracker", None) is not None
            ):
                ctx.memory_tracker.phase_end(
                    rank=ctx.current_rank,
                    ts=t["comp"],
                    profile=self.mem_profile,
                    phase="bwd",
                )
                self._mem_finished = True
            info = f"{self.call_stk} bwd cost {t['comp']-self.st_bwd:.6f} st {self.st_bwd:.6f} ed {t['comp']:.6f}"
            with open(ctx.log_path, 'a') as f:
                f.write(info+'\n')
            return True, None
        return False, blk

    def _bwd(self, t, ctx):
        while self.stk:
            ok, blk = self.stk[-1].bwd(t, ctx)
            if not ok:
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        self.stk.pop(-1)
                    if blk[0] in ("yield_done", "yield_keep"):
                        return False, blk
                return False, blk
            self.stk.pop(-1)

        t["comp"] += 2e-3  # tracing
        return True, None

    def append(self, x):
        self.stk.append(x)

    def __bool__(self):
        return bool(self.stk)


class RecomputeBlockJob:
    """Replay a checkpointed forward block before running its backward."""

    def __init__(self, call_stk='', fwd_jobs=None, bwd_jobs=None):
        self.call_stk = call_stk
        self._has_recompute = bool(fwd_jobs)
        self.recompute_fwd = FwdQue(
            call_stk=f"{call_stk}-recompute_block",
            que=fwd_jobs if fwd_jobs else [],
            phase="recompute_fwd",
        )
        self.bwd_stk = BwdStk(
            call_stk=f"{call_stk}-checkpoint_bwd",
            stk=bwd_jobs if bwd_jobs else [],
        )
        self._recompute_done = False

    def bwd(self, t, ctx):
        if self._has_recompute and not self._recompute_done:
            ok, blk = self.recompute_fwd.step(t, ctx)
            if not ok:
                return False, blk
            self._recompute_done = True
        elif not self._has_recompute:
            self._recompute_done = True
        return self.bwd_stk.bwd(t, ctx)

    

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
        self.children_modules:List[MetaModule] = []  # Unordered list of all child modules.
        self.children_modules_names:Dict[MetaModule, str] = {}
        self.default_dtype = strategy.dtype 
        self._init_strategy = False
        self.input_info = None
        self.output_info_ = None
        # self.cache_info = []
        self.enable_recompute = False
        self.recompute_granularity = "full"
        self.enable_block_recompute_schedule = False
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
        self.use_variance_tail_model = bool(strategy.recompute_variance)
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
        if self.use_variance_tail_model:
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

    def get_root_module(self):
        module = self
        while getattr(module, "parent_module", None) is not None:
            module = module.parent_module
        return module

    def is_last_leaf_in_root(self):
        root = self.get_root_module()
        leaf_nodes = getattr(root, "all_leaf_nodes", None)
        return bool(leaf_nodes) and leaf_nodes[-1] is self

    def build_simu_mem_profile(self, phase: str = "fwd"):
        if not self.is_leaf_module or not self._info_ready:
            return None

        act_info = self.get_act_info()
        cache_size_bytes = 0
        cache_alloc_phase = None
        if self.strategy.enable_recompute and self.enable_recompute:
            recompute_peak_mem_no_cache = act_info.fwd_peak_mem_no_cache
            if self.recompute_status == RecomputeStatus.FIRST:
                if not self.offload_inputs:
                    cache_size_bytes = self.all_input_element_num()
                    cache_alloc_phase = "fwd"
            else:
                cache_size_bytes = act_info.total_activation_mem_cache
                cache_alloc_phase = "recompute_fwd"
        else:
            cache_size_bytes = act_info.total_activation_mem_cache
            cache_alloc_phase = "fwd"
            recompute_peak_mem_no_cache = 0

        if self.use_variance_tail_model and self.is_variance_node:
            if cache_alloc_phase == "recompute_fwd":
                cache_size_bytes = 0
                cache_alloc_phase = None

        bwd_peak_mem_no_cache = act_info.bwd_peak_mem_no_cache

        return OpMemoryProfile(
            op_name=self.full_name or self.call_stk,
            fwd_peak_mem_no_cache=int(act_info.fwd_peak_mem_no_cache),
            bwd_peak_mem_no_cache=int(bwd_peak_mem_no_cache),
            recompute_peak_mem_no_cache=int(recompute_peak_mem_no_cache),
            cache_size_bytes=int(cache_size_bytes),
            cache_alloc_phase=cache_alloc_phase,
            cache_token_scope=self.call_stk,
        )

    def prefill_fwd(self):
        fwd = FwdQue(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="fwd") if self.is_leaf_module else None,
        )
        for layer in self.layers:
            fwd.append(layer.prefill_fwd())
        return fwd

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        fwd = FwdQue(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="recompute_fwd") if self.is_leaf_module else None,
            phase="recompute_fwd",
        )
        recompute_cost = self._cost_info.recompute_compute_time if self.is_leaf_module else recompute_cost_override
        for layer in self.layers:
            fwd.append(layer.prefill_recompute_fwd(recompute_cost))
        return fwd

    def _use_block_recompute_schedule(self):
        if self.is_leaf_module or not self.enable_block_recompute_schedule:
            return False
        nodes = self.get_all_leaf_modules() if self.status_ready else self.layers
        return any(getattr(node, "enable_recompute", False) for node in nodes)

    def _append_checkpoint_segment(self, bwd, segment):
        if not segment:
            return
        recompute_jobs = [
            layer.prefill_recompute_fwd()
            for layer in segment
            if not (
                getattr(layer, "use_variance_tail_model", False)
                and getattr(layer, "is_variance_node", False)
            )
        ]
        bwd_jobs = [layer.prefill_bwd() for layer in segment]
        bwd.append(
            RecomputeBlockJob(
                call_stk=self.call_stk,
                fwd_jobs=recompute_jobs,
                bwd_jobs=bwd_jobs,
            )
        )

    def prefill_bwd(self):
        if self._use_block_recompute_schedule():
            bwd = BwdStk(call_stk=self.call_stk)
            nodes = self.get_all_leaf_modules() if self.status_ready else self.layers
            checkpoint_segment = []
            for node in nodes:
                if getattr(node, "enable_recompute", False):
                    if (
                        checkpoint_segment
                        and getattr(node, "recompute_status", RecomputeStatus.MIDDLE) == RecomputeStatus.FIRST
                    ):
                        self._append_checkpoint_segment(bwd, checkpoint_segment)
                        checkpoint_segment = []
                    checkpoint_segment.append(node)
                    if getattr(node, "recompute_status", RecomputeStatus.MIDDLE) == RecomputeStatus.LAST:
                        self._append_checkpoint_segment(bwd, checkpoint_segment)
                        checkpoint_segment = []
                    continue

                self._append_checkpoint_segment(bwd, checkpoint_segment)
                checkpoint_segment = []
                bwd.append(node.prefill_bwd())

            self._append_checkpoint_segment(bwd, checkpoint_segment)
            return bwd

        bwd = BwdStk(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="bwd") if self.is_leaf_module else None,
        )
        for layer in self.layers:
            bwd.append(layer.prefill_bwd())
        return bwd
        
    def get_all_leaf_modules(self):
        assert self.status_ready, f"{self.__class__.__name__} is not ready yet, please run set_first_last_recompute_status() first"
        return self.all_leaf_nodes

    def set_first_last_recompute_status(self):
        self.pre_enable_recompute = False
        self.p_recom_m: MetaModule = None
        self.all_recompute_nodes = []
        self.all_leaf_nodes = []

        def dfs(module: MetaModule):
            ordered = module.children_ordered_module or module.children_modules
            if module.is_leaf_module or len(ordered) == 0:
                module.call_idx = len(self.all_leaf_nodes)
                self.all_leaf_nodes.append(module)

                if module.enable_recompute:
                    module.recompute_status = RecomputeStatus.MIDDLE
                    self.all_recompute_nodes.append(module)

                if not self.pre_enable_recompute and module.enable_recompute:
                    module.recompute_status = RecomputeStatus.FIRST
                if self.pre_enable_recompute and not module.enable_recompute and self.p_recom_m is not None:
                    self.p_recom_m.recompute_status = RecomputeStatus.LAST
                if module.enable_recompute:
                    self.p_recom_m = module
                self.pre_enable_recompute = module.enable_recompute
                return

            for child in ordered:
                dfs(child)

        dfs(self)
        if self.pre_enable_recompute and self.p_recom_m is not None:
            self.p_recom_m.recompute_status = RecomputeStatus.LAST
    
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
        # TODO(sherry): support register hooks
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
    def main_grad_element_size(self):
        """Main gradient precision used by memory/communication modeling."""
        if self.strategy.grad_reduce_in_bf16 or (not self.strategy.use_fp32_accum_grad):
            return self.dtype_to_element_size["bf16"]
        return self.dtype_to_element_size["fp32"]

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
        self._act_info.bwd_peak_mem_no_cache = 0

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
            if self.use_variance_tail_model and self.is_variance_node:
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
            main_str += f"\n\t2. memory: (d_w={module_info.dense_weight_bytes}, d_g={module_info.dense_grad_bytes}, d_s={module_info.dense_state_bytes}, m_w={module_info.moe_weight_bytes}, m_g={module_info.moe_grad_bytes}, m_s={module_info.moe_state_bytes})"

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

    def _record_te_dummy_wgrad_shape(self, output_size=None, input_size=None, grouped_linear=False):
        version_enabled = (
            self.strategy.te_grouped_linear_dummy_wgrad_memory_enabled
            if grouped_linear
            else self.strategy.te_dummy_wgrad_memory_enabled
        )
        if not (
            self.strategy.use_fused_grad_accumulation
            and version_enabled
        ):
            return
        output_size = self.output_size if output_size is None else output_size
        input_size = self.input_size if input_size is None else input_size
        # TE caches dummy tensors by (rows, cols, dtype). The dtype is the parameter dtype,
        # not the main_grad accumulation dtype.
        elem_size = self.dtype_to_element_size.get(self.strategy.dtype, self.dtype_to_element_size["bf16"])
        self._model_info.te_dummy_wgrad_shapes.add((int(output_size), int(input_size), int(elem_size)))
    
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
            return dict(B=[bs, bs, 1], M=[seq_len, seq_len, out], K=[inp, out, bs*seq_len], N=[out, inp, inp], layout=['TN', 'NN', 'NT'], accumulate=[False, False, True], out_dtype=['bf16', 'bf16', 'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'])


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


class BarrierBackend:
    def __init__(self):
        # gid -> state
        self.st = {}
        self.done = {}  # gid -> (end_t, set(waiters))

    def arrive(self, gid, rank, ready_t, expected, cost):
        # If this gid already completed and the same rank participated in that
        # completion, return the cached done state directly.
        d = self.done.get(gid)
        if d is not None:
            end_t, waiters = d
            if rank in waiters:
                return True, list(waiters), end_t

        s = self.st.get(gid)
        if s is None:
            s = {"expected": expected, "arrived": 0, "max_ready": 0.0, "waiters": [], "cost": cost}
            self.st[gid] = s
        elif rank in s["waiters"]:
            # Blocking comm jobs may be retried locally while still waiting for
            # their peer. Keep the original arrival instead of double-counting
            # the same rank and spuriously completing the barrier.
            return False, None, None

        s["arrived"] += 1
        s["max_ready"] = max(s["max_ready"], ready_t)
        s["waiters"].append(rank)

        if s["arrived"] == s["expected"]:
            end_t = s["max_ready"] + s["cost"]
            waiters = set(s["waiters"])
            del self.st[gid]
            self.done[gid] = (end_t, waiters)  # Cache completion for local retries.
            return True, list(waiters), end_t

        return False, None, None


class P2PBackend:
    """Dedicated backend for point-to-point send_recv-* rendezvous."""

    def __init__(self):
        self.st = {}
        self.done = {}

    def arrive(self, gid, rank, ready_t, cost):
        d = self.done.get(gid)
        if d is not None:
            end_t, waiters = d
            if rank in waiters:
                return True, list(waiters), end_t

        s = self.st.get(gid)
        if s is None:
            s = {"arrived": 0, "waiters": [], "arrivals": []}
            self.st[gid] = s
        elif rank in s["waiters"]:
            return False, None, None

        s["arrived"] += 1
        s["waiters"].append(rank)
        s["arrivals"].append((rank, ready_t, cost))

        if s["arrived"] == 2:
            end_t = max(arrival_ready + arrival_cost for _, arrival_ready, arrival_cost in s["arrivals"])
            waiters = set(s["waiters"])
            del self.st[gid]
            self.done[gid] = (end_t, waiters)
            return True, list(waiters), end_t

        return False, None, None


@dataclass
class CommEntry:
    eid: int
    rank: int
    gid: tuple
    cost: float
    issue_t: float
    stream: str
    mode: str
    backend_kind: str
    expected: int | None = None
    status: str = "queued"
    ready_t: float | None = None
    launch_t: float | None = None
    end_t: float | None = None
    log_call_stk: str | None = None
    log_id: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class AsyncP2PState:
    gid: tuple
    cost: float = 0.0
    ready_t: float | None = None
    pair_logged: bool = False
    finalize_enqueued: bool = False
    post_unblock_enqueued: bool = False
    send_rank: int | None = None
    recv_rank: int | None = None
    send_eid: int | None = None
    recv_eid: int | None = None
    send_post_t: float | None = None
    recv_post_t: float | None = None
    send_post_order: int | None = None
    recv_post_order: int | None = None
    send_meta: dict = field(default_factory=dict)
    recv_meta: dict = field(default_factory=dict)


class State_Thread:
    def __init__(self):
        self.comm_order = 0    
    
class SimuThread:
    def __init__(self, rank=None):
        self.rank = rank  # Exposed so SimuSystem can manage per-rank scheduling.
        self.job = []
        self.t = defaultdict(float, {"comp": 0.0, "comm": 0.0, "off": 0.0})
        self.thread_state = State_Thread()

    def _sync_time(self):
        # Optional lane merge for legacy behavior.
        m = max(self.t.values()) if self.t else 0.0
        for lane in list(self.t.keys()):
            self.t[lane] = m

    def step(self, ctx):
        ctx.current_rank = self.rank
        progressed = False
        while self.job:
            if isinstance(self.job[0], FwdQue):
                ok, blk = self.job[0].step(self.t, ctx)  # Returns (ok, block_key).
                if not ok:
                    if ctx.sync_lanes:
                        self._sync_time()
                    return "BLOCKED", blk
            else:
                ok, blk = self.job[0].bwd(self.t, ctx)
                if not ok:
                    if ctx.sync_lanes:
                        self._sync_time()
                    return "BLOCKED", blk

            progressed = True

            if not self.job[0]:
                self.job.pop(0)

            if ctx.sync_lanes:
                self._sync_time()

        return ("DONE", None) if not progressed else ("PROGRESSED", None)



class SimuSystem:
    def __init__(self):
        self.threads = []  # thread must have .rank and .step()

    def simu(self, ctx):
        # ctx.backend: BarrierBackend
        # ctx.threads_by_rank: dict[int, thread]
        threads_by_rank = {th.rank: th for th in self.threads}
        ctx.threads_by_rank = threads_by_rank

        ver = {r: 0 for r in threads_by_rank}
        blocked = set()
        heap = []
        blocked_on = {} 
        def cur_time(r):
            th = threads_by_rank[r]
            if ctx.sync_lanes:
                return max(th.t.values()) if th.t else 0.0
            # Overlap mode: schedule rank as soon as one lane can make progress.
            active = [t for lane, t in th.t.items() if lane != "off"]
            return min(active) if active else 0.0

        def push(r):
            ver[r] += 1
            heapq.heappush(heap, (cur_time(r), r, ver[r]))

        for r in threads_by_rank:
            push(r)

        done = set()
        while len(done) < len(threads_by_rank):
            if not heap:
                print("DEADLOCK: heap empty")
                print("done", len(done), "blocked", len(blocked), "total", len(threads_by_rank))
                # Print ranks that have not finished yet.
                alive = [r for r in threads_by_rank if r not in done]
                print("alive ranks:", alive[:50], "..." if len(alive) > 50 else "")
                print("blocked_on sample:", list(blocked_on.items())[:20])
                blocked_gids = [key[1] for key in blocked_on.values() if isinstance(key, tuple) and len(key) > 1 and key[0] == "async_wait"]
                if blocked_gids:
                    async_meta = {}
                    for gid in blocked_gids[:12]:
                        state = ctx.get_async_state(gid)
                        async_meta[gid] = {
                            "send": state.send_meta,
                            "recv": state.recv_meta,
                            "send_rank": state.send_rank,
                            "recv_rank": state.recv_rank,
                            "send_post": state.send_post_t,
                            "recv_post": state.recv_post_t,
                        }
                    print("blocked async meta:", async_meta)
                if hasattr(ctx, "rank_comm_queue"):
                    queue_state = {}
                    for rr, q in ctx.rank_comm_queue.items():
                        if q:
                            queue_state[rr] = list(q)[:6]
                    print("rank_comm_queue sample:", dict(list(queue_state.items())[:20]))
                if hasattr(ctx, "comm_entries"):
                    sample_entries = {}
                    for rr, q in getattr(ctx, "rank_comm_queue", {}).items():
                        if q:
                            eid = q[0]
                            sample_entries[rr] = ctx.comm_entries.get(eid)
                    print("head comm entries:", sample_entries)
                if hasattr(ctx, "pending_async_posts"):
                    print("pending_async_posts:", ctx.pending_async_posts[:20])
                if hasattr(ctx, "async_states"):
                    async_state_sample = {}
                    for i, (gid, state) in enumerate(ctx.async_states.items()):
                        async_state_sample[gid] = {
                            "ready_t": state.ready_t,
                            "send_rank": state.send_rank,
                            "recv_rank": state.recv_rank,
                            "send_post_t": state.send_post_t,
                            "recv_post_t": state.recv_post_t,
                            "send_eid": state.send_eid,
                            "recv_eid": state.recv_eid,
                            "pair_logged": state.pair_logged,
                        }
                        if i >= 19:
                            break
                    print("async_states sample:", async_state_sample)
                # Print unfinished gids still tracked by the barrier backend.
                print("pending barriers:", len(ctx.backend.st))
                # Print a few concrete gids with expected/arrived counts.
                for i, (gid, s) in enumerate(ctx.backend.st.items()):
                    # if i >= 10: break
                    print(gid, "arrived", s["arrived"], "expected", s["expected"], "waiters_sample", s["waiters"][:8])
                raise RuntimeError("deadlock")

            t, r, v = heapq.heappop(heap)
            if v != ver[r]:
                continue
            if r in blocked or r in done:
                continue

            status, key = threads_by_rank[r].step(ctx)  # run-until-block
            ctx.pump_comm_queue()
            if status == "BLOCKED":
                blocked_on[r] = key

            # Handle completions triggered by this step via pending_completions.
            while ctx.pending_completions:
                gid, waiters, end_t, stream = ctx.pending_completions.pop()
                for w in waiters:
                    th = threads_by_rank[w]
                    # Blocking collectives are synchronous at rank level:
                    # once completed, both compute and comm lanes should observe end_t.
                    th.t["comm"] = max(th.t["comm"], end_t)
                    th.t["comp"] = max(th.t["comp"], end_t)
                    if stream not in ("comm", "comp"):
                        th.t[stream] = max(th.t[stream], end_t)

                    # Only unblock ranks that are actually waiting on this gid.
                    if blocked_on.get(w) == ("barrier", gid):
                        del blocked_on[w]
                        push(w)
            while ctx.pending_comm_entry_completions:
                eid = ctx.pending_comm_entry_completions.pop()
                to_unblock = [w for w, wait_key in list(blocked_on.items()) if wait_key == ("comm_entry", eid)]
                for w in to_unblock:
                    del blocked_on[w]
                    push(w)
            ctx.flush_async_pair_logs()
            while ctx.pending_async_posts:
                gid = ctx.pop_async_post_unblock()
                to_unblock = [w for w, key in list(blocked_on.items()) if key in (("async_recv", gid), ("async_wait", gid))]
                for w in to_unblock:
                    del blocked_on[w]
                    push(w)
                    
            if status == "DONE":
                done.add(r)
                continue
            if status == "BLOCKED":
                if isinstance(key, tuple) and key and key[0] in ("yield", "yield_done", "yield_keep"):
                    blocked_on.pop(r, None)
                    push(r)
                    continue
                continue

            # PROGRESSED
            push(r)



        # Iteration end is the latest completed lane across all ranks.
        end_t = 0.0
        for th in threads_by_rank.values():
            if th.t:
                end_t = max(end_t, max(th.t.values()))
        print(f'end in {end_t}')
        return end_t

class SimuContext:
    def __init__(self, backend, merge_lanes=True, log_path='./tmp/log.log', sync_lanes=False):
        self.backend = backend
        self.p2p_backend = P2PBackend()
        self.pending_completions = []  # list[(waiters, end_t, stream)]
        self.pending_comm_entry_completions = []  # list[eid]
        self.pending_async_finalizations = []  # list[gid], LIFO for compatibility
        self.pending_async_posts = []  # list[gid], LIFO for compatibility
        self.pending_async_slot_releases = []  # legacy, unused in single-stream async p2p
        self.async_states = {}  # gid -> AsyncP2PState
        self.host_issue_seq = 0
        self.comm_entry_seq = 0
        self.comm_entries = {}  # eid -> entry dict
        self.rank_comm_queue = {}  # rank -> deque[eid]
        self.rank_comm_tail = {}  # rank -> end_t of last completed comm entry
        self.threads_by_rank = None
        self.merge_lanes = merge_lanes
        self.sync_lanes = sync_lanes
        self.log_path = log_path
        self.current_rank = None
        self.memory_tracker = None


    @staticmethod
    def comm_lane_key(rank, stream):
        return (rank, stream)

    def get_async_state(self, gid):
        state = self.async_states.get(gid)
        if state is None:
            state = AsyncP2PState(gid=gid)
            self.async_states[gid] = state
        return state

    @staticmethod
    def p2p_channel(op_id):
        if "-backward-" in op_id:
            return "backward"
        return "forward"

    def register_async_send(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        order,
        call_stk,
        log_id,
    ):
        state = self.get_async_state(gid)
        state.cost = cost
        state.send_rank = rank
        state.send_post_t = post_t
        state.send_post_order = order
        state.send_meta = {"call_stk": call_stk, "id": log_id}

    def register_async_recv(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        order,
        call_stk,
        log_id,
    ):
        state = self.get_async_state(gid)
        state.cost = cost
        state.recv_rank = rank
        state.recv_post_t = post_t
        state.recv_post_order = order
        state.recv_meta = {"call_stk": call_stk, "id": log_id}

    def post_async_send_entry(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        stream,
        mode,
        call_stk,
        log_id,
    ):
        order = self.next_issue_seq()
        self.register_async_send(
            gid=gid,
            rank=rank,
            post_t=post_t,
            cost=cost,
            order=order,
            call_stk=call_stk,
            log_id=log_id,
        )
        eid = self.issue_comm_entry(
            rank=rank,
            gid=gid,
            cost=cost,
            issue_t=post_t,
            stream=stream,
            mode=mode,
            backend_kind="p2p",
            expected=2,
            log_call_stk=call_stk,
            log_id=log_id,
            meta={"post_order": order, "post_ts": post_t},
        )
        self.attach_async_send_eid(gid, eid)
        self.pump_comm_queue()
        return eid

    def post_async_recv_entry(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        stream,
        mode,
        call_stk,
        log_id,
    ):
        order = self.next_issue_seq()
        self.register_async_recv(
            gid=gid,
            rank=rank,
            post_t=post_t,
            cost=cost,
            order=order,
            call_stk=call_stk,
            log_id=log_id,
        )
        eid = self.issue_comm_entry(
            rank=rank,
            gid=gid,
            cost=cost,
            issue_t=post_t,
            stream=stream,
            mode=mode,
            backend_kind="p2p",
            expected=2,
            log_call_stk=call_stk,
            log_id=log_id,
            meta={"post_order": order, "post_ts": post_t},
        )
        self.attach_async_recv_eid(gid, eid)
        self.pump_comm_queue()
        return eid

    def attach_async_send_eid(self, gid, eid):
        state = self.get_async_state(gid)
        state.send_eid = eid
        if state.send_meta is not None:
            state.send_meta["eid"] = eid

    def attach_async_recv_eid(self, gid, eid):
        state = self.get_async_state(gid)
        state.recv_eid = eid
        if state.recv_meta is not None:
            state.recv_meta["eid"] = eid

    def get_async_send_eid(self, gid):
        state = self.get_async_state(gid)
        if state.send_eid is not None:
            return state.send_eid
        return state.send_meta.get("eid")

    def get_async_recv_eid(self, gid):
        state = self.get_async_state(gid)
        if state.recv_eid is not None:
            return state.recv_eid
        return state.recv_meta.get("eid")

    def has_async_posted_send(self, gid):
        return self.get_async_state(gid).send_post_t is not None

    def has_async_posted_recv(self, gid):
        return self.get_async_state(gid).recv_post_t is not None

    def set_async_ready_t(self, gid, ready_t):
        state = self.get_async_state(gid)
        state.ready_t = ready_t

    def get_async_ready_t(self, gid):
        return self.get_async_state(gid).ready_t

    def queue_async_post_unblock(self, gid):
        state = self.get_async_state(gid)
        if state.post_unblock_enqueued:
            return
        self.pending_async_posts.append(gid)
        state.post_unblock_enqueued = True

    def pop_async_post_unblock(self):
        gid = self.pending_async_posts.pop()
        self.get_async_state(gid).post_unblock_enqueued = False
        return gid

    def queue_async_finalize(self, gid):
        state = self.get_async_state(gid)
        if state.finalize_enqueued:
            return
        self.pending_async_finalizations.append(gid)
        state.finalize_enqueued = True

    def pop_async_finalize(self):
        gid = self.pending_async_finalizations.pop()
        self.get_async_state(gid).finalize_enqueued = False
        return gid

    def flush_async_pair_logs(self):
        while self.pending_async_finalizations:
            gid = self.pop_async_finalize()
            self.emit_async_pair_logs(gid)

    def next_issue_seq(self):
        seq = self.host_issue_seq
        self.host_issue_seq += 1
        return seq

    def next_comm_entry_seq(self):
        seq = self.comm_entry_seq
        self.comm_entry_seq += 1
        return seq

    def issue_comm_entry(
        self,
        *,
        rank,
        gid,
        cost,
        issue_t,
        stream,
        mode,
        backend_kind,
        expected=None,
        log_call_stk=None,
        log_id=None,
        meta=None,
    ):
        eid = self.next_comm_entry_seq()
        entry = {
            "eid": eid,
            "rank": rank,
            "gid": gid,
            "cost": cost,
            "issue_t": issue_t,
            "stream": stream,
            "mode": mode,
            "backend_kind": backend_kind,
            "expected": expected,
            "log_call_stk": log_call_stk,
            "log_id": log_id,
            "meta": meta or {},
        }
        self.comm_entries[eid] = CommEntry(**entry)
        lane_key = self.comm_lane_key(rank, stream)
        self.rank_comm_queue.setdefault(lane_key, deque()).append(eid)
        return eid

    def get_entry(self, eid):
        return self.comm_entries.get(eid)

    def entry_done(self, eid):
        entry = self.comm_entries.get(eid)
        return bool(entry) and entry.status == "done"

    def get_entry_end(self, eid):
        entry = self.comm_entries.get(eid)
        return None if entry is None else entry.end_t

    def get_rank_comm_tail(self, rank, stream):
        return self.rank_comm_tail.get(self.comm_lane_key(rank, stream), 0.0)

    def _complete_comm_entry(self, eid, launch_t, end_t):
        entry = self.comm_entries[eid]
        rank = entry.rank
        lane_key = self.comm_lane_key(rank, entry.stream)
        queue = self.rank_comm_queue.setdefault(lane_key, deque())
        if not queue or queue[0] != eid:
            raise RuntimeError(
                f"comm queue out of order on lane {lane_key}: expected head {eid}, got {queue[0] if queue else None}"
            )
        if launch_t + 1e-9 < self.get_rank_comm_tail(rank, entry.stream):
            raise RuntimeError(
                f"comm launch regressed on lane {lane_key}: launch_t={launch_t}, "
                f"tail={self.get_rank_comm_tail(rank, entry.stream)}, gid={entry.gid}"
            )
        entry.status = "done"
        entry.launch_t = launch_t
        entry.end_t = end_t
        queue.popleft()
        self.rank_comm_tail[lane_key] = end_t
        if self.threads_by_rank is not None and rank in self.threads_by_rank:
            self.threads_by_rank[rank].t[entry.stream] = max(self.threads_by_rank[rank].t[entry.stream], end_t)
        self.pending_comm_entry_completions.append(eid)
        self._maybe_finalize_async_ready(entry.gid)
        self._queue_async_finalize(entry.gid)
    def _maybe_finalize_async_ready(self, gid):
        state = self.get_async_state(gid)
        if state.ready_t is not None:
            return state.ready_t
        send_eid = self.get_async_send_eid(gid)
        recv_eid = self.get_async_recv_eid(gid)
        if send_eid is None or recv_eid is None:
            return None
        if not self.entry_done(send_eid) or not self.entry_done(recv_eid):
            return None
        send_entry = self.get_entry(send_eid)
        recv_entry = self.get_entry(recv_eid)
        if not send_entry or not recv_entry:
            return None
        if send_entry.end_t is None or recv_entry.end_t is None:
            return None
        ready_t = max(send_entry.end_t, recv_entry.end_t)
        self.set_async_ready_t(gid, ready_t)
        self.queue_async_post_unblock(gid)
        return ready_t

    def _queue_async_finalize(self, gid):
        state = self.get_async_state(gid)
        if state.pair_logged:
            return
        if self._maybe_finalize_async_ready(gid) is None:
            return
        self.queue_async_finalize(gid)

    def _pump_local_entry(self, eid):
        entry = self.comm_entries[eid]
        rank = entry.rank
        launch_t = max(entry.issue_t, self.get_rank_comm_tail(rank, entry.stream))
        end_t = launch_t + entry.cost
        self._complete_comm_entry(eid, launch_t, end_t)

    def _pump_rendezvous_entry(self, eid):
        entry = self.comm_entries[eid]
        rank = entry.rank
        if entry.status == "done":
            return
        if entry.status == "waiting":
            # This rank has already arrived at the rendezvous. Re-arriving the
            # same queued head would double-count the participant and can make
            # a p2p/collective appear to complete locally before its peer(s)
            # actually arrive.
            return
        ready_t = max(entry.issue_t, self.get_rank_comm_tail(rank, entry.stream))
        entry.ready_t = ready_t
        if entry.backend_kind == "p2p":
            done, waiters, end_t = self.p2p_backend.arrive(entry.gid, rank, ready_t, entry.cost)
        else:
            done, waiters, end_t = self.backend.arrive(
                entry.gid, rank, ready_t, entry.expected, entry.cost
            )
        entry.status = "waiting"
        if not done:
            return
        for waiter_rank in waiters:
            # Find the matching head on any lane for this waiting gid.
            queue = None
            waiter_eid = None
            waiter_entry = None
            for lane_key, candidate_queue in self.rank_comm_queue.items():
                if lane_key[0] != waiter_rank or not candidate_queue:
                    continue
                candidate_eid = candidate_queue[0]
                candidate_entry = self.comm_entries[candidate_eid]
                if candidate_entry.gid == entry.gid:
                    queue = candidate_queue
                    waiter_eid = candidate_eid
                    waiter_entry = candidate_entry
                    break
            if queue is None:
                raise RuntimeError(f"comm completion without queued head on rank {waiter_rank} for {entry.gid}")
            if waiter_entry.gid != entry.gid:
                raise RuntimeError(
                    f"comm completion gid mismatch on rank {waiter_rank}: head={waiter_entry.gid} done={entry.gid}"
                )
            waiter_ready_t = waiter_entry.ready_t
            if waiter_ready_t is None:
                waiter_ready_t = max(
                    waiter_entry.issue_t, self.get_rank_comm_tail(waiter_rank, waiter_entry.stream)
                )
                waiter_entry.ready_t = waiter_ready_t
            launch_t = max(waiter_ready_t, end_t - waiter_entry.cost)
            self._complete_comm_entry(waiter_eid, launch_t, end_t)
        return

    def pump_comm_queue(self):
        progressed = True
        while progressed:
            progressed = False
            for lane_key in sorted(self.rank_comm_queue):
                queue = self.rank_comm_queue.get(lane_key)
                if not queue:
                    continue
                eid = queue[0]
                entry = self.comm_entries[eid]
                before_status = entry.status
                if entry.backend_kind == "local":
                    self._pump_local_entry(eid)
                else:
                    self._pump_rendezvous_entry(eid)
                if self.entry_done(eid) or self.comm_entries[eid].status != before_status:
                    progressed = True

    def ensure_async_ready(self, gid):
        state = self.get_async_state(gid)
        ready_t = self._maybe_finalize_async_ready(gid)
        if ready_t is None:
            self.pump_comm_queue()
            ready_t = self._maybe_finalize_async_ready(gid)
        return ready_t

    def emit_async_pair_logs(self, gid):
        state = self.get_async_state(gid)
        if state.pair_logged:
            return state.ready_t
        ready_t = state.ready_t
        if ready_t is None:
            return None
        send_eid = self.get_async_send_eid(gid)
        recv_eid = self.get_async_recv_eid(gid)
        send_entry = self.get_entry(send_eid)
        recv_entry = self.get_entry(recv_eid)
        if not send_entry or not recv_entry:
            return None
        if send_entry.end_t is None or recv_entry.end_t is None:
            return None
        send_meta = state.send_meta
        recv_meta = state.recv_meta
        if send_meta is not None and recv_meta is not None:
            send_post = state.send_post_t if state.send_post_t is not None else send_entry.launch_t
            recv_post = state.recv_post_t if state.recv_post_t is not None else recv_entry.launch_t
            send_order = state.send_post_order if state.send_post_order is not None else -1
            recv_order = state.recv_post_order if state.recv_post_order is not None else -1
            send_line = (
                f"{send_meta['call_stk']} gid {send_meta['id']} {gid[0]} cost {send_entry.end_t-send_entry.launch_t:.6f} "
                f"st {send_entry.launch_t:.6f} ed {send_entry.end_t:.6f} post {send_post:.6f} order {send_order}"
            )
            recv_line = (
                f"{recv_meta['call_stk']} gid {recv_meta['id']} {gid[0]} cost {recv_entry.end_t-recv_entry.launch_t:.6f} "
                f"st {recv_entry.launch_t:.6f} ed {recv_entry.end_t:.6f} post {recv_post:.6f} order {recv_order}"
            )
            with open(self.log_path, "a") as f:
                f.write(send_line + "\n")
                f.write(recv_line + "\n")
                if ready_t > recv_entry.end_t + 1e-9:
                    wait_call_stk = recv_meta["call_stk"].replace("-async_recv", "-async_wait_recv")
                    f.write(
                        f"{wait_call_stk} gid {recv_meta['id']} {gid[0]} cost {ready_t-recv_entry.end_t:.6f} "
                        f"st {recv_entry.end_t:.6f} ed {ready_t:.6f} post {recv_post:.6f} order {recv_order}\n"
                    )
        state.pair_logged = True
        return ready_t

    def finalize_async_p2p(self, gid, stream="comm"):
        ready_t = self.ensure_async_ready(gid)
        if ready_t is None:
            return None
        return self.emit_async_pair_logs(gid)


class LeafModel():
    def __init__(self, specific_name=''):
        self.st = None
        self.st_bwd = None
        self.call_stk =f'-{self.__class__.__name__}'
        self.forward_op = "fwd"
        if specific_name:
            self.call_stk =f'-{specific_name}'

    # def step(self, t, ctx):
    #     # Default behavior is to call _step; subclasses can override it.
    #     out = self._step(t, ctx)
    #     return out if isinstance(out, tuple) else (bool(out), None)
    
    def step(self, t, ctx):
        # t is dict: {"comp","comm","off"}
        if self.st is None:
            self.st = t["comp"]

        out = self._step(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            if t['comp'] == self.st:
                return True, None
            info = f"{self.call_stk} {self.forward_op} cost {t['comp']-self.st:.6f} st {self.st:.6f} ed {t['comp']:.6f}"
            with open(ctx.log_path, 'a') as f:
                f.write(info+'\n')
            return True, None
        return False, blk
    
    # def bwd(self, t, ctx):
    #     out = self._bwd(t, ctx)
    #     return out if isinstance(out, tuple) else (bool(out), None)

    def bwd(self, t, ctx):
        if self.st_bwd is None:
            self.st_bwd = t["comp"]
        out = self._bwd(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            if t['comp'] == self.st_bwd:
                return True, None
            info = f"{self.call_stk} bwd cost {t['comp']-self.st_bwd:.6f} st {self.st_bwd:.6f} ed {t['comp']:.6f}"
            with open(ctx.log_path, 'a') as f:
                f.write(info+'\n')
            return True, None
        return False, blk
    
    def _step(self, t, ctx):
        return True  # Default leaf behavior: no blocking.

    def _bwd(self, t, ctx):
        return True
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
    
    def prefill_fwd(self):
        return self

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        return self.prefill_fwd()

    def prefill_bwd(self):
        return self


    
class AtomModel(LeafModel):
    #simplify LeafModel with cost information
    def __init__(self, fwd_cost, bwd_cost, specific_name='', recompute_cost=None):
        super().__init__(specific_name)
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.recompute_cost = fwd_cost if recompute_cost is None else recompute_cost
        # self.fwd_cost = fwd_cost*(1+random.random()*0.6)
        # self.bwd_cost = bwd_cost*(1+random.random()*0.6)
    def _step(self, t, ctx):
        t["comp"] += self.fwd_cost
        return True

    def _bwd(self, t, ctx):
        t["comp"] += self.bwd_cost
        return True

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        recompute_cost = self.recompute_cost if recompute_cost_override is None else recompute_cost_override
        clone = AtomModel(
            fwd_cost=recompute_cost,
            bwd_cost=self.bwd_cost,
            recompute_cost=recompute_cost,
        )
        clone.call_stk = self.call_stk
        clone.forward_op = "recompute_fwd"
        return clone
    
class Com(LeafModel):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0,
                 call_stk='', global_rank=None, stream="comm"):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.rank = rank
        self.group_size = group_size
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.global_rank = global_rank
        self.stream = stream
        self._completed = set()  # store completed gid for this rank/op
        self._fwd_launch_st = None
        self._bwd_launch_st = None
        self._fwd_issue_order = None
        self._bwd_issue_order = None
        self._fwd_entry_eid = None
        self._bwd_entry_eid = None
        self._blocking_start_by_gid = {}
        self._fwd_done_t = None
        self._bwd_done_t = None
        self._batch_submit_by_gid = {}

    def _prime_batch_submit(self, phase, submit_t):
        gid = (phase, self.id)
        self._batch_submit_by_gid.setdefault(gid, submit_t)

    def _event_start_t(self, entry):
        # For rendezvous/barrier-style communications, the profile-visible
        # event should include local waiting before common completion.
        if entry.backend_kind == "barrier" or self.id.startswith("send_recv-"):
            return entry.issue_t
        return entry.launch_t

    def step(self, t, ctx):
        out = self._step(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            done_t = self._fwd_done_t if self._fwd_done_t is not None else t[self.stream]
            if self._fwd_launch_st is None or done_t == self._fwd_launch_st:
                return True, None
            info = (
                f"{self.call_stk} gid {self.id} fwd cost {done_t-self._fwd_launch_st:.6f} "
                f"st {self._fwd_launch_st:.6f} ed {done_t:.6f}"
            )
            with open(ctx.log_path, "a") as f:
                f.write(info + "\n")
            self._fwd_launch_st = None
            self._fwd_done_t = None
            return True, None
        return False, blk

    def bwd(self, t, ctx):
        out = self._bwd(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            done_t = self._bwd_done_t if self._bwd_done_t is not None else t[self.stream]
            if self._bwd_launch_st is None or done_t == self._bwd_launch_st:
                return True, None
            info = (
                f"{self.call_stk} gid {self.id} bwd cost {done_t-self._bwd_launch_st:.6f} "
                f"st {self._bwd_launch_st:.6f} ed {done_t:.6f}"
            )
            with open(ctx.log_path, "a") as f:
                f.write(info + "\n")
            self._bwd_launch_st = None
            self._bwd_done_t = None
            return True, None
        return False, blk
        
    def _step(self, t, ctx):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")

        if self.fwd_cost == 0 or self.group_size <= 1:
            return True, None

        gid = ("fwd", self.id)
        if gid in self._completed:
            return True, None

        if self._fwd_issue_order is None:
            self._fwd_issue_order = ctx.next_issue_seq()
        if self._fwd_entry_eid is None:
            expected = 2 if self.id.startswith("send_recv-") else self.group_size
            backend_kind = "barrier"
            if self.id.startswith("send_recv-"):
                backend_kind = "p2p"
            elif ctx.merge_lanes and 'default_group' not in self.id:
                backend_kind = "local"
            elif 'default_group' in self.id:
                expected = int(self.id.split('pp_size:')[1])
            self._fwd_entry_eid = ctx.issue_comm_entry(
                rank=self.global_rank,
                gid=gid,
                cost=self.fwd_cost,
                issue_t=t["comp"],
                stream=self.stream,
                mode="sync",
                backend_kind=backend_kind,
                expected=expected,
                log_call_stk=self.call_stk,
                log_id=self.id,
            )
            ctx.pump_comm_queue()
        if not ctx.entry_done(self._fwd_entry_eid):
            return False, ("comm_entry", self._fwd_entry_eid)
        end_t = ctx.get_entry_end(self._fwd_entry_eid)
        entry = ctx.get_entry(self._fwd_entry_eid)
        self._fwd_launch_st = self._event_start_t(entry)
        self._fwd_done_t = end_t
        t[self.stream] = max(t[self.stream], end_t)
        t["comp"] = max(t["comp"], end_t)
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")
        if self.bwd_cost == 0 or self.group_size <= 1:
            return True, None
        gid = ("bwd", self.id)
        if gid in self._completed:
            return True, None
        if self._bwd_issue_order is None:
            self._bwd_issue_order = ctx.next_issue_seq()
        if self._bwd_entry_eid is None:
            expected = 2 if self.id.startswith("send_recv-") else self.group_size
            backend_kind = "barrier"
            if self.id.startswith("send_recv-"):
                backend_kind = "p2p"
            elif ctx.merge_lanes and 'default_group' not in self.id:
                backend_kind = "local"
            elif 'default_group' in self.id:
                expected = int(self.id.split('pp_size:')[1])
            self._bwd_entry_eid = ctx.issue_comm_entry(
                rank=self.global_rank,
                gid=gid,
                cost=self.bwd_cost,
                issue_t=t["comp"],
                stream=self.stream,
                mode="sync",
                backend_kind=backend_kind,
                expected=expected,
                log_call_stk=self.call_stk,
                log_id=self.id,
            )
            ctx.pump_comm_queue()
        if not ctx.entry_done(self._bwd_entry_eid):
            return False, ("comm_entry", self._bwd_entry_eid)
        end_t = ctx.get_entry_end(self._bwd_entry_eid)
        entry = ctx.get_entry(self._bwd_entry_eid)
        self._bwd_launch_st = self._event_start_t(entry)
        self._bwd_done_t = end_t
        t[self.stream] = max(t[self.stream], end_t)
        t["comp"] = max(t["comp"], end_t)
        self._completed.add(gid)
        return True, None

    def _blocking_step_impl(self, t, ctx, *, phase):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")
        cost = self.fwd_cost if phase == "fwd" else self.bwd_cost
        if cost == 0 or self.group_size <= 1:
            return True, None
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        m = max(t["comp"], t["comm"])
        t["comp"] = t["comm"] = m
        ready_t = self._batch_submit_by_gid.get(gid, t[self.stream])
        done, waiters, end_t = ctx.backend.arrive(gid, self.global_rank, ready_t, 2, cost)
        if not done:
            self._blocking_start_by_gid.setdefault(gid, ready_t)
            return False, ("barrier", gid)
        # Blocking communication should cover the local call interval from the
        # moment this rank enters the communication until the common completion
        # time. Any rendezvous wait is part of the visible blocking comm span.
        event_start_t = self._blocking_start_by_gid.pop(gid, ready_t)
        done_t = end_t
        if phase == "fwd":
            self._fwd_launch_st = event_start_t
            self._fwd_done_t = done_t
        else:
            self._bwd_launch_st = event_start_t
            self._bwd_done_t = done_t
        # Retried blocking p2p ops may observe a cached completion whose end_t
        # is earlier than the rank's current visible time (for example, when a
        # longer sibling op in the same batch finished later). Never move local
        # time backwards on replay.
        end_t = max(end_t, t["comp"], t["comm"])
        t["comp"] = t["comm"] = end_t
        self._batch_submit_by_gid.pop(gid, None)
        self._completed.add(gid)
        ctx.pending_completions.append((gid, waiters, end_t, self.stream))
        return True, None


    
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

    def _bwd(self, t, ctx):
        return True

class all_gather_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-all_gather'

    def _step(self, t, ctx):
        return True

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


class all2all_fwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)

    def _bwd(self, t, ctx):
        return True


class all2all_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)

    def _step(self, t, ctx):
        return True

class send(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==0 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

    def _step(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="fwd")

    def _bwd(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="bwd")

class recv(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==1 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

    def _step(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="fwd")

    def _bwd(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="bwd")

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


class async_send(LeafModel):
    def __init__(self, id, fwd_cost=0, call_stk='', global_rank=None, stream="comm"):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.fwd_cost = fwd_cost
        self.global_rank = global_rank
        self.stream = stream
        self._completed = set()
        self._entry_by_gid = {}

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_send {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        start_t = t["comp"]
        eid = ctx.post_async_send_entry(
            gid=gid,
            rank=self.global_rank,
            post_t=start_t,
            cost=self.fwd_cost,
            stream=self.stream,
            mode="async_send",
            call_stk=self.call_stk,
            log_id=f"{phase}:{self.id}",
        )
        self._entry_by_gid[gid] = eid
        self._completed.add(gid)
        return False, ("yield_done", gid)

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)


class async_recv(LeafModel):
    def __init__(self, id, call_stk='', global_rank=None, stream="comm", fwd_cost=0):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.global_rank = global_rank
        self.stream = stream
        self.fwd_cost = fwd_cost
        self._launched = set()
        self._entry_by_gid = {}

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._launched:
            return True, None
        eid = ctx.post_async_recv_entry(
            gid=gid,
            rank=self.global_rank,
            post_t=t["comp"],
            cost=self.fwd_cost,
            stream=self.stream,
            mode="async_recv",
            call_stk=self.call_stk,
            log_id=f"{phase}:{self.id}",
        )
        self._entry_by_gid[gid] = eid
        self._launched.add(gid)
        return False, ("yield_done", gid)

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)


class async_recv_prev(async_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_send_next(async_send):
    def __init__(self, id, rank, fwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{rank}-{next_rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, fwd_cost=fwd_cost, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_recv_next(async_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_send_prev(async_send):
    def __init__(self, id, rank, fwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{rank}-{prev_rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, fwd_cost=fwd_cost, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True

class async_wait_recv(LeafModel):
    def __init__(self, id, call_stk='', global_rank=None, stream="comm", fwd_cost=0):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.global_rank = global_rank
        self.stream = stream
        self.fwd_cost = fwd_cost
        self._completed = set()

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_wait_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        ready_t = ctx.get_async_ready_t(gid)
        if ready_t is None:
            if not ctx.has_async_posted_send(gid) or not ctx.has_async_posted_recv(gid):
                return False, ("async_wait", gid)
            ready_t = ctx.ensure_async_ready(gid)
            if ready_t is None:
                return False, ("async_wait", gid)
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def _event_call_stk(self):
        return self.call_stk.replace("async_wait_recv", "async_recv")

    def _emit_async_pair_logs(self, ctx, gid, ready_t, op):
        return

    def step(self, t, ctx):
        gid = ("fwd", self.id)
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="async_recv",
                call_stk=self._event_call_stk(),
                log_id=f"fwd:{self.id}",
            )
            return False, ("yield_keep", gid)
        ok, blk = self._step(t, ctx, phase="fwd")
        if ok:
            ready = ctx.get_async_ready_t(gid) or t[self.stream]
            self._emit_async_pair_logs(ctx, gid, ready, "fwd")
            return True, None
        return False, blk

    def bwd(self, t, ctx):
        gid = ("bwd", self.id)
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="async_recv",
                call_stk=self._event_call_stk(),
                log_id=f"bwd:{self.id}",
            )
            return False, ("yield_keep", gid)
        ok, blk = self._bwd(t, ctx)
        if ok:
            ready = ctx.get_async_ready_t(gid) or t[self.stream]
            self._emit_async_pair_logs(ctx, gid, ready, "bwd")
            return True, None
        return False, blk


class async_wait_recv_prev(async_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_wait_recv_next(async_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class sync_send(async_send):
    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"sync_send {self.id}: global_rank is None")
        gid = (phase, self.id)
        if not ctx.has_async_posted_send(gid):
            eid = ctx.post_async_send_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="sync_send",
                call_stk=self.call_stk,
                log_id=f"{phase}:{self.id}",
            )
            self._entry_by_gid[gid] = eid
        ready_t = ctx.ensure_async_ready(gid)
        if ready_t is None:
            return False, ("comm_entry", self._entry_by_gid[gid])
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")


class sync_send_next(async_send_next):
    def __init__(self, *args, **kwargs):
        kwargs["stream"] = "comm"
        super().__init__(*args, **kwargs)


class sync_send_prev(async_send_prev):
    def __init__(self, *args, **kwargs):
        kwargs["stream"] = "comm"
        super().__init__(*args, **kwargs)


class sync_wait_recv(async_wait_recv):
    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"sync_wait_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="sync_recv",
                call_stk=self._event_call_stk(),
                log_id=f"{phase}:{self.id}",
            )
        ready_t = ctx.ensure_async_ready(gid)
        if ready_t is None:
            return False, ("comm_entry", ctx.get_async_recv_eid(gid))
        t[self.stream] = max(t[self.stream], ready_t)
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def _event_call_stk(self):
        return self.call_stk.replace("sync_wait_recv", "sync_recv")


class sync_wait_recv_prev(sync_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        kwargs["stream"] = "comm"
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class sync_wait_recv_next(sync_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        kwargs["stream"] = "comm"
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True
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
        "cp_group_id",
        "pp_group_id",
        "dp_group_id",
        "dp_cp_group_id",
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
