"""Configuration classes for SimuMax """
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from collections import OrderedDict
import json
import copy
import math
import warnings
import re

from simumax.core.utils import to_json_string

capture_graph_only = False
ENABLE_SIMU_GRAPH = int(os.environ.get("ENABLE_SIMU_GRAPH", "0"))
SIMU_CHECK = int(os.environ.get("SIMU_CHECK", "0"))
SIMU_DEBUG = int(os.environ.get('SIMU_DEBUG', '0'))
SIMU_TMP_PATH_OVERRIDE = os.environ.get("SIMUMAX_TMP_PATH", "").strip()
if SIMU_TMP_PATH_OVERRIDE:
    TMP_PATH = SIMU_TMP_PATH_OVERRIDE
elif SIMU_CHECK:
    TMP_PATH = "tmp_check"
else:
    TMP_PATH = "tmp" + time.strftime("_%Y%m%d_%H%M%S", time.localtime())

kNetOp = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "p2p",
    "all2all",
)


def set_capture_graph_only(value: bool):
    global capture_graph_only
    capture_graph_only = value

def get_capture_graph_only():
    return capture_graph_only

class ParameterExtractor:
    def __init__(self, param_patterns: Dict[str, Any]):
        # Parameter patterns and default values.
        self.param_patterns = param_patterns
    
    def extract_parameters(self, input_string):
        """Extract all configured parameters from input string."""
        parameters = {}
        
        for param_name, (pattern, default_value) in self.param_patterns.items():
            match = re.search(pattern, input_string)
            if match:
                parameters[param_name] = int(match.group(1))
            elif default_value is not None:
                parameters[param_name] = default_value
                print(f"Warning: parameter {param_name} not found, use default {default_value}")

        return parameters
    
    def extract_single_parameter(self, input_string, param_name, default_value=None):
        """Extract a single parameter by name."""
        if param_name not in self.param_patterns:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        pattern, default = self.param_patterns[param_name]
        if default_value is not None:
            default = default_value
        
        match = re.search(pattern, input_string)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: parameter {param_name} not found, use default {default}")
            return default
@dataclass
class Config:
    """
    Base class for all configuration
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Automatically includes properties and fields.
        """
        def _normalize_jsonable(value):
            if isinstance(value, dict):
                return {
                    key: _normalize_jsonable(val)
                    for key, val in value.items()
                }
            if isinstance(value, list):
                return [_normalize_jsonable(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_normalize_jsonable(item) for item in value)
            if isinstance(value, set):
                return [_normalize_jsonable(item) for item in sorted(value)]
            return value

        # Start with the regular dataclass fields
        output = asdict(self)

        # Use reflection to automatically add all @property attributes
        for attr_name in dir(self):
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property):
                output[attr_name] = _normalize_jsonable(getattr(self, attr_name))

        return _normalize_jsonable(output)

    def sanity_check(self) -> None:
        # Implement basic sanity checks here
        pass

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return to_json_string(self.to_dict())

    def __str__(self):
        return self.to_json_string()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        """
        Initializes an instance from a dictionary.
        It handles nested dictionaries recursively.
        """
        return cls(**config_dict)

    @staticmethod
    def read_json_file(json_file: str) -> Dict[str, Any]:
        """Reads a JSON file and returns a dictionary."""
        with open(json_file, "r", encoding="utf-8") as reader:
            return json.load(reader)

    @classmethod
    def init_from_config_file(cls, config_file: str):
        """Initializes an instance from a JSON config file."""
        config_dict = cls.read_json_file(config_file)
        return cls.init_from_dict(config_dict)

@dataclass
class AttentionRecomputeConfig(Config):
    # input_norm_recompute:bool = False
    # qkv_norm_recompute:bool = False
    # qkv_recompute:bool = False
    # attn_recompute:bool = False
    # out_recompute:bool = False

    input_layernorm_recompute:bool = False

    q_down_recompute:bool = False
    kv_down_recompute:bool = False
    q_up_recompute:bool = False
    kv_up_recompute:bool = False

    q_layernorm_recompute:bool = False
    kv_layernorm_recompute:bool = False

    rope_recompute:bool = False
    core_attn_recompute:bool = False

    out_recompute:bool = False

    megatron_layernorm: bool = False
    megatron_mla_up_proj: bool = False

    def set_all_status(self, status:bool):
        self.input_layernorm_recompute = status
        self.q_down_recompute = status
        self.kv_down_recompute = status
        self.q_up_recompute = status
        self.kv_up_recompute = status
        self.q_layernorm_recompute = status
        self.kv_layernorm_recompute = status
        self.rope_recompute = status
        self.core_attn_recompute = status
        self.out_recompute = status

    @property
    def is_recompute_all(self):
        return all(self.__dict__.values())

@dataclass
class MLPRecomputeConfig(Config):
    pre_mlp_norm_recompute:bool = False
    shared_linear_recompute:bool = False
    linear_recompute:bool = False # Noraml MLP and grouped MLP
    router_recompute:bool = False
    permutation_recompute:bool = False

    megatron_layernorm: bool = False
    megatron_mlp: bool = False
    megatron_moe: bool = False
    megatron_moe_act: bool = False
    
    @property
    def is_recompute_all(self):
        return (self.pre_mlp_norm_recompute and 
                self.linear_recompute and 
                self.router_recompute and 
                self.permutation_recompute)
@dataclass
class StrategyConfig(Config):
    """
    Training strategy configuration
    """

    seq_len: Optional[int] = None
    micro_batch_size: Optional[int] = None
    micro_batch_num: Optional[int] = None
    dtype: Optional[int] = 'bf16'
    fp8: Optional[bool] = False
    
    # dist strategy
    world_size: Optional[int] = 8
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    cp_comm_type: str = "a2a"
    cp_a2a_mode: str = "async_cp"
    order_of_paralielism: str = "tp-cp-ep-dp-pp"
    moe_dispatcher_policy: str = "all2all"
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # memory optimization
    grad_reduce_in_bf16: bool = False
    cache_groupgemm_col_fp8_inputs: Optional[bool] = False
    offload_groupgemm_col_inputs: Optional[bool] = False

    attn_recompute: bool = False
    mla_rms_recompute: bool = False 
    mlp_recompute: bool  = False
    mlp_rms_recompute: bool = False

    enable_sequence_parallel: bool = True
    interleaving_size: int = 1
    microbatch_group_size_per_vp_stage: Optional[int] = None
    pp_comm_async: bool = True
    enable_straggler_model: bool = True
    zero_state: int = 1

    attention_sparse_ratio: float = (
        0.0  # 0.0 means dense attention; 0.5 means compute optimize for causal attention
    )
    enable_dropout: bool = False
    use_fp32_accum_grad: bool = True
    use_accm_weight:bool = True # TODO(sherry): if True, No need to generate temporary variables of weight

    # recompute
    enable_recompute: bool = True
    recompute_granularity: Optional[str] = None
    recompute_layer_num: int = 0
    recompute_variance: bool = False
    megatron_recompute: bool = False
    megatron_recompute_modules: Optional[List[str]] = None

    # fused kernel
    use_flash_sdp: bool = True
    use_math_sdp: bool = False
    use_fused_norm: bool = True
    use_fused_swiglu: bool = True
    use_fused_grad_accumulation: bool = True
    cross_entropy_loss_fusion: bool = False
    overlap_grad_reduce: bool = True

    # TE release audit:
    # - regular linear starts using get_dummy_wgrad in release_v2.3
    # - CP A2A attention starts saving pre-PostA2A O in release_v2.8
    # - grouped linear starts using get_dummy_wgrad in release_v2.10
    te_version: Optional[str] = None
    te_dummy_wgrad_min_version: str = "2.3.0"
    te_cp_a2a_save_pre_posta2a_min_version: str = "2.8.0"
    te_grouped_linear_dummy_wgrad_min_version: str = "2.10.0"

    # network strategy
    # TODO: auto choose network strategy
    tp_net: Optional[str] = "auto"
    cp_net: Optional[str] = "auto"
    pp_net: Optional[str] = "auto"
    dp_net: Optional[str] = "auto"
    ep_net: Optional[str] = "auto"
    etp_net: Optional[str] = "auto"
    edp_net: Optional[str] = "auto"

    # Megatron related
    dispatch_probs: bool = False # The new version of Megatron combines probs in Silu after Groupgemm1 in ExpertMLP

    mem_factor: float = 0.94
    
    valid_recompute_granularity = [
            "full_block",
            "attn_only",
            "mlp_only",
            "sdp_only",
            "selective_recompute"
        ]
    valid_megatron_recompute_modules = [
        "core_attn",
        "layernorm",
        "mla_up_proj",
        "moe_act",
        "mlp",
        "moe",
    ]
    valid_cp_a2a_modes = [
        "async_cp",
        "sync_cp",
    ]
    
    @classmethod
    def init_from_format_strings(cls, strs):
        """
        Docstring for init_from_format_strings
        parse format like:
        find
        seq{self.seq_len}.mbs{self.micro_batch_size}.mbc{self.micro_batch_num}.gbs{self.global_batch_size} tp{self.tp_size}.ep{self.ep_size}.pp{self.pp_size}.dp{self.dp_size}.etp{self.etp_size}.edp{self.edp_size}, world_size:{self.world_size}

        :param cls: Description
        :param strs: Description
        :return: Description
        :rtype: Any
        """
        param_patterns = {
            'seq_len': (r'seq(\d+)', 4096),
            'micro_batch_size': (r'mbs(\d+)', 1),
            'micro_batch_num': (r'mbc(\d+)', 1),
            'global_batch_size': (r'gbs(\d+)', 8),
            'tp_size': (r'tp(\d+)', 1),
            'cp_size': (r'cp(\d+)', 1),
            'ep_size': (r'ep(\d+)', 1),
            'pp_size': (r'pp(\d+)', 1),
            'world_size': (r'world_size:(\d+)', 8)
        }
        extractor = ParameterExtractor(param_patterns=param_patterns)
        params = extractor.extract_parameters(strs)
        global_batch_size = params.pop('global_batch_size')
        strategty = StrategyConfig(**params)
        strategty.reset_global_batch_size(global_batch_size)
        return strategty
        
    @property
    def shard_size(self):
        return self.pp_size * self.tp_size * self.cp_size

    @property
    def dp_size(self):
        assert self.world_size % self.shard_size == 0
        return self.world_size // self.shard_size

    @property
    def global_batch_size(self):
        global_batch_size = self.micro_batch_size * self.micro_batch_num * self.dp_size
        return global_batch_size

    @property
    def edp_size(self):
        return self.world_size // (self.ep_size * self.etp_size * self.pp_size)
    
    @property
    def parallelism(self):
        sp_tag = f'sp{self.tp_size}.' if self.enable_sequence_parallel else ''
        return f'seq{self.seq_len}.mbs{self.micro_batch_size}.mbc{self.micro_batch_num}.gbs{self.global_batch_size} tp{self.tp_size}.{sp_tag}cp{self.cp_size}.ep{self.ep_size}.pp{self.pp_size}.dp{self.dp_size}.etp{self.etp_size}.edp{self.edp_size}, world_size:{self.world_size}'

    @property
    def megatron_recompute_module_set(self):
        return set(self.megatron_recompute_modules or [])

    @staticmethod
    def _version_tuple(version: Optional[str]):
        if not version:
            return None
        parts = re.findall(r"\d+", str(version))
        if not parts:
            return None
        nums = [int(part) for part in parts[:3]]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)

    @property
    def te_dummy_wgrad_memory_enabled(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_dummy_wgrad_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def te_grouped_linear_dummy_wgrad_memory_enabled(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_grouped_linear_dummy_wgrad_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def te_cp_a2a_saves_pre_posta2a_output(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_cp_a2a_save_pre_posta2a_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def use_variance_tail_model(self):
        return self.recompute_variance or (
            self.is_megatron_selective_recompute
            and bool(self.megatron_recompute_module_set & {"layernorm", "mla_up_proj", "moe_act"})
        )

    @property
    def is_megatron_selective_recompute(self):
        return (
            self.enable_recompute
            and self.recompute_layer_num > 0
            and self.recompute_granularity == "selective_recompute"
            and self.megatron_recompute
            and bool(self.megatron_recompute_module_set)
        )
    
    @property
    def is_recompute(self):
        is_full_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'full_block'
        is_partial_recompute = self.recompute_layer_num > 0 and self.recompute_granularity in ['attn_only', 'mlp_only', 'sdp_only']
        is_selective_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'selective_recompute' and any([self.attn_recompute, self.mla_rms_recompute, self.mlp_recompute, self.mlp_rms_recompute])
        return self.enable_recompute and (
            is_full_recompute
            or is_partial_recompute
            or is_selective_recompute
            or self.is_megatron_selective_recompute
        )
    
    @property
    def recompute_status(self):
        is_full_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'full_block'
        is_partial_recompute = self.recompute_layer_num > 0 and self.recompute_granularity in ['attn_only', 'mlp_only', 'sdp_only']
        is_selective_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'selective_recompute' and any([self.attn_recompute, self.mla_rms_recompute, self.mlp_recompute, self.mlp_rms_recompute])
        if not self.is_recompute:
            return 'No Recompute'
        if is_full_recompute:
            return f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}"
        elif is_partial_recompute:
            return f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}"
        elif self.is_megatron_selective_recompute:
            modules = ",".join(sorted(self.megatron_recompute_module_set))
            return (
                f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}, "
                f"megatron_recompute=True, modules=[{modules}]"
            )
        elif is_selective_recompute:
            return f'{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}, attn={self.attn_recompute}, attn_rms={self.mla_rms_recompute}, mlp={self.mlp_recompute}, mlp_rms={self.mlp_rms_recompute}, recompute_variance={self.recompute_variance}'
        else:
            return 'Unknown Recompute Status'
    @property
    def net(self):
        return f"pp_net={self.pp_net}, tp_net={self.tp_net}, cp_net={self.cp_net}, dp_net={self.dp_net}, ep_net={self.ep_net}, etp_net={self.etp_net}"
    
    def parse_attention_recompute(self, layer_idx):
        if self.recompute_granularity is None or layer_idx >= self.recompute_layer_num:
            return AttentionRecomputeConfig()
        conf = AttentionRecomputeConfig()
        if self.is_megatron_selective_recompute:
            modules = self.megatron_recompute_module_set
            conf.megatron_layernorm = "layernorm" in modules
            conf.megatron_mla_up_proj = "mla_up_proj" in modules
            conf.input_layernorm_recompute = conf.megatron_layernorm
            conf.q_down_recompute = conf.megatron_layernorm
            conf.kv_down_recompute = conf.megatron_layernorm
            conf.q_up_recompute = conf.megatron_mla_up_proj
            conf.kv_up_recompute = conf.megatron_mla_up_proj
            conf.q_layernorm_recompute = conf.megatron_mla_up_proj
            conf.kv_layernorm_recompute = conf.megatron_mla_up_proj
            conf.rope_recompute = conf.megatron_mla_up_proj
            conf.core_attn_recompute = conf.megatron_mla_up_proj
            return conf
        if self.recompute_granularity == "full_block":
            conf.set_all_status(True)
        elif self.recompute_granularity == "attn_only":
            conf.q_down_recompute = True
            conf.kv_down_recompute = True
            conf.q_up_recompute = True
            conf.kv_up_recompute = True
            conf.q_layernorm_recompute = True
            conf.kv_layernorm_recompute = True
            conf.rope_recompute = True
            conf.core_attn_recompute = True
            conf.out_recompute = True
        elif self.recompute_granularity == "sdp_only":
            conf.core_attn_recompute = True
        elif self.recompute_granularity == "mlp_only":
            pass

        elif self.recompute_granularity == "selective_recompute":
            if self.mla_rms_recompute:
                assert self.attn_recompute, "mla_rms_recompute requires attn_recompute"
            conf.input_layernorm_recompute =  self.mla_rms_recompute
            conf.q_down_recompute = self.mla_rms_recompute
            conf.kv_down_recompute = self.mla_rms_recompute
            conf.q_up_recompute = self.attn_recompute 
            conf.kv_up_recompute = self.attn_recompute 
            conf.q_layernorm_recompute = self.attn_recompute 
            conf.kv_layernorm_recompute = self.attn_recompute 
            conf.rope_recompute = self.attn_recompute
            conf.core_attn_recompute = self.attn_recompute 
            conf.out_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")

        return conf
    
    def parse_mlp_recompute(self, layer_idx):
        if self.recompute_granularity is None or layer_idx >= self.recompute_layer_num:
            return MLPRecomputeConfig()
        if self.is_megatron_selective_recompute:
            modules = self.megatron_recompute_module_set
            megatron_moe = "moe" in modules
            megatron_moe_act = "moe_act" in modules and not megatron_moe
            megatron_mlp = "mlp" in modules
            megatron_layernorm = "layernorm" in modules
            return MLPRecomputeConfig(
                pre_mlp_norm_recompute=megatron_layernorm,
                shared_linear_recompute=False,
                linear_recompute=False,
                router_recompute=False,
                permutation_recompute=False,
                megatron_layernorm=megatron_layernorm,
                megatron_mlp=megatron_mlp,
                megatron_moe=megatron_moe,
                megatron_moe_act=megatron_moe_act,
            )
        
        if self.recompute_granularity == "full_block":
            pre_mlp_norm_recompute = True 
            linear_recompute = True
            shared_linear_recompute = True
            router_recompute = True
            permutation_recompute = True
        elif self.recompute_granularity in ["attn_only", "sdp_only"]:
            pre_mlp_norm_recompute = False
            shared_linear_recompute = False
            linear_recompute = False
            router_recompute = False
            permutation_recompute = False
        elif self.recompute_granularity == "mlp_only":
            pre_mlp_norm_recompute = True
            shared_linear_recompute = True
            linear_recompute = True
            router_recompute = True
            permutation_recompute = True
        elif self.recompute_granularity == "selective_recompute":
            pre_mlp_norm_recompute = self.mlp_rms_recompute # normalization before mlp, after attention
            if self.mlp_rms_recompute:
                assert self.mlp_recompute, "mlp_rms_recompute requires mlp_recompute"
            shared_linear_recompute = self.mlp_rms_recompute 
            linear_recompute = self.mlp_recompute
            router_recompute = self.mlp_rms_recompute
            permutation_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")
        return MLPRecomputeConfig(pre_mlp_norm_recompute = pre_mlp_norm_recompute,
                                  shared_linear_recompute = shared_linear_recompute,
                                  linear_recompute = linear_recompute,
                                  router_recompute= router_recompute,
                                  permutation_recompute = permutation_recompute)

    def get_mesh_size(self, order="tp-dp-pp"):
        """According to the order to return the mesh size"""
        res = []
        for x in order.split("-"):
            assert x in (
                "tp",
                "dp",
                "pp",
                "ep",
                "etp",
                "edp",
            ), f"order {x} is not supported"
            res.append(getattr(self, f"{x}_size"))
        return res

    def sanity_check(self):
        if self.order_of_paralielism != "tp-cp-ep-dp-pp":
            raise ValueError(f"Invalid order_of_paralielism, we only support tp-cp-ep-dp-pp now, but got {self.order_of_paralielism}")
        assert self.cp_a2a_mode in self.valid_cp_a2a_modes, (
            f"cp_a2a_mode {self.cp_a2a_mode} must be in [{','.join(self.valid_cp_a2a_modes)}]"
        )
        if self.cache_groupgemm_col_fp8_inputs:
            assert self.fp8, "cache_groupgemm_col_fp8_inputs requires fp8"
            
        if self.offload_groupgemm_col_inputs:
            assert self.recompute_granularity != 'full_block', "offload_groupgemm_col_inputs is not allowed when recompute_granularity = 'full_block'"

        assert self.seq_len % self.cp_size == 0, f"seq_len must be divisible by cp_size, but seq_len = {self.seq_len}, cp_size = {self.cp_size}"
        assert (
            self.world_size % self.shard_size == 0
        ), f"world_size must be divisible by pp_size * tp_size * cp_szie, but world_size = {self.world_size}, pp_size = {self.pp_size}, tp_size = {self.tp_size}, cp_size={self.cp_size}"
        assert self.zero_state in [0, 1, 2, 3], "zero_state must be in [0, 1, 2, 3]"
        assert self.recompute_granularity is None or self.recompute_granularity in self.valid_recompute_granularity, f"recompute_granularity {self.recompute_granularity} must be in [{','.join(self.valid_recompute_granularity)}]"
        assert self.recompute_layer_num >= 0
        if not self.megatron_recompute:
            assert not self.megatron_recompute_module_set, (
                "megatron_recompute_modules requires megatron_recompute=True"
            )
        else:
            assert self.enable_recompute, "megatron_recompute requires enable_recompute=True"
            assert self.recompute_granularity == "selective_recompute", (
                "megatron_recompute requires recompute_granularity='selective_recompute'"
            )
            assert self.recompute_layer_num > 0, (
                "megatron_recompute requires recompute_layer_num > 0"
            )
            invalid_modules = self.megatron_recompute_module_set.difference(
                self.valid_megatron_recompute_modules
            )
            assert not invalid_modules, (
                f"invalid megatron_recompute_modules: {sorted(invalid_modules)}"
            )
            assert self.megatron_recompute_module_set, (
                "megatron_recompute requires non-empty megatron_recompute_modules"
            )
            assert "core_attn" not in self.megatron_recompute_module_set, (
                "megatron_recompute core_attn is not supported in SimuMax yet"
            )
            assert not any(
                [
                    self.attn_recompute,
                    self.mla_rms_recompute,
                    self.mlp_recompute,
                    self.mlp_rms_recompute,
                    self.recompute_variance,
                ]
            ), (
                "megatron_recompute is mutually exclusive with legacy selective flags "
                "and recompute_variance"
            )
        assert (
            self.world_size % (self.ep_size * self.etp_size * self.pp_size) == 0
        ), f"world_size must be divisible by ep_size * etp_size * pp_size, but world_size = {self.world_size}, ep_size = {self.ep_size}, etp_size = {self.etp_size}, pp_size = {self.pp_size}"
        assert self.moe_dispatcher_policy in [
            "all2all",
            "all2all-seq",
        ], "moe_dispatcher_policy must be 'all2all' (legacy alias 'all2all-seq' is accepted with warning)"
        if self.moe_dispatcher_policy == "all2all-seq":
            warnings.warn(
                "moe_dispatcher_policy='all2all-seq' is no longer supported. "
                "Falling back to 'all2all'."
            )
            self.moe_dispatcher_policy = "all2all"
        assert self.interleaving_size >= 1, "interleaving_size must be >= 1"
        if self.interleaving_size > 1:
            assert self.pp_size > 1, "interleaving_size > 1 requires pp_size > 1"
            assert self.pp_comm_async or self.pp_size > 2, (
                "When interleaved schedule is used and p2p communication overlap is disabled, "
                "pipeline-model-parallel size should be greater than 2 to avoid having multiple "
                "p2p sends and recvs between same 2 ranks per communication batch"
            )
            if self.microbatch_group_size_per_vp_stage is None:
                self.microbatch_group_size_per_vp_stage = self.pp_size
            assert self.microbatch_group_size_per_vp_stage >= self.pp_size, (
                "microbatch_group_size_per_vp_stage must be >= pp_size "
                f"(got {self.microbatch_group_size_per_vp_stage} < {self.pp_size})"
            )
            warnings.warn(
                "interleaving_size is enabled. VPP-aware timing/simulation paths are active; "
                "validate target configs with smoke/probe cases when introducing new schedules."
            )
        if self.enable_dropout:
            warnings.warn(
                "enable_dropout is not supported yet, the configuration will be ignored."
            )
        if self.enable_recompute:
            warnings.warn("Recompute is currently in experimental feature.")
        if self.zero_state in [2, 3]:
            warnings.warn(
                "zero_state 2 and 3 are not supported yet, the configuration will be ignored."
            )

        if self.recompute_granularity == "full_block":
            self.recompute_variance = False # megatron-LM's full recompute does not support variance
    def reset_global_batch_size(self, global_batch_size):
        assert global_batch_size % (self.dp_size * self.micro_batch_size)==0, f"global_batch_size {global_batch_size} must be divisible by dp_size*miro_batch_size(dp_size={self.dp_size}, micro_batch_size={self.micro_batch_size})"
        self.micro_batch_num = global_batch_size // (self.dp_size * self.micro_batch_size)
        
@dataclass
class BandwidthConfig:
    gbps: int
    efficient_factor: int
    latency_us: int
    fixed_latency: float = 0
    fixed_latency_us_by_comm_num: Dict[str, float] = None


@dataclass
class CompOpConfig:
    tflops: int
    efficient_factor: int
    accurate_efficient_factor:dict = None


@dataclass
class AcceleratorConfig:
    backend: str
    mem_gbs: int
    bandwidth: Dict[str, BandwidthConfig]
    op: Dict[str, CompOpConfig]
    mode: str


@dataclass
class OpConfig:
    scale: float
    offset: float
    eff: float


@dataclass
class NetOpConfig:
    scale: float
    offset: float
    efficient_factor: float = None
    latency_us: float = None
    fixed_latency_us: float = None
    fixed_latency_us_by_comm_num: Dict[str, float] = None
    dp_fixed_bw: float = None


@dataclass
class NetworkConfig:
    processor_usage: float  # for overlap
    bandwidth: BandwidthConfig
    op: Dict[str, OpConfig]


@dataclass
class SystemConfig(Config):
    """Accelerator system configuration"""

    sys_name: str = "null"
    num_per_node: int = 8
    accelerator: AcceleratorConfig = None
    networks: Dict[str, NetworkConfig] = None
    real_comm_bw: dict = field(default_factory=OrderedDict)
    FC8: bool = False
    intra_with_pcie: bool = False
    miss_efficiency: dict = field(default_factory=OrderedDict)
    hit_efficiency: dict = field(default_factory=OrderedDict)

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        config_dict = copy.deepcopy(config_dict)
        accelerator = config_dict.pop("accelerator")
        sys_name = config_dict.pop("sys_name")
        num_per_node = config_dict.pop("num_per_node")
        networks = config_dict.pop("networks")
        intra_with_pcie = networks.pop('intra_with_pcie') if "intra_with_pcie" in networks else False
        accelerator = AcceleratorConfig(
            backend=accelerator["backend"],
            mem_gbs=accelerator["mem_gbs"],
            bandwidth={k: BandwidthConfig(**v) for k, v in accelerator["bandwidth"].items()},
            op={k: CompOpConfig(**v) for k, v in accelerator["op"].items()},
            mode=accelerator["mode"],
        )
        networks = {
            net_name: NetworkConfig(
                processor_usage=network["processor_usage"],
                bandwidth=BandwidthConfig(**network["bandwidth"]),
                op={k: NetOpConfig(**v) for k, v in network["op"].items()},
            )
            for net_name, network in networks.items()
        }
        FC8 = config_dict.pop("FC8", False)
        return cls(
            sys_name=sys_name,
            num_per_node=num_per_node,
            accelerator=accelerator,
            networks=networks,
            FC8=FC8,
            intra_with_pcie = intra_with_pcie,
        )
    
    def record_miss_efficiency(self, op_name:str, flops:int, shape_desc:str, use_eff):
        if shape_desc:
            if op_name not in self.miss_efficiency:
                self.miss_efficiency[op_name] = {}
            self.miss_efficiency[op_name][f'shape={shape_desc}'] = {
                'flops': flops,
                'use_eff': use_eff
            }
    def record_net_bw(self, op_name:str, net, comm_num, comm_stage:str, base_bw, real_bw, eff_factor, total_time, comm_size, latency):
        if op_name not in self.real_comm_bw:
            self.real_comm_bw[op_name] = {}
        self.real_comm_bw[op_name][comm_stage.lower()] = {"net":net, "base_bw":base_bw, "real_bw":real_bw, "eff_factor":eff_factor, "comm_num":comm_num, "comm_size":comm_size, "total_time":total_time, "latency": latency, "FC8":self.FC8} 

    def record_hit_efficiency(self, op_name:str, flops:int, shape_desc:str, eff):
        if op_name not in self.hit_efficiency:
            self.hit_efficiency[op_name] = {}
        self.hit_efficiency[op_name][shape_desc] = (flops, eff)

    def reset_record_info(self):
        self.miss_efficiency.clear()
        self.hit_efficiency.clear()
        self.real_comm_bw.clear()

    def compute_op_accuracy_time(self, op_name:str, flops:int, shape_desc:str, reture_detail=False):
        """
        compute float point operation time,
        return time in ms

        matmul_input_shapes: list of input shapes, e.g. "[1, 16384, 4096] x [1, 4096, 128256]" 
        """
        if flops == 0:
            if reture_detail:
                return dict(op_name=op_name, 
                                tflops=None, 
                                efficient_factor=None,
                                compute_only_time = 0.0)
            else:
                return 0
            
        op = self.accelerator.op.get(op_name, None)
        if op is None:
            warnings.warn(
                f"{op_name} not exist on {self.accelerator.op.keys()}, use default value"
            )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
            self.record_miss_efficiency(op_name, flops, shape_desc, None)

        if ( op.accurate_efficient_factor is not None ) and \
        (op.accurate_efficient_factor.get(shape_desc, None) is not None):
            # marmul use accurate efficient factor to get accurate time
            efficient_factor = op.accurate_efficient_factor[shape_desc]
            self.record_hit_efficiency(op_name, flops, shape_desc, efficient_factor) 
            if SIMU_DEBUG:
                print(f"=== \033[32m{op_name} input shape {shape_desc} use accurate compute efficient factor {efficient_factor}\033[0m, flops={flops}")
        else:
            efficient_factor = op.efficient_factor
            self.record_miss_efficiency(op_name, flops, shape_desc, efficient_factor)

            if SIMU_DEBUG:
                print(f"{op_name} input shape {shape_desc} use default compute efficient factor {efficient_factor}, flops={flops}")

        time = flops / (op.tflops * 1e12 * efficient_factor) * 1e3
        if reture_detail:
            return dict(op_name=op_name, 
                            tflops=op.tflops, 
                            efficient_factor=efficient_factor,
                            compute_only_time = time)
        else:
            return time

    def compute_mem_access_time(self, op_name, mem_bytes: int, reture_detail=False):
        """
        compute memory access time,
        return time in ms
        """
        
        op = self.accelerator.bandwidth.get(op_name, None)
        if op is None:
            op = self.accelerator.bandwidth.get("default", None)
        else:
            if op_name != "default" and SIMU_DEBUG:
                print(f'{op_name} use accurate memory bw efficiency {op.efficient_factor}')
        
        time = (
            mem_bytes
            / (
                op.gbps
                * 1024**3
                * op.efficient_factor
            )
            * 1e3
        )
        time += op.latency_us / 1e3
        if mem_bytes == 0:
            time = 0
        if reture_detail:
            return dict(gbps=op.gbps, 
                            efficient_factor=op.efficient_factor,
                            latency_us=op.latency_us,
                            io_time = time)
        return time

    @staticmethod
    def _lookup_comm_num_value(values: Dict[str, Any], comm_num: int, default=None):
        if not values:
            return default
        for key in (str(comm_num), comm_num):
            if key in values:
                return values[key]
        return default

    def compute_net_op_time(self, op_name: str, size: int, comm_num: int, net="", comm_stage="unkonw", strategy:StrategyConfig=None):
        """
        compute network operation time,
        return time in ms
        """
        # Using ring alg for now
        assert op_name in kNetOp, f"{op_name} not exist on {kNetOp}"
        net_data = self.networks.get(net, None)
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}, op_name={op_name}"
        op:NetOpConfig = net_data.op.get(op_name, None)  # 0: scale 1: offset 2: efficient_factor
        assert op is not None, f"{op_name} not exist on {net_data}"
        scale, offset, eff_factor = op.scale, op.offset, op.efficient_factor
        
        # Calculate the actual communication data based on the scale and offset of the communication operator
        if eff_factor is None:
            eff_factor = net_data.bandwidth.efficient_factor
        actual_size = size * scale
        chunk_size = actual_size / comm_num
        actual_size += chunk_size * offset

        # Specially adapted to the dense-dp-family communication bandwidth of
        # A100 PCIe. `dp_cp` is Megatron's dense optimizer/data-parallel group
        # with context parallel folded in, so it should reuse the same dense-DP
        # bandwidth family here.
        is_dense_dp_stage = comm_stage in {"dp", "dp_cp"}

        if 'pcie' in net and is_dense_dp_stage and op.dp_fixed_bw and op.dp_fixed_bw.get(str(comm_num), None):
            dp_fixed_bw = op.dp_fixed_bw.get(str(comm_num))
            self.real_comm_bw[op_name + "_dp"] = {"net":net, "bw":f"{dp_fixed_bw} GB/S", "comm_num":comm_num, "latency": None} 
            return actual_size / (dp_fixed_bw * 1024**3)  * 1000
        
        # Intra Bandwidth decision
        bw = net_data.bandwidth.gbps
        if self.FC8 and net == "high_intra_node": # If the internal bandwidth is FC8 mode, the bandwidth changes according to the number of communications.
            bw *= (comm_num-1)/7

        # Inter Bandwidth decision
        if net == "inter_node":
            # 1. pp
            if op_name == "p2p":
                bw /= self.num_per_node
                
            # 2. ep & a2a cp
            if op_name == "all2all":
                if "ep" in comm_stage.lower():
                    # Only consider the case where ep is an integer multiple of num_per_node
                    # K machines cross ep, the total communication size = (k-1)/k *actual_size, 1 piece of data is sent to the self. 
                    # At the same time, cross-machine a2a will use one network card, so the bw is the bw of the single network card
                    
                    # decision comm_size
                    k = max(1, math.ceil(comm_num / self.num_per_node))
                    actual_size = (k-1)/k * actual_size
                    
                    # decision bw
                    bw /= self.num_per_node # bw of the single network card 
                elif "cp" in comm_stage.lower(): 
                    # Similar to ep all2all: when cp spans multiple nodes, only cross-node
                    # traffic contributes to inter-node transfer and each group is limited by one NIC.
                    k = max(1, math.ceil(comm_num / self.num_per_node))
                    actual_size = (k - 1) / k * actual_size
                    bw /= self.num_per_node
            
            # 3. tp+sp & ag cp & dp
            if op_name in ["all_reduce", "all_gather", "reduce_scatter"]:
                if strategy is not None: 
                    if is_dense_dp_stage:
                        # zero0: all_reduce 
                        # zero1: reduce_scatter & all_gather
                        # num_per_node = 8
                        # TP1, each DP group uses all 8 IBs
                        # TP2, each DP group uses 4 IBs, ...
                        #
                        # Distinguish two semantics:
                        # - `dp_cp`: dense optimizer group with CP folded into
                        #   the group itself, so per-node group multiplicity is
                        #   still driven by TP only.
                        # - `dp`: pure dense DP group. If CP is present, each
                        #   `(tp, cp)` slice owns its own DP group, so the
                        #   inter-node contention factor grows with `tp * cp`.
                        dense_group_multiplicity = strategy.tp_size
                        if comm_stage == "dp":
                            dense_group_multiplicity *= strategy.cp_size
                        bw /= min(self.num_per_node, dense_group_multiplicity)
                    elif comm_stage == "edp":
                        # Same as dp
                        bw /= min(self.num_per_node, strategy.ep_size*strategy.etp_size)
                    

        base_latency = op.latency_us if op.latency_us is not None else net_data.bandwidth.latency_us
        fixed_latency = self._lookup_comm_num_value(
            op.fixed_latency_us_by_comm_num,
            comm_num,
            op.fixed_latency_us,
        )
        if fixed_latency is None:
            fixed_latency = self._lookup_comm_num_value(
                net_data.bandwidth.fixed_latency_us_by_comm_num,
                comm_num,
                net_data.bandwidth.fixed_latency,
            )
        latency = base_latency
        if comm_num == 1:
            return 0
        if self.num_per_node == 8 and op_name in ["all_reduce", "all_gather", "reduce_scatter", "all2all"]:
            latency = base_latency * (comm_num + offset) * scale
        time = (
            actual_size / (bw * 1024**3 * eff_factor) * 1e3
            + (latency+fixed_latency) / 1e3
        )
        if SIMU_DEBUG:
            if net == "high_intra_node" and op_name=="reduce_scatter":
                print(f"op_name={op_name}, comm_num={comm_num}, net={net}, bw={bw*eff_factor} GB/S, latency={latency} us size={size}")
        self.record_net_bw(op_name, net, comm_num, comm_stage, net_data.bandwidth.gbps, bw*eff_factor, eff_factor, time*1e3, actual_size, latency)
        return time

    def compute_end2end_time(self, compute_time, mem_time):
        """
        According to the accelerator mode, return the end2end time.
        Users can plug in other methods here to simulate
        """
        assert self.accelerator.mode in ["only_compute", "roofline"]
        if self.accelerator.mode == "only_compute":
            # when compute time equal zero, backoff to mem_time
            total_time = compute_time
            if total_time == 0:
                total_time = mem_time
        elif self.accelerator.mode == "roofline":
            total_time = max(compute_time, mem_time)
        else:
            raise NotImplementedError(f"{self.accelerator.mode} is not supported")

        return total_time

    def sanity_check(self):
        pass


@dataclass
class ModelConfig(Config):
    """Transformer model(decode-only) configuration"""
    hidden_size: int
    head_num: int
    kv_head_num: int
    model_type:str = None
    model_name:str = None
    head_size: int = None
    intermediate_size: int = None
    layer_num: int = None
    vocab_size: int = None
    orig_vocab_size: int = None
    use_swiglu: bool = None
    expert_num: int = 1
    topk: int = None
    attention_type: str = 'mha'
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    v_head_dim: int = None
    qk_head_dim: int = None
    qk_pos_emb_head_dim: int = None
    q_lora_rank: int = None
    kv_lora_rank: int = None
    dense_layers: int = 0 # number of dense layers in moe model
    moe_pad_expert_input_to_capacity:bool = True
    capacity:int = 1
    group_linear_mode:str = "parallel"
    make_vocab_size_divisible_by = 128 # default is 128 in megatron
    padded_vocab_size = True # When tokinzer is NullTokenizer, pad vocab size to make it divisible by make_vocab_size_divisible_by * tp_size in Megatron
    

    def __post_init__(self):
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.intermediate_size
        if self.model_type is None:
            if self.expert_num > 1:
                self.model_type = 'moe'
            else:
                self.model_type = 'dense'

    @classmethod
    def init_from_config_file(cls, config_file: str):
        """Initializes an instance from a JSON config file."""
        config_dict = cls.read_json_file(config_file)
        if config_dict.get('moe_ffn_hidden_size') is None:
            config_dict['moe_ffn_hidden_size'] = config_dict['intermediate_size']
        return cls.init_from_dict(config_dict)
    
 
    def maybe_pad_vocab_size(self, tp_size, log=False):
        """ref Megatron-LM: Megatron-LM/megatron/training/tokenizer/tokenizer.py:105
        Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""
        if self.padded_vocab_size:
            if self.orig_vocab_size is None:
                self.orig_vocab_size = self.vocab_size
            multiple = self.make_vocab_size_divisible_by * tp_size
            after = int(math.ceil(self.orig_vocab_size / multiple) * multiple)
            if log:
                print(
                    ' > padded vocab (size: {}) with {} dummy tokens '
                    '(new size: {})'.format(self.orig_vocab_size, after - self.orig_vocab_size, after),
                    flush=True,
                )
            self.vocab_size = after
    
    def set_vocab_size(self, vocab_size):
        self.orig_vocab_size = vocab_size 
        self.vocab_size = vocab_size
        
    @property
    def param_numel(self):
        return (
            2 * self.vocab_elements
            + self.layer_elements * self.layer_num
            + self.norm_elements
        )

    @property
    def activated_param_numel(self):
        return (
            2 * self.vocab_elements
            + self.layer_act_elements * self.layer_num
            + self.norm_elements
        )

    def flops_per_token(self, context_seq_len, with_attn=True):
        """compute theoretical FLOPs per token"""
        attn_matmul = (
            3 * 2 * self.layer_num * (self.qkv_proj_elements + self.attn_proj_elements)
        )
        factor = 1
        res = 0
        if self.topk is not None and self.topk > 1:
            factor += self.topk - 1
            attn_router = 3 * 2 * self.layer_num * self.hidden_size * self.expert_num
            res += attn_router
        if self.moe_shared_expert_intermediate_size is not None:
            factor += self.moe_shared_expert_intermediate_size / self.moe_ffn_hidden_size
        mlp_matmul = 3 * 2 * self.layer_num * self.mlp_elements * factor
        res += attn_matmul + mlp_matmul
        if with_attn:
            attn_sdp = 3 * 2 * self.layer_num * (2 * context_seq_len * self.hidden_size)
            if self.attention_type == 'mla':
                attn_sdp = 3 * 2 * self.layer_num * (context_seq_len * (self.qk_head_dim+self.qk_pos_emb_head_dim) * self.head_num+
                                                     context_seq_len * self.v_head_dim * self.head_num)
            res += attn_sdp
            if SIMU_DEBUG:
                print(f"1layer mlp_matmul={mlp_matmul/self.layer_num}; 1layer attn_matmul={attn_matmul/self.layer_num}; 1layer attn_sdp={attn_sdp/self.layer_num}")

            # res += attn_sdp*7/6  #for fa addition bmm; in this case mfu_6nd_with_attn is equal to mean mfu bwtween pp stages
        if SIMU_DEBUG:
            print(f"1layer={res/self.layer_num}; embdedding={3 * 2 * (self.hidden_size * self.vocab_size)}")
        res += 3 * 2 * (self.hidden_size * self.vocab_size)  #for linear in ce
        return res

    @property
    def mlp_elements(self):
        mlp_weight_factor = 3 if self.use_swiglu else 2
        mlp_elements = mlp_weight_factor * self.hidden_size * self.moe_ffn_hidden_size
        return mlp_elements

    @property
    def base_proj_elements(self):
        if self.attention_type=='mla':
            return self.v_head_dim * self.head_num * self.hidden_size
        attn_proj_elements = self.hidden_size * self.hidden_size
        return attn_proj_elements

    @property
    def attn_proj_elements(self):
        return self.base_proj_elements

    @property
    def norm_elements(self):
        # consider rms norm for now
        return self.hidden_size

    @property
    def qkv_proj_elements(self):
        assert self.head_num is not None

        kv_head_num = self.head_num if self.kv_head_num is None else self.kv_head_num
        if self.attention_type=='mla':
            if self.q_lora_rank is None:
                elements = self.hidden_size * self.head_num * (self.qk_head_dim + self.qk_pos_emb_head_dim)
            else:
                elements = self.hidden_size * self.q_lora_rank  #q_down
                elements += self.q_lora_rank * self.head_num * (self.qk_head_dim + self.qk_pos_emb_head_dim) #q_up
            elements += self.hidden_size * (self.kv_lora_rank + self.qk_pos_emb_head_dim)  #kv_down
            elements += self.kv_lora_rank * self.head_num * (self.qk_head_dim + self.v_head_dim) #kv_up
            return elements
        else:
            proj_size = self.head_size * self.head_num + 2 * self.head_size * kv_head_num
            return self.hidden_size * proj_size

    @property
    def vocab_elements(self):
        return self.vocab_size * self.hidden_size

    @property
    def layer_elements(self):
        return (
            self.qkv_proj_elements
            + 2 * self.norm_elements
            + self.attn_proj_elements
            + self.expert_num * self.mlp_elements
        )

    @property
    def layer_act_elements(self):
        factor = 1
        if self.topk is not None and self.topk > 1:
            factor += self.topk - 1
        return (
            self.qkv_proj_elements
            + 2 * self.norm_elements
            + self.attn_proj_elements
            + factor * self.mlp_elements
        )

    def sanity_check(self):
        if not self.v_head_dim: 
            # not used for MLA
            # assert self.head_num * self.head_size == self.hidden_size
            ...
