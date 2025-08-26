"""Configuration classes for SimuMax """
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import copy
import math
import warnings

from simumax.core.utils import to_json_string

SIMU_CHECK = int(os.environ.get("SIMU_CHECK", "0"))
SIMU_DEBUG = int(os.environ.get('SIMU_DEBUG', '0'))
if SIMU_CHECK:
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
        # Start with the regular dataclass fields
        output = asdict(self)

        # Use reflection to automatically add all @property attributes
        for attr_name in dir(self):
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property):
                output[attr_name] = getattr(self, attr_name)

        return output

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
    input_norm_recompute:bool = False
    qkv_norm_recompute:bool = False
    qkv_recompute:bool = False
    attn_recompute:bool = False
    out_recompute:bool = False

    @property
    def is_recompute_all(self):
        return (self.input_norm_recompute and
                self.qkv_norm_recompute and
                self.qkv_recompute and
                self.attn_recompute and
                self.out_recompute)

@dataclass
class MLPRecomputeConfig(Config):
    pre_mlp_norm_recompute:bool = False
    linear_recompute:bool = False
    router_recompute:bool = False
    permutation_recompute:bool = False

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
    dtype: Optional[int] = None
    fp8: Optional[bool] = False
    # dist strategy
    world_size: Optional[int] = None
    tp_size: int = None
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    moe_dispatcher_policy: str = "all2all"
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False
    grad_reduce_in_bf16: bool = False

    attn_recompute: bool = False
    mla_rms_recompute: bool = False 
    mlp_recompute: bool  = False
    mlp_rms_recompute: bool = False

    enable_sequence_parallel: bool = True
    interleaving_size: int = 1
    zero_state: int = 0

    no_sync: bool = True
    attention_sparse_ratio: float = (
        0.0  # 0.0 means dense attention; 0.5 means compute optimize for causal attention
    )
    enable_dropout: bool = False
    use_fp32_accum_grad: bool = True
    use_accm_weight:bool = True # TODO(sherry): if True, No need to generate temporary variables of weight

    # recompute
    enable_recompute: bool = False
    skip_ckpt_micro_batch_num: int = 1
    recompute_granularity: Optional[str] = None
    recompute_layer_num: int = 0

    # fused kernel
    use_flash_sdp: bool = True
    use_math_sdp: bool = False
    use_fused_norm: bool = True
    use_fused_swiglu: bool = True
    use_fused_grad_accumulation: bool = True

    # network strategy
    # TODO: auto choose network strategy
    tp_net: Optional[str] = "auto"
    pp_net: Optional[str] = "auto"
    dp_net: Optional[str] = "auto"
    ep_net: Optional[str] = "auto"
    etp_net: Optional[str] = "auto"

    mem_factor: float = 0.94

    recompute_breakpoints = ["LLMBlock", "MLAAttention", "MLP"]
    
    valid_recompute_granularity = [
            "full_block",
            "attn_only",
            "mlp_only",
            "sdp_only",
            "selective_recompute"
        ]
    @property
    def shard_size(self):
        return self.pp_size * self.tp_size

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
        return f'seq{self.seq_len}.mbs{self.micro_batch_size}.mbc{self.micro_batch_num}.gbs{self.global_batch_size} tp{self.tp_size}.ep{self.ep_size}.pp{self.pp_size}.dp{self.dp_size}.etp{self.etp_size}.edp{self.edp_size}, world_size:{self.world_size}'
    
    @property
    def recompute_status(self):
        is_full_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'full_block'
        is_selective_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'selective_recompute' and any([self.attn_recompute, self.mla_rms_recompute, self.mlp_recompute, self.mlp_rms_recompute])
        if not is_full_recompute and not is_selective_recompute:
            return 'No Recompute'
        if is_full_recompute:
            return f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}"
        else:
            return f'{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}, attn={self.attn_recompute}, attn_rms={self.mla_rms_recompute}, mlp={self.mlp_recompute}, mlp_rms={self.mlp_rms_recompute}'
    @property
    def net(self):
        return f"pp_net={self.pp_net}, tp_net={self.tp_net}, dp_net={self.dp_net}, ep_net={self.ep_net}, etp_net={self.etp_net}"
    
    def parse_attention_recompute(self, layer_idx):
        if self.recompute_granularity is None:
            return AttentionRecomputeConfig()
        
        if self.recompute_granularity == "full_block":
            input_norm_recompute = True 
            qkv_norm_recompute = True
            qkv_recompute = True
            attn_recompute = True
            out_recompute = True
        elif self.recompute_granularity == "attn_only":
            input_norm_recompute = False
            qkv_norm_recompute = True # TODO(sherry): check this, theoretically, it should be enabled, but the old version does not
            qkv_recompute = True    # attn only
            attn_recompute = True   # attn only
            out_recompute = True    # attn only
        elif self.recompute_granularity == "sdp_only":
            input_norm_recompute = False
            qkv_norm_recompute = False 
            qkv_recompute = False
            attn_recompute = True   # sdp only
            out_recompute = False
        elif self.recompute_granularity == "selective_recompute":
            # selective_recompute support:
            #     attn_recompute: bool = False
            #     mla_rms_recompute: bool = False 
            #     mlp_recompute: bool  = False
            #     mlp_rms_recompute: bool = False
            input_norm_recompute = self.mla_rms_recompute # normalization before attention
            
            if self.mla_rms_recompute:
                assert self.attn_recompute, "mla_rms_recompute requires attn_recompute"
            qkv_norm_recompute = self.mla_rms_recompute or self.attn_recompute  # qkv norm recompute made with attn recompute
            qkv_recompute = self.mla_rms_recompute or self.attn_recompute 
            attn_recompute = self.attn_recompute
            out_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")
        enable_layer_recompute = (
                layer_idx < self.recompute_layer_num
            )
        return AttentionRecomputeConfig(input_norm_recompute and enable_layer_recompute,
                                        qkv_norm_recompute and enable_layer_recompute,
                                        qkv_recompute and enable_layer_recompute,
                                        attn_recompute and enable_layer_recompute,
                                        out_recompute and enable_layer_recompute)
    
    def parse_mlp_recompute(self, layer_idx):
        if self.recompute_granularity is None:
            return MLPRecomputeConfig()
        
        if self.recompute_granularity == "full_block":
            pre_mlp_norm_recompute = True 
            linear_recompute = True
            router_recompute = True
            permutation_recompute = True
        elif self.recompute_granularity in ["attn_only", "sdp_only"]:
            pre_mlp_norm_recompute = False
            linear_recompute = False
            router_recompute = False
            permutation_recompute = False
        elif self.recompute_granularity == "selective_recompute":
            pre_mlp_norm_recompute = self.mlp_rms_recompute # normalization before mlp, after attention
            if self.mlp_rms_recompute:
                assert self.mlp_recompute, "mlp_rms_recompute requires mlp_recompute"
            linear_recompute = self.mlp_rms_recompute or self.mlp_recompute
            router_recompute = self.mlp_rms_recompute or self.mlp_recompute
            permutation_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")
        enable_layer_recompute = (
                layer_idx < self.recompute_layer_num
            )
        return MLPRecomputeConfig(pre_mlp_norm_recompute = pre_mlp_norm_recompute and enable_layer_recompute,
                                  linear_recompute = linear_recompute and enable_layer_recompute,
                                  router_recompute= router_recompute and enable_layer_recompute,
                                  permutation_recompute = permutation_recompute and enable_layer_recompute)

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
        assert (
            self.world_size % self.shard_size == 0
        ), "world_size must be divisible by pp_size * tp_size"
        assert self.zero_state in [0, 1, 2, 3], "zero_state must be in [0, 1, 2, 3]"
        assert self.recompute_granularity is None or self.recompute_granularity in self.valid_recompute_granularity, f"recompute_granularity must be in [{','.join(self.valid_recompute_granularity)}]"
        assert self.recompute_layer_num >= 0
        assert (
            self.world_size % (self.ep_size * self.etp_size * self.pp_size) == 0
        ), "world_size must be divisible by ep_size * etp_size * pp_size"
        assert (
            self.dp_size % self.ep_size == 0
        ), f"dp_size {self.dp_size} is not divisible by ep_size {self.ep_size}"
        assert (
            self.ep_size == 1 or self.enable_sequence_parallel
        ), "when using ep, sp must be used"
        assert self.moe_dispatcher_policy in [
            "all2all",
            "all2all-seq",
        ], "moe_dispatcher_policy must be in ['all2all', 'all2all-seq']"
        if self.zero_state in [2, 3]:
            assert (
                self.micro_batch_num == 1 or not self.no_sync
            ), "when using zero_state 2 and 3, no_sync must be False"
        if self.interleaving_size == 1:
            warnings.warn(
                "interleaving_size is not supported yet, the configuration will be ignored."
            )
        if self.enable_dropout:
            warnings.warn(
                "enable_dropout is not supported yet, the configuration will be ignored."
            )
        if self.enable_recompute:
            warnings.warn("Recompute is currently in experimental feature.")
        if self.moe_dispatcher_policy == "all2all-seq":
            assert (
                self.etp_size == self.tp_size
            ), "etp_size must be equal to tp_size when using all2all-seq"
        if self.skip_ckpt_micro_batch_num > 0:
            warnings.warn(
                "skip_ckpt_micro_batch_num is not supported yet, the configuration will be ignored."
            )
            self.skip_ckpt_micro_batch_num = 0
        if self.zero_state in [2, 3]:
            warnings.warn(
                "zero_state 2 and 3 are not supported yet, the configuration will be ignored."
            )


@dataclass
class BandwidthConfig:
    gbps: int
    efficient_factor: int
    latency_us: int
    fixed_latency:int = 0


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
    real_comm_bw = {}
    FC8:bool = False
    intra_with_pcie:bool = False

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
    
    def compute_op_accuracy_time(self, op_name:str, flops:int, matmul_input_shapes:str, reture_detail=False):
        """
        compute float point operation time,
        return time in ms

        matmul_input_shapes: list of input shapes, e.g. "[1, 16384, 4096] x [1, 4096, 128256]" 
        """
        op = self.accelerator.op.get(op_name, None)
        if op is None:
            warnings.warn(
                f"{op_name} not exist on {self.accelerator.op}, use default value"
            )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
        
        if "matmul" in op_name and \
            ( op.accurate_efficient_factor is not None ) and \
        (op.accurate_efficient_factor.get(matmul_input_shapes, None) is not None):
            # matmul use accurate efficient factor to get accurate time
            efficient_factor = op.accurate_efficient_factor[matmul_input_shapes] 
            print(f"matmul input shape {matmul_input_shapes} use accurate efficient factor {efficient_factor}")
        else:
            efficient_factor = op.efficient_factor

        time = flops / (op.tflops * 1e12 * efficient_factor) * 1e3
        if reture_detail:
            return dict(op_name=op_name, 
                            tflops=op.tflops, 
                            efficient_factor=efficient_factor,
                            latency_us=self.accelerator.bandwidth.latency_us,
                            compute_only_time = time)
        else:
            return time
    
    def compute_op_accuracy_time2(self, op_name:str, flops:int, shape_desc:str, reture_detail=False):
        """
        compute float point operation time,
        return time in ms

        matmul_input_shapes: list of input shapes, e.g. "[1, 16384, 4096] x [1, 4096, 128256]" 
        """
        op = self.accelerator.op.get(op_name, None)
        if op is None:
            # warnings.warn(
            #     f"{op_name} not exist on {self.accelerator.op.keys()}, use default value"
            # )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
        if ( op.accurate_efficient_factor is not None ) and \
        (op.accurate_efficient_factor.get(shape_desc, None) is not None):
            # marmul use accurate efficient factor to get accurate time
            efficient_factor = op.accurate_efficient_factor[shape_desc] 
            if SIMU_DEBUG:
                print(f"=== \033[32m{op_name} input shape {shape_desc} use accurate compute efficient factor {efficient_factor}\033[0m, flops={flops}")
        else:
            efficient_factor = op.efficient_factor
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

    def compute_op_time(self, op_name: str, flops: int, reture_detail=False):
        """
        compute float point operation time,
        return time in ms
        """
        op = self.accelerator.op.get(op_name, None)
        if op is None:
            warnings.warn(
                f"{op_name} not exist on {self.accelerator.op}, use default value"
            )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
        time = flops / (op.tflops * 1e12 * op.efficient_factor) * 1e3
        if reture_detail:
            return dict(op_name=op_name, 
                            tflops=op.tflops, 
                            efficient_factor=op.efficient_factor,
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
        if reture_detail:
            return dict(gbps=op.gbps, 
                            efficient_factor=op.efficient_factor,
                            latency_us=op.latency_us,
                            io_time = time)
        return time

    def compute_net_op_time(self, op_name: str, size: int, comm_num: int, net="", comm_stage="unkonw"):
        """
        compute network operation time,
        return time in ms
        """
        # Using ring alg for now
        assert op_name in kNetOp, f"{op_name} not exist on {self.kNetOp}"
        net_data = self.networks.get(net, None)

        fixed_latency = net_data.bandwidth.fixed_latency
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}, op_name={op_name}"
        op:NetOpConfig = net_data.op.get(op_name, None)  # 0: scale 1: offset 2: efficient_factor
        assert op is not None, f"{op_name} not exist on {net_data}"
        scale, offset, eff_factor = op.scale, op.offset, op.efficient_factor
        
        if eff_factor is None:
            eff_factor = net_data.bandwidth.efficient_factor
        actual_size = size * scale
        chunk_size = actual_size / comm_num
        actual_size += chunk_size * offset

        if 'pcie' in net and comm_stage == 'dp' and op.dp_fixed_bw and op.dp_fixed_bw.get(str(comm_num), None):
            dp_fixed_bw = op.dp_fixed_bw.get(str(comm_num))
            self.real_comm_bw[op_name + "_dp"] = {"net":net, "bw":f"{dp_fixed_bw} GB/S", "comm_num":comm_num, "latency": None} 
            return actual_size / (dp_fixed_bw * 1024**3)  * 1000
        
        # Bandwidth decision
        bw = net_data.bandwidth.gbps
        if self.FC8 and net == "high_intra_node":
            bw *= (comm_num-1)/7
        latency = net_data.bandwidth.latency_us
        if op.latency_us is not None:
            latency = op.latency_us
        self.real_comm_bw[op_name] = {"net":net, "bw":f"{bw*eff_factor} GB/S", "comm_num":comm_num, "latency": latency} 
        if comm_num == 1:
            return 0
        if op in ["all_reduce", "all_gather", "reduce_scatter", "all2all"]:
            latency = net_data.bandwidth.latency_us * (comm_num + offset) * scale
        time = (
            actual_size / (bw * 1024**3 * eff_factor) * 1e3
            + (latency+fixed_latency) / 1e3
        )
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
    
    def test_gemm_time(b, m, n, k, dtype, grad_accumulation):
        """
        Test the time of gemm
        """
        try:
            import torch
            from transformer_engine.pytorch.cpp_extensions.gemm import (
                general_gemm,
            )
        except ImportError:
            raise ImportError()
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            raise NotImplementedError(f"{dtype} is not supported")

        if grad_accumulation:
            # grad_accumulation is not supported yet
            raise NotImplementedError("grad_accumulation is not supported yet")

        # test the time of gemm
        a = torch.randn(b, m, k, dtype=dtype)
        b = torch.randn(k, n, dtype=dtype)
        c = torch.randn(b, m, n, dtype=dtype)



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
    attention_type: str = None
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    v_head_dim: int = None
    qk_head_dim: int = None
    qk_pos_emb_head_dim: int = None
    q_lora_rank: int = None
    kv_lora_rank: int = None
    dense_layers: int = 0 # number of dense layers in moe model
    moe_pad_expert_input_to_capacity:bool = False
    capacity:int = 1
    group_linear_mode:str = "parallel"
    make_vocab_size_divisible_by = 128 # default is 128 in megatron
    padded_vocab_size = True # When tokinzer is NullTokenizer, pad vocab size to make it divisible by make_vocab_size_divisible_by * tp_size in Megatron

    def __post_init__(self):
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.intermediate_size

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
            assert self.head_num * self.head_size == self.hidden_size
