"""Configuration classes for SimuMax """

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import copy
import warnings

from .utils import to_json_string


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
class StrategyConfig(Config):
    """
    Training strategy configuration
    """

    seq_len: Optional[int] = None
    micro_batch_size: Optional[int] = None
    micro_batch_num: Optional[int] = None
    dtype: Optional[int] = None
    # dist strategy
    world_size: Optional[int] = None
    tp_size: int = None
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    moe_dispatcher_policy: str = "all2all"

    enable_sequence_parallel: bool = True
    interleaving_size: int = 1
    zero_state: int = 0

    no_sync: bool = True
    attention_sparse_ratio: float = (
        0.0  # 0.0 means dense attention; 0.5 means compute optimize for causal attention
    )
    enable_dropout: bool = False
    use_fp32_accum_grad: bool = True

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

    # network strategy
    # TODO: auto choose network strategy
    tp_net: Optional[str] = None
    pp_net: Optional[str] = None
    dp_net: Optional[str] = None
    ep_net: Optional[str] = None
    etp_net: Optional[str] = None

    mem_factor: float = 0.94

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
        assert self.recompute_granularity in [
            "full_block",
            "attn_only",
            "mlp_only",
            "sdp_only",
        ], "recompute_granularity must be in ['full_block', 'attn_only', 'sdp_only', 'mlp_only']"
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


@dataclass
class CompOpConfig:
    tflops: int
    efficient_factor: int


@dataclass
class AcceleratorConfig:
    backend: str
    mem_gbs: int
    bandwidth: BandwidthConfig
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

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        config_dict = copy.deepcopy(config_dict)
        accelerator = config_dict.pop("accelerator")
        sys_name = config_dict.pop("sys_name")
        num_per_node = config_dict.pop("num_per_node")
        networks = config_dict.pop("networks")
        accelerator = AcceleratorConfig(
            backend=accelerator["backend"],
            mem_gbs=accelerator["mem_gbs"],
            bandwidth=BandwidthConfig(**accelerator["bandwidth"]),
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
        return cls(
            sys_name=sys_name,
            num_per_node=num_per_node,
            accelerator=accelerator,
            networks=networks,
        )

    def compute_op_time(self, op_name: str, flops: int):
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
        return time

    def compute_mem_access_time(self, mem_bytes: int):
        """
        compute memory access time,
        return time in ms
        """
        time = (
            mem_bytes
            / (
                self.accelerator.bandwidth.gbps
                * 1024**3
                * self.accelerator.bandwidth.efficient_factor
            )
            * 1e3
        )
        time += self.accelerator.bandwidth.latency_us / 1e3
        return time

    def compute_net_op_time(self, op_name: str, size: int, comm_num: int, net=""):
        """
        compute network operation time,
        return time in ms
        """
        # Using ring alg for now
        assert op_name in kNetOp, f"{op_name} not exist on {self.kNetOp}"
        net_data = self.networks.get(net, None)
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}"
        op = net_data.op.get(op_name, None)  # 0: scale 1: offset 2: efficient_factor
        assert op is not None, f"{op_name} not exist on {net_data}"
        scale, offset, eff_factor = op.scale, op.offset, op.efficient_factor
        if eff_factor is None:
            eff_factor = net_data.bandwidth.efficient_factor

        actual_size = size * scale
        chunk_size = actual_size / comm_num
        actual_size += chunk_size * offset

        latency = net_data.bandwidth.latency_us
        if op in ["all_reduce", "all_gather", "reduce_scatter", "all2all"]:
            latency = net_data.bandwidth.latency_us * (comm_num + offset) * scale
        time = (
            actual_size / (net_data.bandwidth.gbps * 1024**3 * eff_factor) * 1e3
            + latency / 1e3
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


@dataclass
class ModelConfig(Config):
    """Transformer model(decode-only) configuration"""

    hidden_size: int
    head_num: int
    kv_head_num: int
    head_size: int
    intermediate_size: int
    layer_num: int
    vocab_size: int
    use_swiglu: bool
    expert_num: int = 1
    topk: int = None

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
        if self.topk is not None and self.topk > 1:
            factor += self.topk - 1
        mlp_matmul = 3 * 2 * self.layer_num * self.mlp_elements * factor
        res = attn_matmul + mlp_matmul
        if with_attn:
            attn_sdp = 3 * 2 * self.layer_num * (2 * context_seq_len * self.hidden_size)
            res += attn_sdp
        return res

    @property
    def mlp_elements(self):
        mlp_weight_factor = 3 if self.use_swiglu else 2
        mlp_elements = mlp_weight_factor * self.hidden_size * self.intermediate_size
        return mlp_elements

    @property
    def base_proj_elements(self):
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
        assert self.head_num * self.head_size == self.hidden_size
