"""models for language model"""

from copy import deepcopy

from ..base_struct import MetaModule
from ..config import ModelConfig, StrategyConfig, SystemConfig
from .dense_module import Embedding, Attention, LayerNorm, LinearCol, MLP, ParallelCE
from .moe_module import ExpertMLP


class LLMBlock(MetaModule):
    """Single block of LLM"""

    def __init__(
        self,
        layer_idx: int,
        enable_recompute: bool,
        config: ModelConfig,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.config = deepcopy(config)
        self.layer_idx = layer_idx
        self.enable_recompute = enable_recompute
        assert self.strategy.recompute_granularity in [
            "full_block",
            "attn_only",
            "mlp_only",
            "sdp_only"
        ]
        self.recompute_granularity = (
            "full"
            if self.strategy.recompute_granularity == "full_block"
            else "submodule"
        )
        enable_norm_recompute = self.enable_recompute and any(
            x in self.strategy.recompute_granularity for x in ["full_block"]
        )
        self.register_module(
            LayerNorm(
                norm_size=self.config.hidden_size,
                norm_type="rms_norm",
                use_fused_norm=self.strategy.use_fused_norm,
                has_cached_inputs=False,
                enable_recompute=enable_norm_recompute,
                strategy=strategy,
                system=system,
            )
        )
        enable_attn_recompute = self.enable_recompute and any(
            x in self.strategy.recompute_granularity
            for x in ["full_block", "attn_only", "sdp_only"]
        )
        self.register_module(
            Attention(
                config=self.config,
                enable_recompute=enable_attn_recompute,
                strategy=strategy,
                system=system,
            )
        )

        self.register_module(
            LayerNorm(
                norm_size=self.config.hidden_size,
                norm_type="rms_norm",
                use_fused_norm=self.strategy.use_fused_norm,
                has_cached_inputs=False,
                enable_recompute=enable_norm_recompute,
                strategy=strategy,
                system=system,
            )
        )
        enable_mlp_recompute = self.enable_recompute and any(
            x in self.strategy.recompute_granularity for x in ["full_block", "mlp_only"]
        )
        if self.config.expert_num == 1:
            self.register_module(
                MLP(
                    config=self.config,
                    enable_recompute=enable_mlp_recompute,
                    strategy=strategy,
                    system=system,
                )
            )
        else:
            self.register_module(
                ExpertMLP(
                    config=self.config,
                    enable_recompute=enable_mlp_recompute,
                    strategy=strategy,
                    system=system,
                )
            )


class LLMModel(MetaModule):
    """Full model of LLM"""

    def __init__(
        self,
        layer_num: int,
        preprocess=True,
        postprocess=True,
        model_config: ModelConfig = None,
        strategy: StrategyConfig = None,
        system: SystemConfig = None,
    ) -> None:
        super().__init__(strategy, system)
        # self.chunk_idx = chunk_idx
        self.model_config = deepcopy(model_config)
        self.recompute_granularity = "submodule"
        if preprocess:
            self.register_module(
                Embedding(
                    hidden_size=self.model_config.hidden_size,
                    vocab_size=self.model_config.vocab_size,
                    strategy=self.strategy,
                    system=self.system,
                )
            )
        for i in range(layer_num):
            enable_recompute = self.strategy.enable_recompute and (
                i < self.strategy.recompute_layer_num
            )
            self.register_module(
                LLMBlock(
                    layer_idx=i,
                    enable_recompute=enable_recompute,
                    config=self.model_config,
                    strategy=self.strategy,
                    system=self.system,
                )
            )
        if postprocess:
            self.register_module(
                LayerNorm(
                    norm_size=self.model_config.hidden_size,
                    norm_type="rms_norm",
                    use_fused_norm=self.strategy.use_fused_norm,
                    has_cached_inputs=False,
                    enable_recompute=False,
                    strategy=strategy,
                    system=system,
                )
            )
            self.register_module(
                LinearCol(
                    input_size=self.model_config.hidden_size,
                    output_size=self.model_config.vocab_size,
                    use_bias=False,
                    has_cached_inputs=False,
                    enable_recompute=False,
                    strategy=strategy,
                    system=system,
                )
            )
            self.register_module(ParallelCE(strategy=self.strategy, system=self.system))
