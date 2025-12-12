"""models for language model"""

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext, LinearBase, RecomputeStatus, RecomputeBreakModule
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig, AttentionRecomputeConfig, MLPRecomputeConfig, SIMU_DEBUG, ENABLE_SIMU_GRAPH
from simumax.core.transformer.dense_module import Embedding, Attention, MLAAttention, LayerNorm, LinearCol, MLP, ParallelCE
from simumax.core.transformer.moe_module import ExpertMLP

@dataclass
class PeakPoint:
    fwd_peak_path:str = None
    fwd_peak_mem:float = 0.

    bwd_peak_path:str = None
    bwd_peak_mem:float = 0.

    recomp_fwd_peak_path:str = None
    recomp_fwd_peak_mem:float = 0.

    recomp_bwd_peak_path:str = None
    recomp_bwd_peak_mem:float = 0.

    forward_activation_mem_cache:float = 0.
    cur_stage:str = "forward"

    def set_peak(self, path, mem, stage:str):
        self.set_stage(stage)
        if self.cur_stage == "forward":
            self.fwd_peak_path = path
            self.fwd_peak_mem = mem
        elif self.cur_stage == "backward":
            self.bwd_peak_path = path
            self.bwd_peak_mem = mem
        elif self.cur_stage == "recompute_forward":
            self.recomp_fwd_peak_path = path
            self.recomp_fwd_peak_mem = mem
        elif self.cur_stage == "recompute_backward":
            self.recomp_bwd_peak_path = path
            self.recomp_bwd_peak_mem = mem

    def update_peak(self, path, mem, stage:str):
        if mem >= self.peak_mem:
            self.set_peak(path, mem, stage)

    def set_stage(self, stage):
        assert stage in ["forward", "backward", "recompute_forward", "recompute_backward"]
        self.cur_stage = stage

    def set_forward_mem_cache(self, mem_cache):
        self.forward_activation_mem_cache = mem_cache

    @property
    def activation_mem_cache(self):
        return self.forward_activation_mem_cache
    
    @property
    def peak_mem(self):
        return max(self.fwd_peak_mem, self.bwd_peak_mem, self.recomp_fwd_peak_mem, self.recomp_bwd_peak_mem)
    
    @property
    def peak_stage(self):
        if self.peak_mem == self.fwd_peak_mem:
            return "forward"
        elif self.peak_mem == self.bwd_peak_mem:
            return "backward"
        elif self.peak_mem == self.recomp_fwd_peak_mem:
            return "recompute_forward"
        else:
            return "recompute_backward"
        
    @property
    def peak_path(self):
        if self.peak_mem == self.fwd_peak_mem:
            return self.fwd_peak_path
        elif self.peak_mem == self.bwd_peak_mem:
            return self.bwd_peak_path
        elif self.peak_mem == self.recomp_fwd_peak_mem:
            return self.recomp_fwd_peak_path
        else:
            return self.recomp_bwd_peak_path
    

    def to_dict(self):
        data_dict =  asdict(self)
        data_dict["activation_mem_cache"] = self.activation_mem_cache
        data_dict["peak_stage"] = self.peak_stage
        data_dict["peak_path"]  = self.peak_path
        data_dict["peak_mem"] = self.peak_mem
        del data_dict["cur_stage"]
        del data_dict["forward_activation_mem_cache"]
        return data_dict
    
    def __repr__(self):
        return f"PeakPoint(path={self.peak_path}, peak_mem={self.peak_mem/1024/1024/1024:.4f} GB, peak_stage={self.peak_stage})"
class LLMBlock(MetaModule):
    """Single block of LLM"""

    def __init__(
        self,
        layer_idx: int,
        enable_recompute: bool,
        attention_recompute:AttentionRecomputeConfig,
        mlp_recompute: MLPRecomputeConfig,
        config: ModelConfig,
        strategy: StrategyConfig,
        system: SystemConfig,
        use_dense: bool=False,
        specific_name='TransformerLayer'
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.config = deepcopy(config)
        self.layer_idx = layer_idx
        self.enable_recompute = enable_recompute
        self.recompute_granularity = (
            "full"
            if self.strategy.recompute_granularity == "full_block"
            else "submodule"
        )
        # enable_norm_recompute = self.enable_recompute and any(
        #     x in self.strategy.recompute_granularity for x in ["full_block"]
        # )
        self.layernorm_input = LayerNorm(
                norm_size=self.config.hidden_size,
                norm_type="rms_norm",
                use_fused_norm=self.strategy.use_fused_norm,
                has_cached_inputs=False,
                enable_recompute=attention_recompute.input_layernorm_recompute,
                strategy=strategy,
                system=system,
            )

        enable_attn_recompute = self.enable_recompute and any(
            x in self.strategy.recompute_granularity
            for x in ["full_block", "attn_only", "sdp_only"]
        )  # for old version 
        if getattr(self.config, 'attention_type', None)=='mla':
            self.attention = MLAAttention(
                    layer_idx=layer_idx,
                    config=self.config,
                    enable_recompute=enable_attn_recompute,  # for old version 
                    attention_recompute_conf = attention_recompute,
                    strategy=strategy,
                    system=system,
                    specific_name='SelfAttention'
                )
        else:
            self.attention = Attention(
                    layer_idx=layer_idx,
                    config=self.config,
                    enable_recompute=enable_attn_recompute,  # for old version 
                    attention_recompute_conf = attention_recompute,
                    strategy=strategy,
                    system=system,
                    specific_name='SelfAttention'
                )
        self.pre_mlp_layernorm = LayerNorm(
                norm_size=self.config.hidden_size,
                norm_type="rms_norm",
                use_fused_norm=self.strategy.use_fused_norm,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.pre_mlp_norm_recompute,
                strategy=strategy,
                system=system,
            )

        enable_mlp_recompute = self.enable_recompute and any(
            x in self.strategy.recompute_granularity for x in ["full_block", "mlp_only"]
        )
        if self.config.expert_num == 1 or use_dense:
            self.mlp = MLP(
                    layer_idx=layer_idx,
                    config=self.config,
                    enable_recompute=enable_mlp_recompute,  # for old version 
                    mlp_recompute_conf = mlp_recompute,
                    strategy=strategy,
                    system=system,
                )
        else:
            self.mlp = ExpertMLP(
                    layer_idx=layer_idx,
                    config=self.config,
                    enable_recompute=enable_mlp_recompute,  # for old version 
                    mlp_recompute = mlp_recompute,
                    strategy=strategy,
                    system=system,
                    specific_name='MoELayer'
                )
    def forward(self, input_info:InputOutputInfo, path_debug_context:PathDebugContext):
        hidden_state = self.layernorm_input(input_info, path_debug_context)
        hidden_state = self.attention(hidden_state, path_debug_context)
        hidden_state = self.pre_mlp_layernorm(hidden_state, path_debug_context)
        out = self.mlp(hidden_state, path_debug_context)
        return out
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = f"{call_stk}{self.call_stk}{self.layer_idx}"
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff=com_buff)


class LLMModel(MetaModule):
    """Full model of LLM"""

    def __init__(
        self,
        layer_num: int,
        dense_layers: int=0,
        preprocess=True,
        postprocess=True,
        model_config: ModelConfig = None,
        strategy: StrategyConfig = None,
        system: SystemConfig = None,
        specific_name = 'GPTModel_0'
    ) -> None:
        super().__init__(strategy, system, specific_name)
        # self.chunk_idx = chunk_idx
        self.model_config = deepcopy(model_config)
        self.recompute_granularity = "submodule"
        self.layer_num = layer_num
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.status_ready = False
        if preprocess:
            self.embedding = Embedding(
                    hidden_size=self.model_config.hidden_size,
                    vocab_size=self.model_config.vocab_size,
                    strategy=self.strategy,
                    system=self.system,
                    specific_name='LanguageModelEmbedding_0'
                )
        # self.layers = []
        for i in range(layer_num):
            enable_recompute = self.strategy.is_recompute and (
                i < self.strategy.recompute_layer_num
            ) 
            attention_recompute_config = self.strategy.parse_attention_recompute(i)
            mlp_recompute_config = self.strategy.parse_mlp_recompute(i)
            setattr(self, f'layer_{i}', LLMBlock(
                    layer_idx=i,
                    enable_recompute=enable_recompute,
                    attention_recompute = attention_recompute_config,
                    mlp_recompute = mlp_recompute_config,
                    config=self.model_config,
                    strategy=self.strategy,
                    system=self.system,
                    use_dense=(i < dense_layers),
                )
            )
        if postprocess:
            self.layernorm = LayerNorm(
                    norm_size=self.model_config.hidden_size,
                    norm_type="rms_norm",
                    use_fused_norm=self.strategy.use_fused_norm,
                    has_cached_inputs=False,
                    enable_recompute=False,
                    strategy=strategy,
                    system=system,
                )
            self.linear_out = LinearCol(
                    layer_idx=-1,
                    input_size=self.model_config.hidden_size,
                    output_size=self.model_config.vocab_size,
                    use_bias=False,
                    has_cached_inputs=False,
                    enable_recompute=False,
                    strategy=strategy,
                    system=system,
                    specific_name='ColumnParallelLinear',
                    enable_fp8 = False,
                )
            self.parallel_ce = ParallelCE(strategy=self.strategy, system=self.system, specific_name='_VocabParallelCrossEntropy')
        
    def __post_init__(self):
        super().__post_init__()
        self.set_first_last_recompute_status()
        self.set_leaf_full_name(self.full_name)
        self.status_ready = True

    def set_first_last_recompute_status(self):
        self.pre_enable_recompute = False
        self.p_recom_m:MetaModule = None
        self.all_recompute_nodes:List[MetaModule] = []
        self.all_leaf_nodes:List[MetaModule] = []

        def add_ordered_module_hook(p_module:MetaModule, sub_module:MetaModule):
            cur_m = sub_module

            if cur_m.is_leaf_module:
                cur_m.call_idx = len(self.all_leaf_nodes)
                self.all_leaf_nodes.append(cur_m)

                # set default recompute status
                if cur_m.enable_recompute:
                     cur_m.recompute_status = RecomputeStatus.MIDDLE
                     self.all_recompute_nodes.append(cur_m)
    
                if not self.pre_enable_recompute and cur_m.enable_recompute:
                    cur_m.recompute_status = RecomputeStatus.FIRST
                if self.pre_enable_recompute and not cur_m.enable_recompute:
                    self.p_recom_m.recompute_status = RecomputeStatus.LAST
                if cur_m.enable_recompute:
                    self.p_recom_m = cur_m
                self.pre_enable_recompute = cur_m.enable_recompute
        
        self.register_add_ordered_module_hooks(add_ordered_module_hook)     

    def set_breakpoints(self, leaf_modules:List[MetaModule]):
        for i in range(0, len(leaf_modules)-1):
            cur_m = leaf_modules[i]
            next_m = leaf_modules[i+1]
            if cur_m.is_breakpoints and cur_m.enable_recompute:
                if SIMU_DEBUG:
                    print(f"--------- Set breakpoint at:{cur_m.full_name}")
                cur_m.recompute_status = RecomputeStatus.LAST
                if next_m.enable_recompute:
                    next_m.recompute_status = RecomputeStatus.FIRST
        for i in range(self.layer_num):
            layer = getattr(self, f"layer_{i}")
            cur_m = layer.children_ordered_module[0]
            if cur_m.enable_recompute:
                if SIMU_DEBUG:
                    print(f"--------- Set breakpoint at:{cur_m.full_name}")
                cur_m.is_breakpoints = True
                cur_m.recompute_status = "first"

    def forward(self, input_info: InputOutputInfo, path_debug_context:PathDebugContext):
        if self.preprocess:
            hidden_states = self.embedding(input_info, path_debug_context)
        else:
            hidden_states = input_info
        
        for i in range(self.layer_num):
            layer = getattr(self, f"layer_{i}")
            hidden_states = layer(hidden_states, path_debug_context)

        
        if self.postprocess:
            hidden_states = self.layernorm(hidden_states, path_debug_context)
            hidden_states = self.linear_out(hidden_states, path_debug_context)
            out = self.parallel_ce(hidden_states, path_debug_context)  
        else:
            out = hidden_states
        return out

    def _comp_fwd_activations(self, enable_recompute,  compute_nodes:List[MetaModule], global_cache_mem:int = 0, peak_point:PeakPoint = None, stage:str = "forward"):
        assert stage in ["forward", "recompute_forward"]
        for i, m in enumerate(compute_nodes):
            assert m.is_leaf_module, f"{m.current_full_module_path} is not a leaf module"
            act_info = m.get_act_info()
            cur_peak_mem = global_cache_mem + act_info.fwd_peak_mem_no_cache
            peak_point.update_peak(f"{m.full_name}: {m.current_full_module_path}", cur_peak_mem, stage)
            
            # update global cache size
            if enable_recompute and m.enable_recompute:
                if stage == "recompute_forward" and m.recompute_status != RecomputeStatus.FIRST:
                    act_info.cache_for_bwd_mem = act_info.total_activation_mem_cache
                    global_cache_mem += act_info.cache_for_bwd_mem 
                elif stage == "forward" and m.recompute_status == RecomputeStatus.FIRST:
                    act_info.cache_for_bwd_mem = m.all_input_element_num() if not m.offload_inputs else 0
                    global_cache_mem += act_info.cache_for_bwd_mem 

            else:
                act_info.cache_for_bwd_mem = act_info.total_activation_mem_cache
                global_cache_mem += act_info.cache_for_bwd_mem 

            if stage == "forward" and enable_recompute:
                # if m.recompute_status in [RecomputeStatus.FIRST, RecomputeStatus.LAST, RecomputeStatus.MIDDLE, RecomputeStatus.NO_RECOMPUTE]:
                if m.recompute_status in [RecomputeStatus.FIRST,  RecomputeStatus.LAST]:
                    if SIMU_DEBUG:
                        print(f"Find {m.recompute_status} node: {m.full_name}")
        peak_point.update_peak(f"{m.full_name}: {m.current_full_module_path}", global_cache_mem, stage)
        
        if stage == "forward":
            peak_point.set_forward_mem_cache(global_cache_mem)

        assert peak_point.peak_mem >= global_cache_mem
        return global_cache_mem
    
    def _comp_bwd_only_activations(self,  nodes:List[MetaModule], global_cache_mem:int = 0,  peak_point:PeakPoint = None, stage = "backward"):
        assert stage in ["backward", "recompute_backward"]

        for m in nodes[::-1]:
            assert m.is_leaf_module, f"{m.current_full_module_path} is not a leaf module"
            act_info = m.get_act_info()
            cur_peak_mem = global_cache_mem + act_info.bwd_peak_mem_no_cache
            peak_point.update_peak(f"{m.full_name}: {m.current_full_module_path}", cur_peak_mem, stage)

            global_cache_mem -= act_info.cache_for_bwd_mem 

            act_info.cache_for_bwd_mem = 0  


        return global_cache_mem

    def _comp_bwd_activations(self, enable_recompute, global_cache_mem:int = 0, peak_point:PeakPoint = None):
        leaf_modules = self.get_all_leaf_modules()
        wait_recompute_nodes:List[MetaModule] = []

        i = len(leaf_modules)-1
        prepare_recompute_ready = False
        while i >=0:
            is_last_module = (i == len(leaf_modules) -1)
            m = leaf_modules[i]
            assert m.is_leaf_module, f"{m.current_full_module_path} is not a leaf module"
            if (enable_recompute and m.enable_recompute and # global recompute enabled and module recompute enabled 
                not m.is_recompute_forward_finished and    # module recompute forward not finished
                not prepare_recompute_ready):              # parepare recompute not ready
                wait_recompute_nodes.append(m) # add recompute node to list
                if m.recompute_status == RecomputeStatus.FIRST:
                    prepare_recompute_ready = True # meet first recompute node, start to recompute
                i -= 1
            elif len(wait_recompute_nodes) > 0:
                wait_recompute_nodes = wait_recompute_nodes[::-1]
                global_cache_mem = self._comp_fwd_activations(enable_recompute, wait_recompute_nodes, global_cache_mem, peak_point, stage="recompute_forward")
                global_cache_mem = self._comp_bwd_only_activations(wait_recompute_nodes, global_cache_mem, peak_point, stage="recompute_backward")
                for m_ in wait_recompute_nodes: # set recompute node to ready
                    m_.is_recompute_forward_finished = True
                wait_recompute_nodes = []
                prepare_recompute_ready = False
            else:
                act_info = m.get_act_info()                
                cur_peak_mem = global_cache_mem + (m.all_input_element_num() if is_last_module else act_info.bwd_peak_mem_no_cache)
                # cur_peak_mem = global_cache_mem + m.all_input_element_num() if is_last_module else act_info.bwd_peak_mem_no_cache
                
                peak_point.update_peak(f"{m.full_name}: {m.current_full_module_path}", cur_peak_mem, "backward")
                
                global_cache_mem -= act_info.cache_for_bwd_mem    
                act_info.cache_for_bwd_mem = 0  
                i -= 1

        if len(wait_recompute_nodes) > 0:
            wait_recompute_nodes = wait_recompute_nodes[::-1]
            global_cache_mem = self._comp_fwd_activations(enable_recompute, wait_recompute_nodes, global_cache_mem, peak_point, stage="recompute_forward")
            global_cache_mem = self._comp_bwd_only_activations(wait_recompute_nodes, global_cache_mem, peak_point, stage="recompute_backward")
        

        assert peak_point.peak_mem >= global_cache_mem

        return  global_cache_mem
    
    def compute_activations(self):
        leaf_nodes = self.get_all_leaf_modules()
        self.set_breakpoints(leaf_nodes)
        peak_point = PeakPoint()        
        enable_recompute = self.strategy.enable_recompute
        global_cache_mem = self._comp_fwd_activations(enable_recompute=enable_recompute, 
                                                        compute_nodes=leaf_nodes, 
                                                        global_cache_mem=0, 
                                                        peak_point=peak_point)

        global_cache_mem = self._comp_bwd_activations(enable_recompute=enable_recompute, 
                                                        global_cache_mem=global_cache_mem, 
                                                        peak_point=peak_point)
       
        for i, m in enumerate(leaf_nodes):
            assert m._act_info.cache_for_bwd_mem == 0, f"{leaf_nodes[i].full_name}._act_info.cache_for_bwd_mem should be 0, but is {m._act_info.cache_for_bwd_mem/1024/1024:.2f} MB"

        assert global_cache_mem == 0, f"global_cache_mem should be 0, but is {global_cache_mem/1024/1024:.2f} MB"          

        return peak_point
    
    def get_all_gemm_cost_info(self):
        all_gemm_info = {
            "Module":[],
            "type":[],
            "B": [],
            "M": [],
            "K": [],
            "N": [],
            'layout':[],
            'accumulate':[],
            'out_dtype':[],
            # "flops":[],
            # "IO":[],
            "compute_cost":[],
            "memory_cost": [],
            "cost":[],
            "bound": []
        }
        leaf_modules = self.get_all_leaf_modules()
        stages = ['fwd', 'bwd_grad_act', 'bwd_grad_w']
        for module in leaf_modules:
            assert module._info_ready, f"{module.full_name} is not ready"
            if isinstance(module, LinearBase):
                compute_info = module.get_compute_info()
                cost_info = module.get_cost_info()
                # fwd/bwd_act/bwd_w
                bmnk_info = module.get_gemm_bmnk('all')
                all_gemm_info['B'].extend(bmnk_info['B'])
                all_gemm_info['M'].extend(bmnk_info['M'])
                all_gemm_info['K'].extend(bmnk_info['K'])
                all_gemm_info['N'].extend(bmnk_info['N'])
                all_gemm_info['layout'].extend(bmnk_info['layout'])
                all_gemm_info['accumulate'].extend(bmnk_info['accumulate'])
                all_gemm_info['out_dtype'].extend(bmnk_info['out_dtype'])
                compute_cost = [module.details[stage]['compute_details']['compute_only_time'] for stage in stages]
                memory_cost = [module.details[stage]['io_details']['io_time'] for stage in stages]
                bound = ['IO bound' if m_cost > c_cost else 'compute bound' for m_cost, c_cost in zip(memory_cost, compute_cost)]
                all_gemm_info['compute_cost'].extend(compute_cost)
                all_gemm_info['memory_cost'].extend(memory_cost)
                all_gemm_info['bound'].extend(bound)
                all_gemm_info['cost'].extend(cost_info.get_all_costs())
                all_gemm_info['Module'].extend([module.full_name+'.fwd', module.full_name + '.bwd_act', module.full_name + '.bwd_w'])
                all_gemm_info['type'].extend([module.__class__.__name__]* 3)
        return all_gemm_info

    def analysis_op_info(self, return_details=False):
        assert self.init_ready and self.input_info and self.status_ready, "Please initialize the model first!"
        leaf_modules = self.get_all_leaf_modules()
        op_infos = {
            "op":[],
            "input_shapes":[],
            "output_shapes":[],
            "flops":[],
            "IO":[],
            "cost":[],
            "compute_only_time":[],
            "IO_time":[],
            "bound":[],
        }
        if return_details:
            op_infos['compute_only_details'] = []
            op_infos['IO_details'] = []

        for m in leaf_modules:
            # forward
            output_shape = m.output_info_.shapes if isinstance(m.output_info_, InputOutputInfo) else [m.output_info_.shape]
            op_infos["op"].append(m.__class__.__name__)
            op_infos["input_shapes"].append(m.input_info.shapes + ([m.weight.shape] if hasattr(m, "weight") else []))
            op_infos['output_shapes'].append(output_shape)
            op_infos["flops"].append(m._compute_info.fwd_flops)  
            op_infos["IO"].append(m._compute_info.fwd_accessed_mem)
            op_infos["cost"].append(m._cost_info.fwd_compute_time)
            print("----------------------", m.full_name , m.details['fwd'])
            op_infos['compute_only_time'].append(m.details['fwd']['compute_details']['compute_only_time'])
            op_infos['IO_time'].append(m.details['fwd']['io_details']['io_time'])
            op_infos['bound'].append("IO bound" if op_infos['IO_time'][-1] > op_infos['compute_only_time'][-1] else "Compute bound")

            if return_details:
                op_infos['compute_only_details'].append(m.details['fwd']['compute_details'])
                op_infos['IO_details'].append(m.details['fwd']['io_details'])
            
            # bwd for act
            op_infos["op"].append(m.__class__.__name__ + "_bwd_act")
            if hasattr(m, "weight"):
                if isinstance(m, LinearBase):
                    weight_shape = [m.get_weight().transpose(-1,-2).shape]
                else:
                    weight_shape = [m.weight.shape]
            else:
                weight_shape = []   

            op_infos["input_shapes"].append(output_shape + weight_shape)
            op_infos['output_shapes'].append(m.input_info.shapes)
            op_infos["flops"].append(m._compute_info.bwd_grad_act_flops)
            op_infos["IO"].append(m._compute_info.bwd_grad_act_accessed_mem)
            op_infos["cost"].append(m._cost_info.bwd_grad_act_time)
            op_infos['compute_only_time'].append(m.details['bwd_grad_act']['compute_details']['compute_only_time'])
            op_infos['IO_time'].append(m.details['bwd_grad_act']['io_details']['io_time'])
            op_infos['bound'].append("IO bound" if op_infos['IO_time'][-1] > op_infos['compute_only_time'][-1] else "Compute bound")
            
            if return_details:
                op_infos['compute_only_details'].append(m.details['bwd_grad_act']['compute_details'])
                op_infos['IO_details'].append(m.details['bwd_grad_act']['io_details'])
          
                
            
            # bwd for weight   
            if hasattr(m, "get_weight") and m.get_weight():
                if isinstance(m, LinearBase):
                    d_w_lhs_shape = [m.input_info.tensors[0].transpose(-1,-2).shape]
                else:
                    d_w_lhs_shape = [m.input_info.shapes]
                op_infos["op"].append(m.__class__.__name__ + "_bwd_w")
                op_infos["input_shapes"].append(d_w_lhs_shape + m.output_info_.shapes)
                op_infos['output_shapes'].append([m.get_weight().shape])
                op_infos["flops"].append(m._compute_info.bwd_grad_w_flops)
                op_infos["IO"].append(m._compute_info.bwd_grad_w_accessed_mem)
                op_infos["cost"].append(m._cost_info.bwd_grad_w_time)
                op_infos['compute_only_time'].append(m.details['bwd_grad_w']['compute_details']['compute_only_time'])
                op_infos['IO_time'].append(m.details['bwd_grad_w']['io_details']['io_time'])
                op_infos['bound'].append("IO bound" if op_infos['IO_time'][-1] > op_infos['compute_only_time'][-1] else "Compute bound")

                if return_details:
                    op_infos['compute_only_details'].append(m.details['bwd_grad_w']['compute_details'])
                    op_infos['IO_details'].append(m.details['bwd_grad_w']['io_details'])

        return op_infos
            

    def prefill(self, args, call_stk='', com_buff=None):
        # self.call_stk = f"{call_stk}{self.call_stk}"
        self.call_stk = f"rank{args.rank}-microbatch{args.microbatch}{call_stk}{self.call_stk}"
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff=com_buff)