from functools import partial
import streamlit as st
import re
import os
import copy
os.environ['SIMU_CHECK'] = '1'
import json
import zipfile
from datetime import datetime
from io import BytesIO
from simumax.core.config import ParameterExtractor
from simumax.core.utils import HumanReadableSize
from simumax.core.perf_llm import PerfLLM
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig

# 设置页面配置
st.set_page_config(
    page_title="SimuMax分析工具",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
perf_model = PerfLLM()

class ParameterAnalyzer:
    def __init__(self):
        # 可选的配置预设
        self.config_presets = {
            'small': {
                'seq_len': 512,
                'micro_batch_size': 8,
                'micro_batch_num': 2,
                'global_batch_size': 16,
                'tp_size': 2,
                'ep_size': 1,
                'pp_size': 2,
                'dp_size': 2,
                'world_size': 4
            },
            'medium': {
                'seq_len': 1024,
                'micro_batch_size': 16,
                'micro_batch_num': 4,
                'global_batch_size': 64,
                'tp_size': 4,
                'ep_size': 1,
                'pp_size': 2,
                'dp_size': 4,
                'world_size': 32
            },
            'large': {
                'seq_len': 2048,
                'micro_batch_size': 32,
                'micro_batch_num': 8,
                'global_batch_size': 256,
                'tp_size': 8,
                'ep_size': 1,
                'pp_size': 4,
                'dp_size': 8,
                'world_size': 256
            }
        }
        
        # 硬件配置选项
        from simumax.utils import RELEASE_MODELS,RELEASE_SYSTEM
        self.simumax_hardware_options = {
            'A100-80GB-PCIE': RELEASE_SYSTEM['a100_pcie'],
        }
        # 模型规模选项
        self.simumax_model_options = RELEASE_MODELS
        self.simumax_model_options = {
            'deepseek_v2': RELEASE_MODELS['deepseekv2'],
            'deepseek_v3': RELEASE_MODELS['deepseekv3'],
            'llama3_8b': RELEASE_MODELS['llama3-8b'],
            'llama3_70b': RELEASE_MODELS['llama3-70b'],
            'llama3_405b': RELEASE_MODELS['llama3-405b_padding_128'],
        }

    def analyze_parameters(self, params, hardware_name, model_name):
        """分析参数并返回结果"""
        hardware = self.hardware_options[hardware_name]
        model = self.model_options[model_name]
        
        # 参数验证
        warnings = []
        recommendations = []
        
        # 计算理论值
        calculated_world_size = params['tp_size'] * params['pp_size'] * params['dp_size']
        actual_global_batch_size = params['micro_batch_size'] * params['micro_batch_num'] * params['dp_size']
        
        # 内存估算 (简化模型)
        activation_memory = (params['seq_len'] * params['micro_batch_size'] * 
                           params['tp_size'] * 4 * 12) / (1024 ** 3)  # GB
        
        model_memory = (model['params'] * 4) / (1024 ** 3)  # 假设 FP32, GB
        
        total_memory_estimate = activation_memory + model_memory
        
        # 检查内存是否足够
        if total_memory_estimate > hardware['memory'] * 0.9:  # 留10%余量
            warnings.append(f"内存估计 {total_memory_estimate:.2f}GB 超过硬件限制 {hardware['memory']}GB")
            recommendations.append("考虑减小批次大小或使用模型并行")
        
        # 检查配置一致性
        if params['world_size'] != calculated_world_size:
            warnings.append(f"world_size 配置不一致: 输入值 {params['world_size']}, 计算值 {calculated_world_size}")
            recommendations.append(f"建议将 world_size 设置为 {calculated_world_size}")
        
        if params['global_batch_size'] != actual_global_batch_size:
            warnings.append(f"global_batch_size 配置不一致: 输入值 {params['global_batch_size']}, 计算值 {actual_global_batch_size}")
            recommendations.append(f"建议将 global_batch_size 设置为 {actual_global_batch_size}")
        
        # 性能估算
        communication_overhead = (params['tp_size'] + params['pp_size']) * 0.05
        efficiency_score = max(0, 100 - communication_overhead * 100)
        
        # 吞吐量估算
        estimated_throughput = (params['global_batch_size'] / 
                              (1 + communication_overhead))  # tokens/step
        
        return {
            'parameters': params,
            'analysis': {
                'calculated_world_size': calculated_world_size,
                'actual_global_batch_size': actual_global_batch_size,
                'memory_estimate_gb': round(total_memory_estimate, 2),
                'activation_memory_gb': round(activation_memory, 2),
                'model_memory_gb': round(model_memory, 2),
                'efficiency_score': round(efficiency_score, 1),
                'estimated_throughput': round(estimated_throughput, 2),
                'communication_overhead': round(communication_overhead, 2),
                'hardware_utilization': round((total_memory_estimate / hardware['memory']) * 100, 1),
                'warnings': warnings,
                'recommendations': recommendations,
                'is_config_valid': len(warnings) == 0
            },
            'hardware_info': {'name': hardware_name, **hardware},
            'model_info': {'name': model_name, **model}
        }

def create_download_zip(perf_model:PerfLLM, mem_result, compute_result):
    """创建下载文件"""
    # 创建内存zip文件
    memory_file = BytesIO()
        
    with zipfile.ZipFile(memory_file, 'w') as zf:
        self = perf_model
        base_info = {}
        base_info["arch"] = str(self.model_chunk_dict)
        base_info["all_param"] = self.model_config.param_numel
        base_info["act_param"] = self.model_config.activated_param_numel
        # 添加文本报告
        zf.writestr('model_arch.txt', base_info["arch"])
        # 添加JSON配置
        zf.writestr('base_info.json', json.dumps(base_info, indent=2, sort_keys=False, ensure_ascii=False))
        zf.writestr('mem_result.json', str(mem_result))
        zf.writestr('compute_result.json', str(compute_result))
        zf.writestr('strategy_config.json', str(self.strategy))
        zf.writestr('system_config.json', str(self.system))
        zf.writestr('model_config.json', str(self.model_config))
    
    memory_file.seek(0)
    return memory_file

def main():
    # 初始化分析器
    analyzer = ParameterAnalyzer()
    
    # 页面标题
    st.title("🚀 SimuMax分析工具")
    st.markdown("分析和优化分布式训练配置参数")
    
    # 侧边栏 - 快速配置
    with st.sidebar:
        st.header("⚡ 快速配置")

        if 'selected_hardware' not in st.session_state:
            st.session_state.selected_hardware = "A100-80GB-PCIE"
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "deepseek_v3"

        def update_hardware(main=False):
            if main:
                st.session_state.selected_hardware = st.session_state.main_hardware
            else:
                st.session_state.selected_hardware = st.session_state.side_hardware
            

        def update_model(main=False):
            if main:
                st.session_state.selected_model = st.session_state.main_model
            else:
                st.session_state.selected_model = st.session_state.side_model
            

        def update_paralism():
            paralism = st.session_state.side_paralism
            param_patterns = {
                'tp_size': (r'TP(\d+)', 1),
                'ep_size': (r'EP(\d+)', 1),
                'pp_size': (r'PP(\d+)', 1),
                'world_size': (r'GPU(\d+)', 8),
            }
            paralism_params = ParameterExtractor(param_patterns=param_patterns).extract_parameters(paralism)
            print(paralism_params)
            for key, value in paralism_params.items():
                if key in st.session_state:
                    st.session_state[key] = value

        # 硬件选择
        side_hardware = st.selectbox(
            "选择硬件配置",
            list(analyzer.simumax_hardware_options.keys()),
            index=list(analyzer.simumax_hardware_options.keys()).index(st.session_state.selected_hardware),
            key="side_hardware",
            on_change = partial(update_hardware, main=False),
        )
        
        # 模型选择
        side_model = st.selectbox(
            "选择模型规模",
            list(analyzer.simumax_model_options.keys()),
            key="side_model",
            index=list(analyzer.simumax_model_options.keys()).index(st.session_state.selected_model),
            on_change = partial(update_model, main=False),
        )
        init_paralisms = ['TP1+PP1+GPU8', 'TP1+PP2+GPU8', 'TP2+PP1+GPU8', 'EP4+PP2+GPU8', 'EP8+PP1+GPU8']
        side_paralism = st.selectbox(
            "选择并行方式",
            init_paralisms,
            key="side_paralism",
            index=init_paralisms.index('EP8+PP1+GPU8'),
            on_change = update_paralism
        )

        st.markdown("---")
    
    
    # 硬件选择
    main_hardware = st.selectbox(
        "选择硬件配置",
        # list(analyzer.hardware_options.keys())
        list(analyzer.simumax_hardware_options.keys()),
        index=list(analyzer.simumax_hardware_options.keys()).index(st.session_state.selected_hardware),
    )

    # 模型选择
    main_model = st.selectbox(
        "选择模型规模",
        # list(analyzer.model_options.keys())
        list(analyzer.simumax_model_options.keys()),
        index=list(analyzer.simumax_model_options.keys()).index(st.session_state.selected_model),
    )
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_format_strings("gbs8"),
        model_config=ModelConfig.init_from_config_file(analyzer.simumax_model_options[main_model]),
        system_config=SystemConfig.init_from_config_file(analyzer.simumax_hardware_options[main_hardware])
    )
    st.success(f"✅ 已选择: {main_model}/{main_hardware}")

    with st.expander("📋 模型详细信息", expanded=True):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        model_info = perf_model.model_config
        with detail_col1:
            st.write(f"**模型类型:** {model_info.model_type}")
            st.write(f"**模型名称:** {model_info.model_name}")
            st.write(f"**注意力类型:** {model_info.attention_type}")
            st.write(f"**隐藏层大小:** {model_info.hidden_size}")
            st.write(f"**头数量:** {model_info.head_num}")
            st.write(f"**KV头数量:** {model_info.kv_head_num}")
        
        with detail_col2:
            st.write(f"**头大小:** {model_info.head_size}")
            st.write(f"**中间隐藏层大小:** {model_info.intermediate_size}")
            st.write(f"**总层数:** {model_info.layer_num}")
            
            if model_info.model_type == 'moe':
                st.write(f"**稠密层数:** {model_info.dense_layers}")
                st.write(f"**专家数量:** {model_info.expert_num}")
                st.write(f"**TopK:** {model_info.topk}")
                st.write(f"**MoE FFN隐藏层大小:** {model_info.moe_ffn_hidden_size}")
                st.write(f"**MoE共享专家隐藏层大小:** {model_info.moe_shared_expert_intermediate_size}")
        
        with detail_col3:
            if model_info.attention_type == 'mla':
                st.write(f"**V注意力头维度:** {model_info.v_head_dim}")
                st.write(f"**QK注意力头维度:** {model_info.qk_head_dim}")
                st.write(f"**Q LoRA秩:** {model_info.q_lora_rank}")
                st.write(f"**KV LoRA秩:** {model_info.kv_lora_rank}")
                st.write(f"**QK位置编码维度:** {model_info.qk_pos_emb_head_dim}")
            
            st.write(f"**词表大小:** {model_info.vocab_size}")
            st.write(f"**使用SwiGLU:** {'是' if model_info.use_swiglu else '否'}")
    
    st.markdown("---")
    st.markdown("### 操作")
    st.markdown("""
    <style>
        div[data-testid="stButton"] > button {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(52, 152, 219, 0.4);
            width: 100%;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(52, 152, 219, 0.6);
            background: linear-gradient(45deg, #2980b9, #3498db);
        }
                
    </style>
    """, unsafe_allow_html=True)

    analyze_btn = st.button("🚀 开始评估配置", use_container_width=True)
    # analyze_btn = st.button("🎯 评估配置", use_container_width=True)
    st.markdown("---")

    st.markdown("### 关于")
    st.info("""
    本工具用于分析分布式训练参数配置，
    提供内存估算、性能评估和优化建议。
    """)

    # 主内容区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🖥️ 硬件参数配置")

        col_hw1, col_hw2, col_hw3 = st.columns(3)

        with col_hw1:
            st.subheader("网络通信")
            if 'high_intra_node' in perf_model.system.networks:
                intra_network_bandwidth = st.number_input(
                    "机内网络带宽 (GB/s)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=st.session_state.get('intra_network_bandwidth', float(perf_model.system.networks['high_intra_node'].bandwidth.gbps)),
                    step=0.1,
                    help="节点内网络通信带宽"
                )
            if 'inter_node' in perf_model.system.networks:
                inter_network_bandwidth = st.number_input(
                    "机间网络带宽 (GB/s)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=st.session_state.get('inter_network_bandwidth', float(perf_model.system.networks['inter_node'].bandwidth.gbps)),
                    step=0.1,
                    help="节点间网络通信带宽"
                )

        with col_hw2:
            st.subheader("计算能力")
            compute_performance = st.number_input(
                "算力 (TFLOPS)",
                min_value=1.0,
                max_value=1000.0,
                value=float(perf_model.system.accelerator.op['matmul'].tflops),
                step=1.0,
                help="单卡理论计算性能"
            )

        # with col_hw3:
        #     st.subheader("算子效率")
        #     st.write("指定shape下的计算效率")
        #     op_efficiency_attn = st.slider(
        #         "注意力算子效率 (%)",
        #         min_value=10,
        #         max_value=100,
        #         value=65,
        #         help="注意力计算在实际shape下的效率"
        #     )
        #     op_efficiency_mlp = st.slider(
        #         "MLP算子效率 (%)", 
        #         min_value=10,
        #         max_value=100,
        #         value=75,
        #         help="MLP计算在实际shape下的效率"
        #     )

        st.header("📊 参数配置")
        
        # 参数输入表格
        with st.container():
            st.subheader("训练参数")
            train_col1_1, train_col1_2 = st.columns(2)
            
            with train_col1_1:
                seq_len = st.number_input(
                    "序列长度 (seq_len)",
                    min_value=1,
                    value=st.session_state.get('seq_len', 4096),
                    key='seq_len'
                )
                
                micro_batch_size = st.number_input(
                    "微批次大小 (mbs)",
                    min_value=1,
                    value=st.session_state.get('micro_batch_size', 1),
                    key='micro_batch_size'
                )
                
                global_batch_size = st.number_input(
                    "全局批次大小 (gbs)",
                    min_value=1,
                    value=st.session_state.get('global_batch_size', 256),
                    key='global_batch_size'
                )
             
                dtype = st.selectbox(
                    "数据类型",
                    ["bf16", "fp8"]
                )
                
            with train_col1_2:
                tp_size = st.number_input(
                    "TP大小",
                    min_value=1,
                    value=st.session_state.get('tp_size', 1),
                    key='tp_size'
                )
            
                ep_size = st.number_input(
                    "EP大小",
                    min_value=1,
                    value=st.session_state.get('ep_size', 8),
                    key='ep_size'
                )
                
                pp_size = st.number_input(
                    "PP大小",
                    min_value=1,
                    value=st.session_state.get('pp_size', 1),
                    key='pp_size'
                )
                with st.expander("PP层数高级选项"):
                    num_layers_in_first_pipeline_stage = st.number_input(
                        "首个 Pipeline Stage的层数",
                        min_value=-1,
                        value=st.session_state.get('num_layers_in_first_pipeline_stage', -1),
                        key='num_layers_in_first_pipeline_stage',
                        help="如果为-1，则表示使用默认值"
                    )
                    num_layers_in_last_pipeline_stage = st.number_input(
                        "最后一个 Pipeline Stage的层数",
                        min_value=-1,
                        value=st.session_state.get('num_layers_in_last_pipeline_stage', -1),
                        key='num_layers_in_last_pipeline_stage',
                        help="如果为-1，则表示使用默认值"
                    )
                world_size = st.number_input(
                    "卡数",
                    min_value=1,
                    value=st.session_state.get('world_size', 8),
                    key='world_size'
                )
                
            if 'previous_model' not in st.session_state:
                st.session_state.previous_model = main_model

            if st.session_state.previous_model != main_model:
                model_config = perf_model.model_config
                # 将dataclass转换为字典并批量更新
                config_dict = model_config.__dict__
                for key, value in config_dict.items():
                    if not key.startswith('_'):  # 跳过私有属性
                        st.session_state[key] = value
                st.session_state.previous_model = main_model
                st.rerun()

            with st.expander("🔽 模型参数配置"):#↓
                model_col1_1, model_col1_2, model_col1_3 = st.columns(3)
                with model_col1_1:
                    # noraml config
                    st.markdown("##### 🎯常规参数")
                    layer_num = st.number_input(
                        "层数",
                        min_value=1,
                        value=st.session_state.get('layer_num', perf_model.model_config.layer_num),
                        key='layer_num'
                    )
                    hidden_size = st.number_input(
                        "hidden_size",
                        min_value=1,
                        value=st.session_state.get('hidden_size', perf_model.model_config.hidden_size),
                        key='hidden_size'
                    )
                    intermediate_size = st.number_input(
                        "intermediate_size",
                        min_value=1,
                        value=st.session_state.get('intermediate_size', perf_model.model_config.intermediate_size),
                        key='intermediate_size'
                    )
                    vocab_size = st.number_input(
                        "vocab_size",
                        min_value=1,
                        value=st.session_state.get('vocab_size', perf_model.model_config.vocab_size),
                        key='vocab_size'
                    )
                with model_col1_2:
                    # attention related
                    st.markdown("##### 👁️Attention参数")
                    head_num = st.number_input(
                        "head_num",
                        min_value=1,
                        value=st.session_state.get('head_num', perf_model.model_config.head_num),
                        key='head_num'
                    )
                    kv_head_num = st.number_input(
                        "kv_head_num",
                        min_value=1,
                        value=st.session_state.get('kv_head_num', perf_model.model_config.kv_head_num),
                        key='kv_head_num'
                    )
                    head_size = st.number_input(
                        "head_size",
                        min_value=1,
                        value=st.session_state.get('head_size', perf_model.model_config.head_size),
                        key='head_size'
                    )
                    if perf_model.model_config.attention_type == 'mla':
                        qk_head_dim = st.number_input(
                            "qk_head_dim",
                            min_value=1,
                            value=st.session_state.get('qk_head_dim', perf_model.model_config.qk_head_dim),
                            key='qk_head_dim'
                        )
                        v_head_dim = st.number_input(
                            "v_head_dim",
                            min_value=1,
                            value=st.session_state.get('v_head_dim', perf_model.model_config.v_head_dim),
                            key='v_head_dim'
                        )
                        qk_pos_emb_head_dim = st.number_input(
                            "qk_pose_emb_head_dim",
                            min_value=1,
                            value=st.session_state.get('qk_pose_emb_head_dim', perf_model.model_config.qk_pos_emb_head_dim),
                            key='qk_pose_emb_head_dim'
                        )
                        q_lora_rank = st.number_input(
                            "q_lora_rank",
                            min_value=1,
                            value=st.session_state.get('q_lora_rank', perf_model.model_config.q_lora_rank),
                            key='q_lora_rank'
                        )
                        kv_lora_rank = st.number_input(
                            "kv_lora_rank",
                            min_value=1,
                            value=st.session_state.get('kv_lora_rank', perf_model.model_config.kv_lora_rank),
                            key='kv_lora_rank'
                        )
                if perf_model.model_config.model_type == 'moe':
                    with model_col1_3:
                        # moe related
                        st.markdown("##### 🏗️Moe参数")
                        dense_layers = st.number_input(
                            "dense_layers",
                            min_value=1,
                            value=st.session_state.get('dense_layers', perf_model.model_config.dense_layers),
                            key='dense_layers'
                        )
                        expert_num = st.number_input(
                            "expert_num",
                            min_value=1,
                            value=st.session_state.get('expert_num', perf_model.model_config.expert_num),
                            key='expert_num'
                        )
                        topk = st.number_input(
                            "topk",
                            min_value=1,
                            value=st.session_state.get('topk', perf_model.model_config.topk),
                            key='topk'
                        )
                        moe_ffn_hidden_size = st.number_input(
                            "moe_ffn_hidden_size",
                            min_value=1,
                            value=st.session_state.get('moe_ffn_hidden_size', perf_model.model_config.moe_ffn_hidden_size),
                            key='moe_ffn_hidden_size'
                        )
                        moe_shared_expert_intermediate_size = st.number_input(
                        "moe_shared_expert_intermediate_size",
                        min_value=1,
                        value=st.session_state.get('moe_shared_expert_intermediate_size', perf_model.model_config.moe_shared_expert_intermediate_size),
                        key='moe_shared_expert_intermediate_size'
                    )

            st.subheader("重计算参数")#🔄
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                recompute_granularity = st.selectbox(
                    "重计算粒度",
                    options=[None, "selective_recompute", "full_recompute"],
                    format_func=lambda x: "无" if x is None else x,
                    key='recompute_granularity'
                )
                recompute_layer_num = st.number_input(
                    "重计算层数",
                    min_value=0,
                    value=st.session_state.get('recompute_layer_num', 0),
                    key='recompute_layer_num'
                )
            if recompute_granularity == "selective_recompute":
                with col1_2:
                    attn_recompute = st.checkbox(
                        "ATTENTION重计算",
                        value=st.session_state.get('attn_recompute', False),
                        key='attn_recompute'
                    )
                    mla_rms_recompute = st.checkbox(
                        "MLA RMS重计算",
                        value=st.session_state.get('mla_rms_recompute', False),
                        key='mla_rms_recompute'
                    )
                    
                    mlp_recompute = st.checkbox(
                        "MLP重计算",
                        value=st.session_state.get('mlp_recompute', False),
                        key='mlp_recompute'
                    )
                    
                    mlp_rms_recompute = st.checkbox(
                        "MLP RMS重计算", 
                        value=st.session_state.get('mlp_rms_recompute', False),
                        key='mlp_rms_recompute'
                    )
            else:
                attn_recompute = False
                mla_rms_recompute = False
                mlp_recompute = False
                mlp_rms_recompute = False

    with col2:
        st.header("📈 分析结果")
        
        # 显示当前选择的配置
        st.info(f"**硬件:** {main_hardware} | **模型:** {main_model} | **并行:TP{tp_size}+EP{ep_size}+PP{pp_size}+GPU{world_size}**")
        
        # 当点击分析按钮时执行分析
        if analyze_btn:
            with st.spinner("正在分析配置..."):    
                try:
                    # 1. set model config, refer 模型参数配置
                    ## normal model config
                    perf_model.model_config.layer_num = layer_num
                    perf_model.model_config.hidden_size = hidden_size
                    perf_model.model_config.intermediate_size = intermediate_size
                    perf_model.model_config.vocab_size = vocab_size
                    perf_model.strategy.dispatch_probs = True

                    ## attention model config
                    perf_model.model_config.head_num = head_num
                    perf_model.model_config.kv_head_num = kv_head_num
                    perf_model.model_config.head_size = head_size
                    if perf_model.model_config.attention_type == 'mla':
                        perf_model.model_config.qk_head_dim = qk_head_dim
                        perf_model.model_config.v_head_dim = v_head_dim
                        perf_model.model_config.qk_pos_emb_head_dim = qk_pos_emb_head_dim
                        perf_model.model_config.q_lora_rank = q_lora_rank
                        perf_model.model_config.kv_lora_rank = kv_lora_rank
                    if perf_model.model_config.model_type == 'moe':
                        perf_model.model_config.dense_layers = dense_layers
                        perf_model.model_config.expert_num = expert_num
                        perf_model.model_config.topk = topk
                        perf_model.model_config.moe_ffn_hidden_size = moe_ffn_hidden_size
                        perf_model.model_config.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size

                    # 2. set bw and tflops
                    if 'high_intra_node' in perf_model.system.networks:
                        perf_model.system.networks['high_intra_node'].bandwidth.gbps = intra_network_bandwidth
                    if 'inter_node' in perf_model.system.networks:
                        perf_model.system.networks['inter_node'].bandwidth.gbps = inter_network_bandwidth
                    perf_model.system.accelerator.op['default'].tflops = compute_performance
                    perf_model.system.accelerator.op['matmul'].tflops = compute_performance
                    perf_model.system.accelerator.op['fp8_matmul'].tflops = compute_performance
                    perf_model.system.accelerator.op['sdp_fwd'].tflops = compute_performance
                    perf_model.system.accelerator.op['sdp_bwd'].tflops = compute_performance
                    perf_model.system.accelerator.op['group_matmul'].tflops = compute_performance
                    perf_model.system.accelerator.op['fp8_group_matmul'].tflops = compute_performance

                    perf_model.model_config.moe_pad_expert_input_to_capacity = True
                    # TODO(sherry): add op efficiency

                    # 3. set paralilel strategy
                    perf_model.strategy.seq_len = seq_len
                    perf_model.strategy.micro_batch_size = micro_batch_size
                    perf_model.strategy.tp_size = tp_size
                    perf_model.strategy.ep_size = ep_size
                    perf_model.strategy.pp_size = pp_size
                    perf_model.strategy.world_size = world_size
                    
                    if num_layers_in_last_pipeline_stage != -1:
                        perf_model.strategy.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage
                    if num_layers_in_first_pipeline_stage != -1:
                        perf_model.strategy.num_layers_in_first_pipeline_stage = num_layers_in_first_pipeline_stage
                    perf_model.strategy.reset_global_batch_size(global_batch_size)


                    # 4. set recompute strategy
                    perf_model.strategy.enable_recompute = True
                    if recompute_granularity == 'full_recompute':
                        perf_model.strategy.recompute_granularity = 'full_block' 
                    elif recompute_granularity == 'selective_recompute':
                        perf_model.strategy.recompute_granularity = 'selective_recompute'
                    else:
                        perf_model.strategy.recompute_granularity = None
                    perf_model.strategy.recompute_layer_num = recompute_layer_num
                    perf_model.strategy.attn_recompute = attn_recompute
                    perf_model.strategy.mla_rms_recompute = mla_rms_recompute
                    perf_model.strategy.mlp_recompute = mlp_recompute
                    perf_model.strategy.mlp_rms_recompute = mlp_rms_recompute

                    # 5. set dtype
                    perf_model.strategy.dtype = 'bf16'
                    if dtype == "fp8":
                        perf_model.strategy.fp8 = True
                    
                    # 6. run estimate
                    perf_model.run_estimate()
                    result = perf_model.analysis()
                    mem_results = perf_model.analysis_mem().data
                    cost_results = perf_model.analysis_cost().data
                    
                    st.session_state.analysis_result = (result, mem_results, cost_results, perf_model.strategy)
                except Exception as e:
                    print(f"评估报错:{e}")
                    st.session_state.warnings = f"评估报错:{e}"
            
            # 警告信息
            if 'warnings' in st.session_state:
                st.subheader("⚠️ 警告信息")
                st.error(st.session_state.warnings)
                # 删除警告信息
                del st.session_state.warnings
            elif 'analysis_result' in st.session_state: # 显示分析结果
                try:
                    result, mem_results, cost_results, strategy = st.session_state.analysis_result
                    peak_mem = max(perf_model.get_pp_stage_peak_mem(mem_results, "peak_mem", False).values())
                    peak_mem_with_reserved = max(perf_model.get_pp_stage_peak_mem(mem_results, "peak_mem_with_reserved", False).values())
                    
                    has_missed_op_efficiency = len(perf_model.system.miss_efficiency) > 0
                    if has_missed_op_efficiency:
                        missed_op_efficiency = copy.deepcopy(perf_model.system.miss_efficiency)
                        perf_model.system.reset_record_info()
                        
                    overflow_memory = peak_mem_with_reserved/2**30 > perf_model.system.accelerator.mem_gbs
                    if overflow_memory or has_missed_op_efficiency:
                        st.subheader("💡 提示/建议") #❗️
                        # st.warning(f"**峰值Reserved显存({HumanReadableSize(peak_mem_with_reserved)})超过系统显存限制({perf_model.system.accelerator.mem_gbs}GB), 建议增加卡数或调整并行策略、重计算策略**")
                        warn_idx = 1
                        if overflow_memory:
                            st.markdown(
                                f'<p style="color:red;">⚠️ <strong>{warn_idx}. 峰值Reserved显存({HumanReadableSize(peak_mem_with_reserved)})超过系统显存限制({perf_model.system.accelerator.mem_gbs}GB), 建议增加卡数或调整并行策略、重计算策略</strong></p>',
                                unsafe_allow_html=True
                            )
                            warn_idx += 1
                        if has_missed_op_efficiency:
                            st.markdown(f'<p style="color:red;">⚠️ <strong>{warn_idx}. 下面的op shape计算效率缺失,可能影响评估准确度,建议通过op测试脚本补全缺失shape的计算效率</strong></p>',
                                unsafe_allow_html=True)
                            st.write(missed_op_efficiency)
                            warn_idx += 1

                    strategy:StrategyConfig  = strategy
                    # 关键指标
                    st.subheader("📊 关键指标")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("计算的卡数", strategy.world_size)
                        st.metric("内存估计" \
                        "(Peak Alloc)", f"{HumanReadableSize(peak_mem)}")
                        
                    with metric_col2:
                        st.metric("实际全局批次大小", strategy.global_batch_size)
                        st.metric("MFU", f"{cost_results['mfu_6nd_with_attn']*100:.2f}%")
                        
                    with metric_col3:
                        st.metric("Token吞吐量(TGS)", f"{cost_results['throughput_per_accelerator']:.2f}")
                        st.metric("算力吞吐量(TFLOPS)", f"{cost_results['throughput per GPU (TFLOP/s/GPU)']:.2f}")
                    
                    # 内存细分
                    st.subheader("💾 内存分析")
                    if perf_model.strategy.pp_size == 1:
                        stages = ['first_stage']
                    elif perf_model.strategy.pp_size == 2:
                        stages = ['first_stage', 'last_stage']
                    elif perf_model.strategy.pp_size > 2:
                        stages = ['first_stage', 'middle_stage', 'last_stage']
                    pp_stage_labels = {
                        'first_stage': 'Pipeline并行第一阶段',
                        'middle_stage': 'Pipeline并行中间阶段',
                        'last_stage': 'Pipeline并行最后阶段'
                    }
                    for stage in stages:
                        if perf_model.strategy.pp_size > 1:
                            context = st.expander(f"🔽 {pp_stage_labels[stage]}", expanded=True)
                        else:
                            context = st.container()
                        with context:#📁
                            if perf_model.strategy.pp_size > 1:
                                st.markdown(f"##### {pp_stage_labels[stage]}")
                                mem_result = mem_results[stage]
                            else:
                                mem_result = mem_results
                            mem_col1, mem_col2, mem_col3 = st.columns(3)
                            
                            with mem_col1:
                                st.metric("前向激活内存(单Batch)", f"{mem_result['fwd_activation_cache_per_micro_batch']}")
                                st.metric("1F1B峰值激活内存(单Batch)", f"{mem_result['peak_activation_mem_in_1F1B']}")

                            with mem_col2:
                                st.metric("模型内存", f"{mem_result['model_mem']}")
                                with st.expander("📊 模型内存细分"):
                                    st.write(f"- MoE部分: {mem_result['model_mem_detail']['moe']}")
                                    st.write(f"- Dense部分: {mem_result['model_mem_detail']['dense']}")
                            with mem_col3:
                                st.metric("总峰值Alloc显存", f"{mem_result['peak_mem']}")
                                st.metric("总峰值Reserved显存", f"{mem_result['peak_mem_with_reserved']}")
                    # 配置验证状态
                    st.subheader("✅ 配置验证")
                    st.success("配置验证通过 ✓")
                    
                    # st.warning("配置验证未通过 ⚠")
                    # st.subheader("💡 优化建议")
                    # with mem_col2:
                    #     st.metric("模型内存", f"{analysis['model_memory_gb']} GB")
                    # with mem_col3:
                    #     st.metric("总内存", f"{analysis['memory_estimate_gb']} GB")
                    
                    
                    # 显示resutlt信息
                    st.subheader("📊 汇总信息")
                    st.write(result)
                    with st.expander("详细通信带宽"):
                        st.write(perf_model.system.real_comm_bw)
                    # 下载报告
                    st.subheader("📥 下载报告")
                    zip_buffer = create_download_zip(perf_model, mem_results, cost_results)
                    st.download_button(
                        label="下载分析报告 (ZIP)",
                        data=zip_buffer,
                        file_name=f"training_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    # del st.session_state.analysis_result
                except Exception as e:
                    print(f"分析报错:{e}")
                    st.subheader("⚠️ 警告信息")
                    st.error(f"分析报错:{e}")
        else:
            st.info("请点击'开始评估配置'按钮开始分析")

if __name__ == "__main__":
    main()