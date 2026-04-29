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

# Page configuration
st.set_page_config(
    page_title="SimuMax Analysis Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
perf_model = PerfLLM()

class ParameterAnalyzer:
    def __init__(self):
        # Optional configuration presets
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

        # Hardware configuration options
        from simumax.utils import RELEASE_MODELS,RELEASE_SYSTEM
        self.simumax_hardware_options = {
            'A100-80GB-PCIE': RELEASE_SYSTEM['a100_pcie'],
        }
        # Model size options
        self.simumax_model_options = RELEASE_MODELS
        self.simumax_model_options = {
            'deepseek_v2': RELEASE_MODELS['deepseekv2'],
            'deepseek_v3': RELEASE_MODELS['deepseekv3'],
            'llama3_8b': RELEASE_MODELS['llama3-8b'],
            'llama3_70b': RELEASE_MODELS['llama3-70b'],
            'llama3_405b': RELEASE_MODELS['llama3-405b_padding_128'],
        }

    def analyze_parameters(self, params, hardware_name, model_name):
        """Analyze parameters and return results"""
        hardware = self.hardware_options[hardware_name]
        model = self.model_options[model_name]

        # Parameter validation
        warnings = []
        recommendations = []

        # Calculate theoretical values
        calculated_world_size = params['tp_size'] * params['pp_size'] * params['dp_size']
        actual_global_batch_size = params['micro_batch_size'] * params['micro_batch_num'] * params['dp_size']

        # Memory estimation (simplified model)
        activation_memory = (params['seq_len'] * params['micro_batch_size'] *
                           params['tp_size'] * 4 * 12) / (1024 ** 3)  # GB

        model_memory = (model['params'] * 4) / (1024 ** 3)  # Assume FP32, GB

        total_memory_estimate = activation_memory + model_memory

        # Check if memory is sufficient
        if total_memory_estimate > hardware['memory'] * 0.9:  # Leave 10% margin
            warnings.append(f"Memory estimate {total_memory_estimate:.2f}GB exceeds hardware limit {hardware['memory']}GB")
            recommendations.append("Consider reducing batch size or using model parallelism")

        # Check configuration consistency
        if params['world_size'] != calculated_world_size:
            warnings.append(f"world_size configuration inconsistent: input value {params['world_size']}, calculated value {calculated_world_size}")
            recommendations.append(f"Recommend setting world_size to {calculated_world_size}")

        if params['global_batch_size'] != actual_global_batch_size:
            warnings.append(f"global_batch_size configuration inconsistent: input value {params['global_batch_size']}, calculated value {actual_global_batch_size}")
            recommendations.append(f"Recommend setting global_batch_size to {actual_global_batch_size}")

        # Performance estimation
        communication_overhead = (params['tp_size'] + params['pp_size']) * 0.05
        efficiency_score = max(0, 100 - communication_overhead * 100)

        # Throughput estimation
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
    """Create download file"""
    # Create in-memory zip file
    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        self = perf_model
        base_info = {}
        base_info["arch"] = str(self.model_chunk_dict)
        base_info["all_param"] = self.model_config.param_numel
        base_info["act_param"] = self.model_config.activated_param_numel
        # Add text report
        zf.writestr('model_arch.txt', base_info["arch"])
        # Add JSON config
        zf.writestr('base_info.json', json.dumps(base_info, indent=2, sort_keys=False, ensure_ascii=False))
        zf.writestr('mem_result.json', str(mem_result))
        zf.writestr('compute_result.json', str(compute_result))
        zf.writestr('strategy_config.json', str(self.strategy))
        zf.writestr('system_config.json', str(self.system))
        zf.writestr('model_config.json', str(self.model_config))

    memory_file.seek(0)
    return memory_file

def main():
    # Initialize analyzer
    analyzer = ParameterAnalyzer()

    # Page title
    st.title("🚀 SimuMax Analysis Tool")
    st.markdown("Analyze and optimize distributed training configuration parameters")

    # Sidebar - Quick configuration
    with st.sidebar:
        st.header("⚡ Quick Configuration")

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

        # Hardware selection
        side_hardware = st.selectbox(
            "Select hardware configuration",
            list(analyzer.simumax_hardware_options.keys()),
            index=list(analyzer.simumax_hardware_options.keys()).index(st.session_state.selected_hardware),
            key="side_hardware",
            on_change = partial(update_hardware, main=False),
        )

        # Model selection
        side_model = st.selectbox(
            "Select model size",
            list(analyzer.simumax_model_options.keys()),
            key="side_model",
            index=list(analyzer.simumax_model_options.keys()).index(st.session_state.selected_model),
            on_change = partial(update_model, main=False),
        )
        init_paralisms = ['TP1+PP1+GPU8', 'TP1+PP2+GPU8', 'TP2+PP1+GPU8', 'EP4+PP2+GPU8', 'EP8+PP1+GPU8']
        side_paralism = st.selectbox(
            "Select parallelism method",
            init_paralisms,
            key="side_paralism",
            index=init_paralisms.index('EP8+PP1+GPU8'),
            on_change = update_paralism
        )

        st.markdown("---")


    # Hardware selection
    main_hardware = st.selectbox(
        "Select hardware configuration",
        # list(analyzer.hardware_options.keys())
        list(analyzer.simumax_hardware_options.keys()),
        index=list(analyzer.simumax_hardware_options.keys()).index(st.session_state.selected_hardware),
    )

    # Model selection
    main_model = st.selectbox(
        "Select model size",
        # list(analyzer.model_options.keys())
        list(analyzer.simumax_model_options.keys()),
        index=list(analyzer.simumax_model_options.keys()).index(st.session_state.selected_model),
    )
    perf_model.configure(
        strategy_config=StrategyConfig.init_from_format_strings("gbs8"),
        model_config=ModelConfig.init_from_config_file(analyzer.simumax_model_options[main_model]),
        system_config=SystemConfig.init_from_config_file(analyzer.simumax_hardware_options[main_hardware])
    )
    st.success(f"✅ Selected: {main_model}/{main_hardware}")

    with st.expander("📋 Model Details", expanded=True):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        model_info = perf_model.model_config
        with detail_col1:
            st.write(f"**Model Type:** {model_info.model_type}")
            st.write(f"**Model Name:** {model_info.model_name}")
            st.write(f"**Attention Type:** {model_info.attention_type}")
            st.write(f"**Hidden Size:** {model_info.hidden_size}")
            st.write(f"**Head Count:** {model_info.head_num}")
            st.write(f"**KV Head Count:** {model_info.kv_head_num}")

        with detail_col2:
            st.write(f"**Head Size:** {model_info.head_size}")
            st.write(f"**Intermediate Size:** {model_info.intermediate_size}")
            st.write(f"**Total Layers:** {model_info.layer_num}")

            if model_info.model_type == 'moe':
                st.write(f"**Dense Layers:** {model_info.dense_layers}")
                st.write(f"**Expert Count:** {model_info.expert_num}")
                st.write(f"**TopK:** {model_info.topk}")
                st.write(f"**MoE FFN Hidden Size:** {model_info.moe_ffn_hidden_size}")
                st.write(f"**MoE Shared Expert Intermediate Size:** {model_info.moe_shared_expert_intermediate_size}")

        with detail_col3:
            if model_info.attention_type == 'mla':
                st.write(f"**V Head Dim:** {model_info.v_head_dim}")
                st.write(f"**QK Head Dim:** {model_info.qk_head_dim}")
                st.write(f"**Q LoRA Rank:** {model_info.q_lora_rank}")
                st.write(f"**KV LoRA Rank:** {model_info.kv_lora_rank}")
                st.write(f"**QK Positional Embedding Dim:** {model_info.qk_pos_emb_head_dim}")

            st.write(f"**Vocab Size:** {model_info.vocab_size}")
            st.write(f"**Uses SwiGLU:** {'Yes' if model_info.use_swiglu else 'No'}")

    st.markdown("---")
    st.markdown("### Actions")
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

    analyze_btn = st.button("🚀 Start Configuration Evaluation", use_container_width=True)
    # analyze_btn = st.button("🎯 Evaluate Configuration", use_container_width=True)
    st.markdown("---")

    st.markdown("### About")
    st.info("""
    This tool analyzes distributed training parameter configurations,
    providing memory estimation, performance evaluation, and optimization suggestions.
    """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🖥️ Hardware Parameter Configuration")

        col_hw1, col_hw2, col_hw3 = st.columns(3)

        with col_hw1:
            st.subheader("Network Communication")
            if 'high_intra_node' in perf_model.system.networks:
                intra_network_bandwidth = st.number_input(
                    "Intra-node Network Bandwidth (GB/s)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=st.session_state.get('intra_network_bandwidth', float(perf_model.system.networks['high_intra_node'].bandwidth.gbps)),
                    step=0.1,
                    help="Intra-node network communication bandwidth"
                )
            if 'inter_node' in perf_model.system.networks:
                inter_network_bandwidth = st.number_input(
                    "Inter-node Network Bandwidth (GB/s)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=st.session_state.get('inter_network_bandwidth', float(perf_model.system.networks['inter_node'].bandwidth.gbps)),
                    step=0.1,
                    help="Inter-node network communication bandwidth"
                )

        with col_hw2:
            st.subheader("Compute Capability")
            compute_performance = st.number_input(
                "Compute (TFLOPS)",
                min_value=1.0,
                max_value=1000.0,
                value=float(perf_model.system.accelerator.op['matmul'].tflops),
                step=1.0,
                help="Theoretical compute performance per GPU"
            )

        # with col_hw3:
        #     st.subheader("Operator Efficiency")
        #     st.write("Compute efficiency at specified shape")
        #     op_efficiency_attn = st.slider(
        #         "Attention Operator Efficiency (%)",
        #         min_value=10,
        #         max_value=100,
        #         value=65,
        #         help="Efficiency of attention computation at actual shape"
        #     )
        #     op_efficiency_mlp = st.slider(
        #         "MLP Operator Efficiency (%)",
        #         min_value=10,
        #         max_value=100,
        #         value=75,
        #         help="Efficiency of MLP computation at actual shape"
        #     )

        st.header("📊 Parameter Configuration")

        # Parameter input table
        with st.container():
            st.subheader("Training Parameters")
            train_col1_1, train_col1_2 = st.columns(2)

            with train_col1_1:
                seq_len = st.number_input(
                    "Sequence Length (seq_len)",
                    min_value=1,
                    value=st.session_state.get('seq_len', 4096),
                    key='seq_len'
                )

                micro_batch_size = st.number_input(
                    "Micro Batch Size (mbs)",
                    min_value=1,
                    value=st.session_state.get('micro_batch_size', 1),
                    key='micro_batch_size'
                )

                global_batch_size = st.number_input(
                    "Global Batch Size (gbs)",
                    min_value=1,
                    value=st.session_state.get('global_batch_size', 256),
                    key='global_batch_size'
                )

                dtype = st.selectbox(
                    "Data Type",
                    ["bf16", "fp8"]
                )

            with train_col1_2:
                tp_size = st.number_input(
                    "TP Size",
                    min_value=1,
                    value=st.session_state.get('tp_size', 1),
                    key='tp_size'
                )

                ep_size = st.number_input(
                    "EP Size",
                    min_value=1,
                    value=st.session_state.get('ep_size', 8),
                    key='ep_size'
                )

                pp_size = st.number_input(
                    "PP Size",
                    min_value=1,
                    value=st.session_state.get('pp_size', 1),
                    key='pp_size'
                )
                with st.expander("Advanced PP Layer Options"):
                    num_layers_in_first_pipeline_stage = st.number_input(
                        "Number of layers in first Pipeline Stage",
                        min_value=-1,
                        value=st.session_state.get('num_layers_in_first_pipeline_stage', -1),
                        key='num_layers_in_first_pipeline_stage',
                        help="If -1, use default value"
                    )
                    num_layers_in_last_pipeline_stage = st.number_input(
                        "Number of layers in last Pipeline Stage",
                        min_value=-1,
                        value=st.session_state.get('num_layers_in_last_pipeline_stage', -1),
                        key='num_layers_in_last_pipeline_stage',
                        help="If -1, use default value"
                    )
                world_size = st.number_input(
                    "Number of GPUs",
                    min_value=1,
                    value=st.session_state.get('world_size', 8),
                    key='world_size'
                )

            if 'previous_model' not in st.session_state:
                st.session_state.previous_model = main_model

            if st.session_state.previous_model != main_model:
                model_config = perf_model.model_config
                # Convert dataclass to dict and batch update
                config_dict = model_config.__dict__
                for key, value in config_dict.items():
                    if not key.startswith('_'):  # Skip private attributes
                        st.session_state[key] = value
                st.session_state.previous_model = main_model
                st.rerun()

            with st.expander("🔽 Model Parameter Configuration"):#↓
                model_col1_1, model_col1_2, model_col1_3 = st.columns(3)
                with model_col1_1:
                    # normal config
                    st.markdown("##### 🎯 Common Parameters")
                    layer_num = st.number_input(
                        "Number of Layers",
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
                    st.markdown("##### 👁️ Attention Parameters")
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
                        st.markdown("##### 🏗️ MoE Parameters")
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

            st.subheader("Recompute Parameters")#🔄
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                recompute_granularity = st.selectbox(
                    "Recompute Granularity",
                    options=[None, "selective_recompute", "full_recompute"],
                    format_func=lambda x: "None" if x is None else x,
                    key='recompute_granularity'
                )
                recompute_layer_num = st.number_input(
                    "Number of Recompute Layers",
                    min_value=0,
                    value=st.session_state.get('recompute_layer_num', 0),
                    key='recompute_layer_num'
                )
            if recompute_granularity == "selective_recompute":
                with col1_2:
                    attn_recompute = st.checkbox(
                        "ATTENTION Recompute",
                        value=st.session_state.get('attn_recompute', False),
                        key='attn_recompute'
                    )
                    mla_rms_recompute = st.checkbox(
                        "MLA RMS Recompute",
                        value=st.session_state.get('mla_rms_recompute', False),
                        key='mla_rms_recompute'
                    )

                    mlp_recompute = st.checkbox(
                        "MLP Recompute",
                        value=st.session_state.get('mlp_recompute', False),
                        key='mlp_recompute'
                    )

                    mlp_rms_recompute = st.checkbox(
                        "MLP RMS Recompute",
                        value=st.session_state.get('mlp_rms_recompute', False),
                        key='mlp_rms_recompute'
                    )
            else:
                attn_recompute = False
                mla_rms_recompute = False
                mlp_recompute = False
                mlp_rms_recompute = False

    with col2:
        st.header("📈 Analysis Results")

        # Display the currently selected configuration
        st.info(f"**Hardware:** {main_hardware} | **Model:** {main_model} | **Parallelism: TP{tp_size}+EP{ep_size}+PP{pp_size}+GPU{world_size}**")

        # Run analysis when the analyze button is clicked
        if analyze_btn:
            with st.spinner("Analyzing configuration..."):
                try:
                    # 1. set model config, refer to Model Parameter Configuration
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

                    # 3. set parallel strategy
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
                    print(f"Evaluation error: {e}")
                    st.session_state.warnings = f"Evaluation error: {e}"

            # Warning messages
            if 'warnings' in st.session_state:
                st.subheader("⚠️ Warning")
                st.error(st.session_state.warnings)
                # Remove the warning message
                del st.session_state.warnings
            elif 'analysis_result' in st.session_state: # Display analysis results
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
                        st.subheader("💡 Tips / Suggestions") #❗️
                        # st.warning(f"**Peak Reserved memory ({HumanReadableSize(peak_mem_with_reserved)}) exceeds system memory limit ({perf_model.system.accelerator.mem_gbs}GB). Consider increasing GPU count or adjusting parallel/recompute strategy.**")
                        warn_idx = 1
                        if overflow_memory:
                            st.markdown(
                                f'<p style="color:red;">⚠️ <strong>{warn_idx}. Peak Reserved memory ({HumanReadableSize(peak_mem_with_reserved)}) exceeds system memory limit ({perf_model.system.accelerator.mem_gbs}GB). Consider increasing GPU count or adjusting parallel/recompute strategy.</strong></p>',
                                unsafe_allow_html=True
                            )
                            warn_idx += 1
                        if has_missed_op_efficiency:
                            st.markdown(f'<p style="color:red;">⚠️ <strong>{warn_idx}. The op shape compute efficiency below is missing, which may affect evaluation accuracy. Consider filling in the missing shape efficiency via the op testing script.</strong></p>',
                                unsafe_allow_html=True)
                            st.write(missed_op_efficiency)
                            warn_idx += 1

                    strategy:StrategyConfig  = strategy
                    # Key metrics
                    st.subheader("📊 Key Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("Computed GPU Count", strategy.world_size)
                        st.metric("Memory Estimate" \
                        " (Peak Alloc)", f"{HumanReadableSize(peak_mem)}")

                    with metric_col2:
                        st.metric("Actual Global Batch Size", strategy.global_batch_size)
                        st.metric("MFU", f"{cost_results['mfu_6nd_with_attn']*100:.2f}%")

                    with metric_col3:
                        st.metric("Token Throughput (TGS)", f"{cost_results['throughput_per_accelerator']:.2f}")
                        st.metric("Compute Throughput (TFLOPS)", f"{cost_results['throughput per GPU (TFLOP/s/GPU)']:.2f}")

                    # Memory breakdown
                    st.subheader("💾 Memory Analysis")
                    if perf_model.strategy.pp_size == 1:
                        stages = ['first_stage']
                    elif perf_model.strategy.pp_size == 2:
                        stages = ['first_stage', 'last_stage']
                    elif perf_model.strategy.pp_size > 2:
                        stages = ['first_stage', 'middle_stage', 'last_stage']
                    pp_stage_labels = {
                        'first_stage': 'Pipeline Parallel - First Stage',
                        'middle_stage': 'Pipeline Parallel - Middle Stage',
                        'last_stage': 'Pipeline Parallel - Last Stage'
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
                                st.metric("Forward Activation Memory (Per Batch)", f"{mem_result['fwd_activation_cache_per_micro_batch']}")
                                st.metric("Peak Activation Memory (Per Batch)", f"{mem_result['peak_activation_mem']}")

                            with mem_col2:
                                st.metric("Model Memory", f"{mem_result['model_mem']}")
                                with st.expander("📊 Model Memory Breakdown"):
                                    st.write(f"- MoE Part: {mem_result['model_mem_detail']['moe']}")
                                    st.write(f"- Dense Part: {mem_result['model_mem_detail']['dense']}")
                            with mem_col3:
                                st.metric("Total Peak Alloc Memory", f"{mem_result['peak_mem']}")
                                st.metric("Total Peak Reserved Memory", f"{mem_result['peak_mem_with_reserved']}")
                    # Configuration validation status
                    st.subheader("✅ Configuration Validation")
                    st.success("Configuration validation passed ✓")

                    # st.warning("Configuration validation failed ⚠")
                    # st.subheader("💡 Optimization Suggestions")
                    # with mem_col2:
                    #     st.metric("Model Memory", f"{analysis['model_memory_gb']} GB")
                    # with mem_col3:
                    #     st.metric("Total Memory", f"{analysis['memory_estimate_gb']} GB")


                    # Display result info
                    st.subheader("📊 Summary")
                    st.write(result)
                    with st.expander("Detailed Communication Bandwidth"):
                        st.write(perf_model.system.real_comm_bw)
                    # Download report
                    st.subheader("📥 Download Report")
                    zip_buffer = create_download_zip(perf_model, mem_results, cost_results)
                    st.download_button(
                        label="Download Analysis Report (ZIP)",
                        data=zip_buffer,
                        file_name=f"training_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    # del st.session_state.analysis_result
                except Exception as e:
                    print(f"Analysis error: {e}")
                    st.subheader("⚠️ Warning")
                    st.error(f"Analysis error: {e}")
        else:
            st.info("Please click the 'Start Configuration Evaluation' button to begin the analysis")

if __name__ == "__main__":
    main()
