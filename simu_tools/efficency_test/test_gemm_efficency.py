import os
import time
import json
from argparse import ArgumentParser
import pandas as pd
from matplotlib import pyplot as plt
import torch
from simumax.core.config import ModelConfig, StrategyConfig, SystemConfig
from simumax.core.perf_llm import PerfLLM
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm
)
from simumax.utils import get_simu_model_config, get_simu_system_config, get_simu_strategy_config
from simu_tools.efficency_test.utils import get_system_name, sync_device, get_torch_profiler, get_all_test_model_configs, get_test_seq_len_list, get_test_mbs_list, get_test_tp_list, get_test_ep_list

def prepare_tensors(B: int, M: int, K: int, N: int,  dtype,  device):
    """
    准备输入矩阵和workspace
    Args:
        B: batch size
        M: A矩阵行数
        N: B矩阵列数
        K: A矩阵列数/B矩阵行数
    Returns:
        A: shape [B, M, K] (TN布局)
        B: shape [K, N] (TN布局)
        workspace: 预分配空间
    """
    if dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("dtype must be 'bf16'")
    
    A = torch.randn((B, M, K), device=device, dtype=dtype)
    B_matrix = torch.randn((K, N), device=device, dtype=dtype)
    
    # 为general_gemm分配workspace (典型大小32MB)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    return A, B_matrix, workspace

def run_te_gemm(B: int, M: int, K: int, N: int, dtype, device, layout, accumulate, out_dtype, test_steps, warmup_steps):
    """
    执行general_gemm操作
    Args:
        A: [B, M, K]
        B: [K, N]
    Returns:
        output: [B, M, N]
    """
    
    out_dtype = torch.bfloat16 if out_dtype == 'bf16' else torch.float32
    out_dtype = torch.bfloat16

    if layout == 'NN':
        B_matrix = torch.randn((K, N), device=device, dtype=torch.bfloat16)
        A_matrix = torch.randn((M, B,  K), device=device, dtype=torch.bfloat16)
        output = torch.empty((M, B,  N), device=device, dtype=out_dtype)
    elif layout == 'TN':
        B_matrix = torch.randn((N, K), device=device, dtype=torch.bfloat16)
        A_matrix = torch.randn((M, B,  K), device=device, dtype=torch.bfloat16)
        output = torch.empty((M, B,  N), device=device, dtype=out_dtype)
    elif layout == 'NT':
        B_matrix = torch.randn((K, B,  N), device=device, dtype=torch.bfloat16) # N
        A_matrix = torch.randn((K, B,  M), device=device, dtype=torch.bfloat16) # T
        output = torch.empty((M, N), device=device, dtype=out_dtype)
    else:
        raise ValueError("layout must be 'NN' or 'TN' or 'NN'")
    # 为general_gemm分配workspace (典型大小32MB)
    print(f'dtype: {dtype}, out_dtype: {out_dtype}, layout: {layout}')
    if dtype == 'fp8':
        # quantizer = Float8Quantizer(torch.tensor(1.0), amax=torch.tensor(1.0), fp8_dtype=tex.DType.kBFloat16)
        te_dtype = tex.DType.kFloat8E4M3
        quantizer = Float8Quantizer(
            scale=torch.full([1], 1.0).to(device).squeeze(),
            amax=torch.full([1], 1.0).to(device),
            fp8_dtype=te_dtype,
        )
        quantizer.set_usage(rowwise=True, columnwise=False)
        A_matrix = quantizer(A_matrix)
        B_matrix = quantizer(B_matrix)
    print(f'A_matrix: {A_matrix.shape} {A_matrix.device} {A_matrix.dtype}, B_matrix: {B_matrix.shape} {B_matrix.device} {B_matrix.dtype}, output: {output.shape} {output.device} {output.dtype}')

    assert warmup_steps < test_steps, "warmup_steps should be less than test_steps"
    print(f'B={B_matrix.shape}, A={A_matrix.shape}, out={output.shape}, layout={layout}')
    sync_device(device)

    for i in range(test_steps):
        # 处理batch维度：对每个batch元素执行GEMM
        if i == warmup_steps:
            sync_device(device)
            start_time = time.time()
        general_gemm(
            B_matrix, 
            A_matrix, 
            get_workspace(),
            out_dtype=out_dtype,
            quantization_params=None,
            out=output,
            accumulate=accumulate,
            layout=layout,
            grad =  True if layout in ['NT', 'NN'] else False,
            use_split_accumulator = True if layout in ['NT', 'NN'] else False,
        )
        """
        wgrad, grad_bias_, _, rs_out = ceg.general_gemm(
                    inputmat_total,
                    grad_output,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    out_dtype=(
                        main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
                    ),
                    bias=(bias if (grad_bias is None and not ctx.fp8) else None),
                    out=main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    ub=ub_obj_wgrad,
                    ub_type=ub_type_wgrad,
                    extra_output=rs_out,
                    bulk_overlap=ctx.ub_bulk_wgrad,
                )
                
        """
        # 检查output中是否存在-1e10，如果存在则说明有错误
        # if torch.any(output == -1e10):
        #     raise ValueError("general_gemm returned an invalid value")
    sync_device(device)
    end_time = time.time()
    dur = (end_time - start_time) / (test_steps - warmup_steps)
    return dur

def plot_topk(ops_info, topk, save_path):
    plt.cla()
    plt.pie(ops_info['percentage'].head(topk), labels=ops_info['Module'].head(topk), autopct='%1.1f%%')
    plt.savefig(save_path)
    plt.show()

def test_gemm_efficency(gemm_shape_list, max_tflops, device, save_root, dtype, res, test_steps, warmup_steps):
    # 遍历df每一行，拿到b, m n k， 测试gemm shape的效率，记录在efficency中 
    efficiency_file_path = f'{save_root}/gemm_efficency.json'
    matmul_key = 'matmul' if dtype == 'bf16' else 'fp8_matmul'
    if os.path.exists(efficiency_file_path):
        all_efficiency = json.load(open(efficiency_file_path))
    else:
        all_efficiency = {
            'matmul': {
                'tflops': max_tflops,
                'efficient_factor': 0,
                'accurate_efficient_factor':{
                }
            },
            'fp8_matmul': {
                'tflops': max_tflops,
                'efficient_factor': 0,
                'accurate_efficient_factor':{
                }
            }
        }
    accurate_efficiency:dict = all_efficiency[matmul_key]['accurate_efficient_factor']

    # Assuming df is a pandas DataFrame with columns 'b', 'm', 'n', 'k'
    for index, row in gemm_shape_list.iterrows():
        b = row['B']  # batch size
        m = row['M']  # rows of matrix A
        k = row['K']  # columns of matrix A / rows of matrix B
        n = row['N']  # columns of matrix B
        layout = row['layout']
        accumulate = row['accumulate']
        out_dtype = row['out_dtype']
        
        # Key for the efficiency dictionary - you can use a tuple as the key
        shape_key = f'b={b}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}'
        
        # Here you would perform your GEMM operation and measure its efficiency
        # This is a placeholder - replace with actual measurement code
        # For example, you might time the operation or measure FLOPs
        if accurate_efficiency.get(shape_key, None) is None:
            # Pseudocode for measurement:
            # Perform GEMM operation with shape (b, m, k, n)
            # A, B_matrix, workspace = prepare_tensors(b, m, k, n, dtype, device)
            execution_time = run_te_gemm(b, m, k, n,dtype,
                                         device=device,
                                         layout = layout,
                                         accumulate=accumulate,
                                         out_dtype = out_dtype,
                                         test_steps=test_steps,
                                         warmup_steps=warmup_steps)
            
            # Calculate efficiency (this depends on your metric - could be GFLOPS, time, etc.)
            # For example, theoretical FLOPs for GEMM: 2 * b * m * n * k
            flops = 2 * b * m * n * k
            tflops = flops / (execution_time * 1e12) if execution_time > 0 else 0
            
            # Store the efficiency metric
            accurate_efficiency[shape_key] = tflops/max_tflops 

            print(f"Shape {shape_key}: {tflops:.2f} TFLOPS, efficiency: {tflops/max_tflops}, execution_time={execution_time*1000} ms")
            res[shape_key] = {'efficiency': tflops/max_tflops, 'execution_time' : f'{execution_time*1000} ms'}
        else:
            print(f'Shape {shape_key} already exists')

    avg_efficiency = sum(accurate_efficiency.values())/len(accurate_efficiency)
    all_efficiency[matmul_key]['efficient_factor'] = avg_efficiency
    print(f'avg_efficiency={avg_efficiency}')
    
    with open(efficiency_file_path, 'w') as f:
        json.dump(all_efficiency, f, indent=4)

def parse_ops_info(perf_model:PerfLLM, save_root):
    # model_config = perf_model.model_config
    # strategy_config = perf_model.strategy
    op_infos = perf_model.analysis_op_info()
    from pprint import pprint
    df = pd.DataFrame(op_infos['first_stage_chunk'])
    df['percentage'] = df['cost']/df['cost'].sum()
    # stage = 'first_stage_chunk'
    # df.to_csv(f"{save_root}/{model_config.model_type}_mbs_{strategy_config.micro_batch_size}_tp_{strategy_config.tp_size}_op_info_{stage}.csv", index=False)
    # plot_topk(df, 20, f"{save_root}/{model_config.model_type}_mbs_{strategy_config.micro_batch_size}_tp_{strategy_config.tp_size}_op_info_{stage}_topk_op.png")
    # print(df.head(100))
    return df

def test(dtype, grad_reduce_in_bf16):
    system, device, MAX_TFLOPS = get_system_name()
    save_root =  f'{system}_gemm_efficency'
    os.makedirs(save_root, exist_ok=True)
    MODEL_CONFIGS = get_all_test_model_configs()
    MBS_LIST = get_test_mbs_list()
    SEQ_LEN_LIST = get_test_seq_len_list()

    res = {}
    system_config = SystemConfig.init_from_config_file(get_simu_system_config('a100_pcie'))
    strategy_config = StrategyConfig.init_from_format_strings("tp1.ep1.tp1.pp1.mbs1.gbs8.world_size8")
    strategy_config.fp8 = False
    strategy_config.grad_reduce_in_bf16 = grad_reduce_in_bf16
    perf_model = PerfLLM()

    for SEQ_LEN in SEQ_LEN_LIST:
        for model_config in MODEL_CONFIGS:
            tp_list = [1]
            ep_list = [1]
            model_name = model_config.model_name
            if model_config.model_type == 'moe':
                model_config.moe_pad_expert_input_to_capacity = True
                model_config.capacity = 1
                model_config.layer_num = 10
            elif model_config.model_type == 'dense':
                tp_list = get_test_tp_list()

            model_config.padded_vocab_size = True
            model_config.make_vocab_size_divisible_by = 128

            for MBS in MBS_LIST:
                for tp in tp_list:
                    for ep in ep_list:
                        strategy_config.tp_size = tp
                        strategy_config.ep_size = ep
                        strategy_config.pp_size = 1
                        strategy_config.micro_batch_size = MBS  # SET micro_batch_size
                        strategy_config.world_size = 8
                        strategy_config.seq_len = SEQ_LEN       # SET sequence-length
            
                        perf_model.configure(
                            strategy_config=strategy_config,
                            model_config=model_config,
                            system_config=system_config
                        )
                        perf_model.run_estimate()
                        perf_model.analysis()

                        ops_info = parse_ops_info(perf_model, os.path.join(save_root, model_name))
                        test_gemm_efficency(gemm_shape_list = ops_info,
                                            max_tflops = MAX_TFLOPS,
                                            device = device,
                                            save_root = save_root,
                                            dtype = dtype,
                                            res=res,
                                            test_steps = 100, 
                                            warmup_steps = 25)

    with open(os.path.join(save_root, 'all_efficiency_and_duration.json'), 'w') as f:
        json.dump(res, f, indent=4)
        
if __name__ == '__main__':
    # test('fp8', grad_reduce_in_bf16=False)
    # test('fp8', grad_reduce_in_bf16=True)
    test('bf16', grad_reduce_in_bf16=False)
    test('bf16', grad_reduce_in_bf16=True)