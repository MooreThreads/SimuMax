import time
import json
import os
import math
from collections import OrderedDict
import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace, _2X_ACC_FPROP
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from simu_tools.efficency_test.utils import get_system_name, sync_device, get_torch_profiler, get_all_test_model_configs, get_test_seq_len_list, get_test_mbs_list, get_test_ep_list

torch_dtype_mapping = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'torch.bfloat16' : 'bf16',
    'torch.float32' : 'fp32'
}

def run_te_grouped_gemm(A, B, out, m_splits, layout, grad,accumulate, use_split_accumulator, single_output,  device, biases = None, warmup=10, repeat=50):
    profiler =  get_torch_profiler(device)
    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            start_time = time.time()
        _ = general_grouped_gemm(
            A,
            B,
            out,
            layout = layout,
            out_dtype = out[0].dtype,
            workspaces=get_multi_stream_cublas_workspace(),
            m_splits=m_splits,
            D_dtype = None,
            use_bias=False,
            bias = biases,
            gelu = False,
            grad = grad,
            accumulate = accumulate,
            use_split_accumulator = use_split_accumulator,
            single_output = single_output
        )
        profiler.step() if profiler else 1
    sync_device(device)
    end_time = time.time()
    return (end_time - start_time)/(repeat-warmup)*1000    


def run_grouped_gemm_fwd_bwd(ng, M, N, K, device, dtype, MAX_TFLOPS, grad_reduce_in_bf16, save_root):
    # single_out_shape = ( # (M, N) x ng-> (ng * M, N) concat all experts' output
    json_path = os.path.join(save_root, 'grouped_gemm_efficency.json')
    all_efficiency = json.load(open(json_path, "r")) if os.path.exists(json_path) else {
        'group_matmul':{
            'tflops': MAX_TFLOPS,
            'efficient_factor': 0,
            'accurate_efficient_factor':{
            }
        },
        'fp8_group_matmul':{
            'tflops': MAX_TFLOPS,
            'efficient_factor': 0,
            'accurate_efficient_factor':{
            }
        }
    }
    matmul_key = 'fp8_group_matmul' if dtype == 'fp8' else 'group_matmul'
    # multi_x = None
    # multi_w = None
    # multi_w_main_grad = None
    # grad_output = None
    # single_out = None
    # m_splits = None

    fwd_out_dtype = 'bf16'
    main_grad_dtype = 'bf16' if grad_reduce_in_bf16 else 'fp32'
    fwd_shape_str = f'ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True'
    bwd_grad_act_shape_str = f'ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False'
    bwd_grad_weight_shape_str = f'ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False'  

    x_quantizer = Float8Quantizer(
            scale=torch.full([1], 0.1).to(device).squeeze(),
            amax=torch.full([1], 1.0).to(device),
            fp8_dtype=tex.DType.kFloat8E4M3,
                )
    w_quantizer = Float8Quantizer(
            scale=torch.full([1], 0.12).to(device).squeeze(),
            amax=torch.full([1], 1.0).to(device),
            fp8_dtype=tex.DType.kFloat8E5M2,
                )
    def create_tensor():
        multi_x = [torch.randn(M, K, dtype=torch.bfloat16, device=device) for _ in range(ng)]
        multi_w = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(ng)]

        single_out = [torch.randn(int(ng*M), N, dtype=torch.bfloat16, device=device)]

        multi_w_main_grad = [torch.randn(N, K, dtype = (torch.float32 if main_grad_dtype == 'fp32' else torch.bfloat16), device=device) for _ in range(ng)]
        grad_input = [torch.randn(M, N, dtype=torch.bfloat16, device=device) for _ in range(ng)]
        grad_output = [torch.randn(M, K, dtype=torch.bfloat16, device=device) for _ in range(ng)]
        
        m_splits = [M] * ng

        if dtype == 'fp8':
            multi_x = [x_quantizer(x) for x in multi_x]
            multi_w = [w_quantizer(w) for w in multi_w]

        return multi_x, multi_w, multi_w_main_grad, grad_input, grad_output, single_out, m_splits
    if any([all_efficiency[matmul_key]['accurate_efficient_factor'].get(fwd_shape_str, None) is None, 
           all_efficiency[matmul_key]['accurate_efficient_factor'].get(bwd_grad_act_shape_str, None) is None, 
           all_efficiency[matmul_key]['accurate_efficient_factor'].get(bwd_grad_weight_shape_str, None) is None]):
        multi_x, multi_w, multi_w_main_grad, grad_input, grad_output, single_out, m_splits = create_tensor()
    else:
        print(f'skip all!!!!!!!!!!!!')
        return
    # fwd
    # groupgemm layout=TN, A=torch.Size([3072, 5120]) x 20, B=torch.Size([1232, 5120]) x 20, out=torch.Size([24640, 3072]) x 1, single_output=True, accumulate=False, grad=False, gelu=False, out_dtype=torch.bfloat16, D_dtype=None, use_bias=False, use_split_accumulator=False   
    

    if all_efficiency[matmul_key]['accurate_efficient_factor'].get(fwd_shape_str, None) is None:
        fwd_dur = run_te_grouped_gemm(
            A=multi_w[::-1],  # ng x [N, K]
            B=multi_x, # ng x [M, K] -> ng x [M, K] * [K, N] = ng x [M, N]
            out =single_out,
            m_splits=m_splits,
            layout='TN',
            grad=False,
            accumulate=False,
            use_split_accumulator=False,
            single_output=True,
            device=device)
        flops = 2 * M * N * K * ng
        fwd_TFLOPS = flops / (fwd_dur* 1e-3  + 1e-12) / 1e12
        fwd_efficency = fwd_TFLOPS / MAX_TFLOPS
        
        all_efficiency[matmul_key]['accurate_efficient_factor'][fwd_shape_str] = fwd_efficency
        print(f"-- [fwd] Running grouped gemm with ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, device={device}, TFLOPS={fwd_TFLOPS}, dur={fwd_dur} ms, fwd_efficency={fwd_efficency}")
    else:
        print(f'-- [fwd] skip {fwd_shape_str}')
    # bwd_grad_act
    # groupgemm layout=NN, A=torch.Size([3072, 5120]) x 20, B=torch.Size([1232, 3072]) x 20, out=torch.Size([1232, 5120]) x 20, single_output=False, accumulate=False, grad=True, gelu=False, out_dtype=torch.bfloat16, D_dtype=None, use_bias=False, use_split_accumulator=True  
    # bwd_grad_act
    if all_efficiency[matmul_key]['accurate_efficient_factor'].get(bwd_grad_act_shape_str, None) is None:
        if dtype == 'fp8':
            fp8_grad_input = [x_quantizer(g) for g in grad_input]
        bwd_grad_act_dur = run_te_grouped_gemm(
            A = multi_w, # ng x [N, K]
            B = fp8_grad_input if dtype == 'fp8' else grad_input, # ng x [M, N] -> ng x [M, N]* [N , K] = ng x [M, K]
            out = grad_output,
            m_splits=m_splits,
            layout='NN',
            grad=True,
            accumulate=False,
            use_split_accumulator=True,
            single_output=False,
            device=device
        )
        flops = 2 * M * N * K * ng
        bwd_grad_act_TFLOPS = flops / (bwd_grad_act_dur* 1e-3  + 1e-12) / 1e12
        bwd_grad_act_efficency = bwd_grad_act_TFLOPS / MAX_TFLOPS
        all_efficiency[matmul_key]['accurate_efficient_factor'][bwd_grad_act_shape_str] = bwd_grad_act_efficency
        print(f"-- [bwd_grad_act] Running grouped gemm with ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, device={device}, TFLOPS={bwd_grad_act_TFLOPS}, dur={bwd_grad_act_dur} ms, bwd_grad_act_efficency={bwd_grad_act_efficency}")
    else:
        print(f'-- [bwd_grad_act] skip {bwd_grad_act_shape_str}')

    # bwd_grad_weight
    # groupgemm layout=NT, A=torch.Size([1232, 5120]) x 20, B=torch.Size([1232, 3072]) x 20, out=torch.Size([3072, 5120]) x 20, single_output=False, accumulate=True, grad=True, gelu=False, out_dtype=torch.bfloat16, D_dtype=None, use_bias=False, use_split_accumulator=True  
    
    if all_efficiency[matmul_key]['accurate_efficient_factor'].get(bwd_grad_weight_shape_str, None) is None:
        if dtype == 'fp8':
            fp8_grad_input2 = [x_quantizer(g) for g in grad_input]
            biases = [torch.tensor([], dtype=torch.bfloat16, device=device) for _ in range(ng)]
        else:
            biases = None
        bwd_grad_w_dur = run_te_grouped_gemm(
            A= multi_x, # ng x [M, K]
            B= fp8_grad_input2 if dtype == 'fp8' else grad_input, # ng x [M, N] -> ng x [N, M] * ng x [M, K] = ng x [N, K]
            out = multi_w_main_grad,
            m_splits=m_splits,
            layout='NT',
            grad=True,
            accumulate=True,
            use_split_accumulator=True,
            single_output=False,
            device=device,
            biases = biases
        )
        flops = 2 * M * N * K * ng  
        bwd_grad_w_TFLOPS = flops / (bwd_grad_w_dur* 1e-3  + 1e-12) / 1e12
        bwd_grad_w_efficency = bwd_grad_w_TFLOPS / MAX_TFLOPS
        all_efficiency[matmul_key]['accurate_efficient_factor'][bwd_grad_weight_shape_str] = bwd_grad_w_efficency
        print(f"-- [bwd_grad_w] Running grouped gemm with ng={ng}, M={M}, N={N}, K={K}, dtype={dtype}, out_dtype={fwd_out_dtype}, main_grad_dtype={main_grad_dtype}, device={device}, TFLOPS={bwd_grad_w_TFLOPS}, dur={bwd_grad_w_dur} ms, bwd_grad_w_efficency={bwd_grad_w_efficency}")
    else:
        print(f"-- [bwd_grad_w] Skip  skip {bwd_grad_weight_shape_str}")

    all_efficiency[matmul_key]['efficient_factor'] = sum(all_efficiency[matmul_key]['accurate_efficient_factor'].values()) / len(all_efficiency[matmul_key]['accurate_efficient_factor'])

    json.dump(all_efficiency, open(f'{save_root}/grouped_gemm_efficency.json', 'w'), indent=4)

def test_grouped_gemm_efficiency(batch, seq_len, num_experts, ep_size, topk, capacity, hidden_size, moe_intermediate_size, device, MAX_TFLOPS, dtype, grad_reduce_in_bf16, save_root, model):
    if num_experts % ep_size != 0:
        return 
    num_local_expert =  num_experts // ep_size
    balance_tokens_per_expert = math.ceil(batch * seq_len * topk / num_experts) * capacity
    num_tokens_per_local_expert = int(balance_tokens_per_expert  * num_experts // num_local_expert)
    print(f"=== [{model}] Run test for batch={batch}, seq_len={seq_len}, ep_size={ep_size}, num_experts={num_experts}, topk={topk}, capacity={capacity}, hidden_size={hidden_size}, moe_intermediate_size={moe_intermediate_size}, Linear1 device={device}, dtype={dtype}, grad_reduce_in_bf16={grad_reduce_in_bf16}")
    run_grouped_gemm_fwd_bwd(
        ng = num_local_expert,
        M = num_tokens_per_local_expert,
        K = hidden_size,
        N = moe_intermediate_size * 2,
        device=device,
        dtype=dtype,
        MAX_TFLOPS=MAX_TFLOPS,
        grad_reduce_in_bf16 = grad_reduce_in_bf16,
        save_root = save_root
    )
    print(f"=== [{model}] Run test for batch={batch}, seq_len={seq_len}, ep_size={ep_size}, num_experts={num_experts}, topk={topk}, capacity={capacity}, hidden_size={hidden_size}, moe_intermediate_size={moe_intermediate_size}, Linear2 device={device}, dtype={dtype}, grad_reduce_in_bf16={grad_reduce_in_bf16}")
    run_grouped_gemm_fwd_bwd(
        ng = num_local_expert,
        M = num_tokens_per_local_expert,
        K = moe_intermediate_size,
        N = hidden_size,
        device=device,
        dtype=dtype,
        MAX_TFLOPS=MAX_TFLOPS,
        grad_reduce_in_bf16 = grad_reduce_in_bf16,
        save_root = save_root
    )

def test(dtype, grad_reduce_in_bf16):
    system, device, MAX_TFLOPS = get_system_name()
    save_root =  f'{system}_grouped_gemm_efficency'
    os.makedirs(save_root, exist_ok=True)
    MODEL_CONFIGS = get_all_test_model_configs()
    MBS_LIST = get_test_mbs_list()
    SEQ_LEN_LIST = get_test_seq_len_list()
    EP_LIST = get_test_ep_list()
    # tp_list = [1]

    for SEQ_LEN in SEQ_LEN_LIST:
        for MBS in MBS_LIST:
            for ep_size in EP_LIST:
                # for tp_size in tp_list:
                    # tp_size
                    # SEQ_LEN = SEQ_LEN // tp_size
                    for model_config in MODEL_CONFIGS:   
                        if model_config.expert_num > 1:
                            test_grouped_gemm_efficiency(batch = MBS, 
                                seq_len = SEQ_LEN, 
                                num_experts = model_config.expert_num, 
                                ep_size = ep_size, 
                                topk = model_config.topk, 
                                capacity = 1, 
                                hidden_size = model_config.hidden_size, 
                                moe_intermediate_size = model_config.moe_ffn_hidden_size, 
                                device = device, 
                                dtype = dtype,
                                MAX_TFLOPS = MAX_TFLOPS,
                                grad_reduce_in_bf16 = grad_reduce_in_bf16,
                                save_root = save_root,
                                model = model_config.model_name)
if __name__ == "__main__":
    # test('fp8', False)
    # test('fp8', True)
    test('bf16', False)
    test('bf16', True)

    