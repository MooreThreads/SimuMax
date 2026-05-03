import json
import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
import transformer_engine.pytorch as te
from simu_tools.efficency_test.utils import get_system_name, get_efficiency_save_root, sync_device, get_torch_profiler, get_all_test_model_configs, get_test_seq_len_list, get_test_mbs_list, get_test_tp_list

CUDA_SDP_IMPL = os.environ.get("CUDA_SDP_IMPL", "te").strip().lower()
FA_TIMING_MODE = os.environ.get("FA_TIMING_MODE", "trace_kernel").strip().lower()
FA_TRACE_WARMUP = int(os.environ.get("FA_TRACE_WARMUP", "5"))
FA_TRACE_ACTIVE = int(os.environ.get("FA_TRACE_ACTIVE", "20"))
FA_KEEP_TRACE = bool(int(os.environ.get("FA_KEEP_TRACE", "0")))
FA_TRACE_DROP_FIRST = int(os.environ.get("FA_TRACE_DROP_FIRST", "1"))
FA_TRACE_REDUCE = os.environ.get("FA_TRACE_REDUCE", "median").strip().lower()
OVERWRITE_EFFICIENCY = bool(int(os.environ.get("EFFICIENCY_OVERWRITE", "0")))
FA_MEASURED_THIS_RUN = set()
FA_EXISTING_KEYS = {"sdp_fwd": set(), "sdp_bwd": set()}


def _append_megatron_paths():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    for env_name in ("MEGATRON_HOME_OVERRIDE", "MEGATRON_HOME"):
        env_path = os.environ.get(env_name)
        if env_path:
            candidates.append(Path(env_path))
            if not Path(env_path).is_absolute():
                candidates.append((repo_root / env_path).resolve())
    candidates.extend(
        [
            repo_root / "simu_tools/megatron_scripts/Megatron-LM-v0.14.0",
            repo_root / "simu_tools/megatron_scripts/Megatron-LM",
        ]
    )
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_append_megatron_paths()

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_TE = True
except Exception as exc:
    HAVE_MEGATRON_TE = False
    TEDotProductAttention = None
    AttnMaskType = None
    TransformerConfig = None
    print(f"[warn] Megatron TEDotProductAttention import failed: {exc}")

# 1. 构造测试输入数据
def generate_test_inputs(device, batch_size, seq_len,  num_q_heads, num_kv_heads, k_head_dim, v_head_dim, qkv_contiguous):
    # batch_size, seq_len, num_heads, head_dim = 4096, 1, 4, 128
    # 生成随机输入张量 (使用BF16格式以测试FlashAttention)
    if device == "musa":
        if qkv_contiguous:
            query = torch.randn(
                (batch_size, num_q_heads, seq_len, k_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            key = torch.randn(
                (batch_size, num_kv_heads, seq_len, k_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            value = torch.randn(
                (batch_size, num_kv_heads, seq_len, v_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            query = torch.randn(
                (batch_size, seq_len, num_q_heads, k_head_dim),
                dtype=torch.bfloat16,
                device=device,
            ).transpose(1, 2)
            key = torch.randn(
                (batch_size, seq_len, num_kv_heads, k_head_dim),
                dtype=torch.bfloat16,
                device=device,
            ).transpose(1, 2)
            value = torch.randn(
                (batch_size, seq_len, num_kv_heads, v_head_dim),
                dtype=torch.bfloat16,
                device=device,
            ).transpose(1, 2)
        return query, key, value

    if device != "cuda":
        raise ValueError(f"Unknown device: {device}")

    if qkv_contiguous:
        query = torch.randn(
            (seq_len, batch_size, num_q_heads, k_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        key = torch.randn(
            (seq_len, batch_size, num_kv_heads, k_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        value = torch.randn(
            (seq_len, batch_size, num_kv_heads, v_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        # Build an actual non-contiguous `sbhd` view. A simple transpose is not
        # enough when batch_size == 1 because the singleton dimension still
        # appears contiguous to PyTorch.
        query = torch.randn(
            (seq_len, batch_size * 2, num_q_heads, k_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[:, ::2, :, :]
        key = torch.randn(
            (seq_len, batch_size * 2, num_kv_heads, k_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[:, ::2, :, :]
        value = torch.randn(
            (seq_len, batch_size * 2, num_kv_heads, v_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[:, ::2, :, :]
    
    return query, key, value


def _reduce_profile_spans_ms(spans_ms, tag_name):
    if not spans_ms:
        raise RuntimeError(f"No kernel spans captured for {tag_name}")
    drop = min(max(0, FA_TRACE_DROP_FIRST), max(0, len(spans_ms) - 1))
    stable = spans_ms[drop:] if drop else spans_ms
    if FA_TRACE_REDUCE == "mean":
        return sum(stable) / len(stable), stable
    if FA_TRACE_REDUCE == "median":
        return median(stable), stable
    raise ValueError(f"Unsupported FA_TRACE_REDUCE={FA_TRACE_REDUCE}")


def _profile_kernel_mean_ms(run_iteration, tag_name, warmup, repeat, device="cuda"):
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}")

    warmup_iters = max(warmup, FA_TRACE_WARMUP)
    active_iters = max(1, min(repeat, FA_TRACE_ACTIVE))

    sync_device(device)
    for _ in range(warmup_iters):
        run_iteration()
        sync_device(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ) as profiler:
        for _ in range(active_iters):
            with torch.profiler.record_function(tag_name):
                run_iteration()
                sync_device(device)

    if FA_KEEP_TRACE:
        trace_dir = Path("./profiler_test/fa_kernel")
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{tag_name}.{int(time.time() * 1000)}.pt.trace.json"
    else:
        fd, tmp_path = tempfile.mkstemp(prefix=f"{tag_name}.", suffix=".pt.trace.json")
        os.close(fd)
        trace_path = Path(tmp_path)

    try:
        profiler.export_chrome_trace(str(trace_path))
        with trace_path.open() as f:
            trace_obj = json.load(f)
    finally:
        if not FA_KEEP_TRACE and trace_path.exists():
            trace_path.unlink()

    annotations = []
    kernel_events = []
    for event in trace_obj.get("traceEvents", []):
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "user_annotation" and event.get("name") == tag_name:
            annotations.append((float(event["ts"]), float(event["ts"]) + float(event["dur"])))
        elif event.get("cat") == "kernel":
            start = float(event["ts"])
            kernel_events.append((start, start + float(event["dur"]), float(event["dur"])))

    if not annotations:
        raise RuntimeError(f"No profiler annotation found for {tag_name}")

    per_iter_kernel_span_us = []
    for ann_start, ann_end in annotations:
        ann_kernels = []
        for kernel_start, kernel_end, kernel_dur in kernel_events:
            if kernel_start >= ann_start and kernel_end <= ann_end:
                ann_kernels.append((kernel_start, kernel_end))
        if not ann_kernels:
            continue
        ann_kernels.sort(key=lambda x: x[0])
        per_iter_kernel_span_us.append(ann_kernels[-1][1] - ann_kernels[0][0])

    reduced_ms, stable_ms = _reduce_profile_spans_ms(
        [span_us / 1000.0 for span_us in per_iter_kernel_span_us],
        tag_name,
    )
    print(
        f"[fa_trace_kernel] {tag_name}: all_ms="
        f"{[round(x, 6) for x in [span_us / 1000.0 for span_us in per_iter_kernel_span_us]]}, "
        f"stable_ms={[round(x, 6) for x in stable_ms]}, "
        f"reduce={FA_TRACE_REDUCE}, value_ms={reduced_ms:.6f}"
    )
    return reduced_ms

def benchmark_flashattention_raw_te(q, k, v, warmup=10, repeat=100, device='cuda'):
    if device != "cuda":
        raise ValueError(f"Unknown device: {device}")
    seq_len, batch_size, num_q_heads, qk_head_dim = q.shape
    _, _, num_kv_heads, v_head_dim = v.shape
    if qk_head_dim != v_head_dim:
        raise ValueError(
            f"TransformerEngine DotProductAttention expects qk_head_dim == v_head_dim, got "
            f"{qk_head_dim} vs {v_head_dim}"
        )

    module = te.DotProductAttention(
        num_attention_heads=num_q_heads,
        kv_channels=qk_head_dim,
        num_gqa_groups=num_kv_heads,
        attention_dropout=0.0,
        qkv_format='sbhd',
        attn_mask_type='causal',
        sequence_parallel=False,
        tp_size=1,
        attention_type='self',
    ).to(device=device)
    measured_repeat = max(1, repeat - warmup)
    if FA_TIMING_MODE == "trace_kernel":
        fwd_elapsed_time = _profile_kernel_mean_ms(
            lambda: module(q, k, v, None, qkv_format='sbhd', attn_mask_type='causal'),
            "fa_raw_te_fwd",
            warmup,
            measured_repeat,
        )
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = module(qq, kk, vv, None, qkv_format='sbhd', attn_mask_type='causal')
        grad = torch.randn_like(out)

        def run_bwd():
            qq.grad = None
            kk.grad = None
            vv.grad = None
            out.backward(grad, retain_graph=True)

        bwd_elapsed_time = _profile_kernel_mean_ms(
            run_bwd,
            "fa_raw_te_bwd",
            warmup,
            measured_repeat,
        )
        return fwd_elapsed_time, bwd_elapsed_time

    profiler = get_torch_profiler(device, False)

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            fwd_start_time = time.time()
        out = module(q, k, v, None, qkv_format='sbhd', attn_mask_type='causal')
        profiler.step() if profiler else 1

    sync_device(device)
    fwd_elapsed_time = (time.time() - fwd_start_time) / (repeat - warmup) * 1000

    grad = torch.randn_like(out)
    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            bwd_start_time = time.time()
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = module(qq, kk, vv, None, qkv_format='sbhd', attn_mask_type='causal')
        out.backward(grad)

    sync_device(device)
    bwd_elapsed_time = (time.time() - bwd_start_time) / (repeat - warmup) * 1000
    profiler.stop() if profiler else 1
    return fwd_elapsed_time, bwd_elapsed_time


def benchmark_flashattention_flash_attn(q, k, v, warmup=10, repeat=100, device='cuda'):
    if device != "cuda":
        raise ValueError(f"Unknown device: {device}")
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    measured_repeat = max(1, repeat - warmup)
    if FA_TIMING_MODE == "trace_kernel":
        def run_fwd():
            return _flash_attn_forward(
                q,
                k,
                v,
                0.0,
                0.1,
                True,
                -1,
                0,
                0.0,
                None,
                False,
            )

        fwd_elapsed_time = _profile_kernel_mean_ms(
            run_fwd,
            "fa_flash_attn_fwd",
            warmup,
            measured_repeat,
        )
        out, softmax_lse, S_dmask, rng_state = run_fwd()

        def run_bwd():
            _flash_attn_backward(
                out,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                0.0,
                1.0,
                True,
                -1,
                0,
                0.0,
                None,
                False,
                rng_state,
            )

        bwd_elapsed_time = _profile_kernel_mean_ms(
            run_bwd,
            "fa_flash_attn_bwd",
            warmup,
            measured_repeat,
        )
        return fwd_elapsed_time, bwd_elapsed_time

    profiler = get_torch_profiler(device, False)

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            fwd_start_time = time.time()
        out, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            0.0,
            0.1,
            True,
            -1,
            0,
            0.0,
            None,
            False,
        )
        profiler.step() if profiler else 1

    sync_device(device)
    fwd_elapsed_time = (time.time() - fwd_start_time) / (repeat - warmup) * 1000

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            bwd_start_time = time.time()
        _flash_attn_backward(
            out,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            0.0,
            1.0,
            True,
            -1,
            0,
            0.0,
            None,
            False,
            rng_state,
        )

    sync_device(device)
    bwd_elapsed_time = (time.time() - bwd_start_time) / (repeat - warmup) * 1000
    profiler.stop() if profiler else 1
    return fwd_elapsed_time, bwd_elapsed_time


def benchmark_flashattention_megatron_te(q, k, v, warmup=10, repeat=100, device='cuda'):
    if device != "cuda":
        raise ValueError(f"Unknown device: {device}")
    if not HAVE_MEGATRON_TE:
        raise RuntimeError("Megatron TEDotProductAttention is unavailable.")

    seq_len, batch_size, num_q_heads, qk_head_dim = q.shape
    _, _, num_kv_heads, v_head_dim = v.shape

    config = TransformerConfig(
        num_layers=1,
        hidden_size=num_q_heads * v_head_dim,
        num_attention_heads=num_q_heads,
        num_query_groups=num_kv_heads,
        kv_channels=qk_head_dim,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        apply_query_key_layer_scaling=False,
    )
    module = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        k_channels=qk_head_dim,
        v_channels=v_head_dim,
    ).to(device=device)

    measured_repeat = max(1, repeat - warmup)
    if FA_TIMING_MODE == "trace_kernel":
        fwd_elapsed_time = _profile_kernel_mean_ms(
            lambda: module(q, k, v, None, AttnMaskType.causal),
            "fa_megatron_te_fwd",
            warmup,
            measured_repeat,
        )
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = module(qq, kk, vv, None, AttnMaskType.causal)
        grad = torch.randn_like(out)

        def run_bwd():
            qq.grad = None
            kk.grad = None
            vv.grad = None
            out.backward(grad, retain_graph=True)

        bwd_elapsed_time = _profile_kernel_mean_ms(
            run_bwd,
            "fa_megatron_te_bwd",
            warmup,
            measured_repeat,
        )
        return fwd_elapsed_time, bwd_elapsed_time

    profiler = get_torch_profiler(device, False)

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            fwd_start_time = time.time()
        out = module(q, k, v, None, AttnMaskType.causal)
        profiler.step() if profiler else 1

    sync_device(device)
    fwd_elapsed_time = (time.time() - fwd_start_time) / (repeat - warmup) * 1000

    grad = torch.randn_like(out)
    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            bwd_start_time = time.time()
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = module(qq, kk, vv, None, AttnMaskType.causal)
        out.backward(grad)

    sync_device(device)
    bwd_elapsed_time = (time.time() - bwd_start_time) / (repeat - warmup) * 1000
    profiler.stop() if profiler else 1
    return fwd_elapsed_time, bwd_elapsed_time


def benchmark_flashattention_musa(q, k, v, warmup=10, repeat=100, device='musa'):
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    profiler = get_torch_profiler(device, False)

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            fwd_start_time = time.time()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            attn_output = torch.ops.aten._scaled_dot_product_attention_flash_musa(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
        profiler.step() if profiler else 1
    sync_device(device)
    fwd_elapsed_time = (time.time() - fwd_start_time) / (repeat - warmup) * 1000

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            bwd_start_time = time.time()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            torch.ops.aten._scaled_dot_product_attention_flash_musa_backward(
                attn_output[0],
                q,
                k,
                v,
                *attn_output,
                is_causal=True,
            )
        profiler.step() if profiler else 1
    sync_device(device)
    bwd_elapsed_time = (time.time() - bwd_start_time) / (repeat - warmup) * 1000
    profiler.stop() if profiler else 1
    return fwd_elapsed_time, bwd_elapsed_time


# 4. 基准测试函数
def benchmark_flashattention(q, k, v, warmup=10, repeat=100, device='cuda'):
    if device == "musa":
        return benchmark_flashattention_musa(q, k, v, warmup=warmup, repeat=repeat, device=device)
    if CUDA_SDP_IMPL == "te":
        if HAVE_MEGATRON_TE:
            return benchmark_flashattention_megatron_te(
                q, k, v, warmup=warmup, repeat=repeat, device=device
            )
        if q.shape[-1] == v.shape[-1]:
            print("[warn] Falling back to raw TE benchmark because Megatron TE path is unavailable.")
            return benchmark_flashattention_raw_te(
                q, k, v, warmup=warmup, repeat=repeat, device=device
            )
        raise RuntimeError(
            "Megatron TE path is unavailable and raw TE cannot cover MLA shapes "
            f"(qk_head_dim={q.shape[-1]}, v_head_dim={v.shape[-1]})."
        )
    if CUDA_SDP_IMPL in {"raw_te", "te_raw"}:
        return benchmark_flashattention_raw_te(q, k, v, warmup=warmup, repeat=repeat, device=device)
    if CUDA_SDP_IMPL in {"flash", "flash_attn"}:
        return benchmark_flashattention_flash_attn(q, k, v, warmup=warmup, repeat=repeat, device=device)
    raise ValueError(f"Unsupported CUDA_SDP_IMPL={CUDA_SDP_IMPL}")

def test(model, all_efficiency, qkv_contiguous, batch, seq_len, num_q_heads, num_kv_heads,  qk_head_dim, v_head_dim, TP=1, MAX_TFLOPS=None, res=None, device='cuda'):
    assert num_q_heads % TP == 0 and num_kv_heads % TP == 0
    num_q_heads //= TP
    num_kv_heads //= TP
    shape_key = f'batch={batch}, seq_len={seq_len}, head_num={num_q_heads}, kv_head_num={num_kv_heads}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}'
    existing_fwd_keys = FA_EXISTING_KEYS['sdp_fwd']
    existing_bwd_keys = FA_EXISTING_KEYS['sdp_bwd']

    if shape_key in FA_MEASURED_THIS_RUN:
        return
    if (
        shape_key in all_efficiency['sdp_fwd']['accurate_efficient_factor']
        and shape_key in all_efficiency['sdp_bwd']['accurate_efficient_factor']
        and (
            (not OVERWRITE_EFFICIENCY)
            or (
                shape_key not in existing_fwd_keys
                and shape_key not in existing_bwd_keys
            )
        )
    ):
        return
    query, key, value = generate_test_inputs(device, batch, seq_len,  num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, qkv_contiguous)
    print(f"- 输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    print(
        f'- backend: {CUDA_SDP_IMPL}, batch: {batch}, seq_len: {seq_len}, '
        f'num_heads: {num_q_heads}&{num_kv_heads}, qk_head_dim: {qk_head_dim}, v_head_dim: {v_head_dim}'
    )
    # 初始化配置和模块
    # 运行基准测试
    fwd_latency, bwd_latency = benchmark_flashattention(query, key, value, device = device)
    
    # Keep the FLOPs convention aligned with the simulator-side SDPA modeling.
    base_flops = batch * (seq_len ** 2) * max(num_q_heads, num_kv_heads) * (qk_head_dim + v_head_dim)
    fwd_flops = 2 * base_flops
    fwd_tflops = (fwd_flops / (fwd_latency * 1e-3+ 1e-12)) / 1e12  # 转换为TFLOPs

    bwd_flops = 5 * base_flops
    bwd_tflops = (bwd_flops / (bwd_latency * 1e-3+ 1e-12)) / 1e12  # 转换为TFLOPs
    
    print(f"=== {model} SDPA性能结果(TP={TP}, backend={CUDA_SDP_IMPL}):")
    print(f"- 延迟: {fwd_latency:.3f} ms/iteration, backward: {bwd_latency:.3f} ms/iteration")
    print(f"- fwd吞吐量: {fwd_tflops:.2f} TFLOPs, flops={fwd_flops} latency={fwd_latency:.2f} ms, 计算效率={fwd_tflops/MAX_TFLOPS:.2f}") 
    print(f"- bwd吞吐量: {bwd_tflops:.2f} TFLOPs, flops={bwd_flops} latency={bwd_latency:.2f} ms, 计算效率={bwd_tflops/MAX_TFLOPS:.2f}")
    print(f"- 输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    res['model'].append(model)
    res['TP'].append(TP)
    res['batch'].append(batch)
    res['seq_len'].append(seq_len)
    res['num_heads'].append(f'q={num_q_heads}, kv={num_kv_heads}')
    res['head_size'].append(f'qk={qk_head_dim//TP:.0f}, v={v_head_dim//TP:.0f}')
    res['flops'].append(f'fwd={fwd_flops}, bwd={bwd_flops}')
    res['time'].append(f'fwd={fwd_latency:.2f} ms, bwd={bwd_latency:.2f} ms')
    res['TFLOPS'].append(f'fwd={fwd_tflops:.2f} TFLOPS, bwd={bwd_tflops:.2f} TFLOPS')
    res['fwd_efficiency'].append(round(fwd_tflops/MAX_TFLOPS, 2))
    res['bwd_efficiency'].append(round(bwd_tflops/MAX_TFLOPS, 2))
    
    
    sdp_key = 'sdp_fwd'
    all_efficiency[sdp_key]['accurate_efficient_factor'][shape_key] = fwd_tflops/MAX_TFLOPS
    all_efficiency[sdp_key]['efficient_factor'] = sum(all_efficiency[sdp_key]['accurate_efficient_factor'].values()) / len(all_efficiency[sdp_key]['accurate_efficient_factor'])

    sdp_key = 'sdp_bwd'
    all_efficiency[sdp_key]['accurate_efficient_factor'][shape_key] = bwd_tflops/MAX_TFLOPS
    all_efficiency[sdp_key]['efficient_factor'] = sum(all_efficiency[sdp_key]['accurate_efficient_factor'].values()) / len(all_efficiency[sdp_key]['accurate_efficient_factor'])
    FA_MEASURED_THIS_RUN.add(shape_key)



# 5. 主测试流程
if __name__ == "__main__":
    # 准备输入数据
    system, device, MAX_TFLOPS = get_system_name()
    save_root = get_efficiency_save_root(system, 'fa_efficiency')
    os.makedirs(save_root, exist_ok=True)
    MODEL_CONFIGS = get_all_test_model_configs()
    MBS_LIST = get_test_mbs_list()
    SEQ_LEN_LIST = get_test_seq_len_list()
    TP_LIST = get_test_tp_list()
    res = {
        'model':[],
        'TP':[],
        'batch':[],
        'seq_len':[],
        'num_heads':[],
        'head_size':[],
        'flops':[],
        'TFLOPS':[],
        'time':[],
        'fwd_efficiency':[],
        'bwd_efficiency':[],
        # 'shape_str':[]
    }
    merged_dict = {
                'sdp_fwd':{
                    'tflops': MAX_TFLOPS,
                    'efficient_factor': 0,
                    'accurate_efficient_factor':{
                    }
                },
                'sdp_bwd':{
                    'tflops': MAX_TFLOPS,
                    'efficient_factor': 0,
                    'accurate_efficient_factor':{
                    }
                },
            }
    FA_EXISTING_KEYS['sdp_fwd'] = set(merged_dict['sdp_fwd']['accurate_efficient_factor'].keys())
    FA_EXISTING_KEYS['sdp_bwd'] = set(merged_dict['sdp_bwd']['accurate_efficient_factor'].keys())
    for SEQ_LEN in SEQ_LEN_LIST:
        for MBS in MBS_LIST:
            for model_config in MODEL_CONFIGS:
                for tp in TP_LIST:
                    model = model_config.model_name
                    print(f"Running {model}...")
                    if model_config.qk_head_dim is not None:
                        qk_head_dim = model_config.qk_head_dim + (model_config.qk_pos_emb_head_dim if model_config.qk_pos_emb_head_dim is not None else 0)
                    else:
                        qk_head_dim = model_config.head_size
                    v_head_dim = model_config.v_head_dim if model_config.v_head_dim is not None else model_config.head_size
                    
                    params = dict(batch=MBS, 
                                  seq_len=SEQ_LEN, 
                                  num_q_heads = model_config.head_num, 
                                  num_kv_heads = model_config.kv_head_num, 
                                  qk_head_dim = qk_head_dim, 
                                  v_head_dim = v_head_dim, 
                                  TP = tp, 
                                  MAX_TFLOPS = MAX_TFLOPS, 
                                  res = res, 
                                  device = device)
                    test(model, merged_dict, True, **params)
                    if device == 'cuda':
                        test(model, merged_dict, False, **params)
        
    import pandas as pd
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(save_root, 'fa_efficiency_test.csv'), index=False)
    with open(os.path.join(save_root, 'fa_efficiency_test.json'), 'w') as f:
        json.dump(merged_dict, f, indent=4)
