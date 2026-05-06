import os
import time
import json
import math
import torch

from simumax.utils import get_simu_model_config, RELEASE_MODELS
from simumax.core.config import ModelConfig

DEFAULT_MODEL_LIST = ["deepseekv2",
                  "deepseekv3",
                  "deepseek-32b",
                  "deepseek-16b",
                  "deepseek-1b",
                  "llama3-8b",
                  "llama3-70b",
                  "qwen3-32b",
                  "kimi-1T",
                  "mixtral-8x7b"]
DEFAULT_MBS_LIST = [1, 2, 4]
DEFAULT_SEQ_LEN_LIST = [4096]
DEFAULT_CP_LIST = [1, 2, 4, 8]
DEFAULT_TP_LIST = [1, 2, 4, 8]
DEFAULT_EP_LIST = [1, 2, 4, 8, 16, 64]
DEFAULT_DTYPE_LIST = ["bf16"]
DEFAULT_EXTRA_VOCAB_SIZE_LIST = [102400, 102401]

PARAM_FILE = os.environ.get("PARAM_FILE", None)
GLOBAL_PARAMS = None
if PARAM_FILE:
    try:
        GLOBAL_PARAMS = json.load(open(PARAM_FILE))
    except Exception as e:
        print(e)


def get_test_seq_len_list():
    if GLOBAL_PARAMS is not None and "seq_len_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["seq_len_list"]
    else:
        return DEFAULT_SEQ_LEN_LIST

def get_test_mbs_list():
    if GLOBAL_PARAMS is not None and "mbs_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["mbs_list"]
    else:
        return DEFAULT_MBS_LIST

def get_test_ep_list():
    if GLOBAL_PARAMS is not None and "ep_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["ep_list"]
    else:
        return DEFAULT_EP_LIST

def get_test_tp_list():
    if GLOBAL_PARAMS is not None and "tp_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["tp_list"]
    else:
        return DEFAULT_TP_LIST

def get_test_cp_list():
    if GLOBAL_PARAMS is not None and "cp_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["cp_list"]
    else:
        return DEFAULT_CP_LIST

def get_dtype_list():
    if GLOBAL_PARAMS is not None and "dtype" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["dtype"]
    else:
        return DEFAULT_DTYPE_LIST

def get_extra_vocab_size_list():
    if GLOBAL_PARAMS is not None and "extra_vocab_size_list" in GLOBAL_PARAMS:
        return GLOBAL_PARAMS["extra_vocab_size_list"]
    else:
        return DEFAULT_EXTRA_VOCAB_SIZE_LIST
def get_all_test_model_configs():
    if GLOBAL_PARAMS is not None and "model_list" in GLOBAL_PARAMS:
        model_list =  GLOBAL_PARAMS["model_list"]
    else:
        model_list =  DEFAULT_MODEL_LIST
    
    configs = []
    for model in model_list:
        if model in RELEASE_MODELS:
            configs.append(ModelConfig.init_from_config_file(get_simu_model_config(model)))
        else:
            print(f"ERROR: Given model_name {model} not found in simumax release model list(path:{RELEASE_MODELS['root']}), skip first!")
    return configs

def get_torch_profiler(device, profile = False):
    def trace_handler(prof):
        rank = 0
        curr_trace_dir_name = "iteration_" + str(prof.step_num)
        curr_trace_dir = os.path.join('./profiler_test', curr_trace_dir_name)
        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)
        curr_trace_path = os.path.join(curr_trace_dir, f"rank{rank}.{int(time.time()*1000)}.pt.trace.json")
        print(f"Dumping profiler traces at step {prof.step_num} to {curr_trace_path}")
        begin = time.monotonic()
        prof.export_chrome_trace(curr_trace_path)
        print(
            f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
    )
    if not profile:
        return None
    profiler =  torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MUSA if device=='musa' else torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, skip_first=2,repeat=100),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            with_modules=True,
        )
    return profiler

def sync_device(device):
    torch.musa.synchronize() if device=='musa' else torch.cuda.synchronize()

def get_system_name():
    system = None
    device = None
    MAX_TFLOPS = None
    
    try:
        system = torch.cuda.get_device_name(0)
        device = 'cuda'
        if 'A100' in system:
            MAX_TFLOPS = 312
            system_name = 'a100'
        elif 'H100' in system:
            MAX_TFLOPS = 1979
            system_name = 'h100'
        elif 'B200' in system:
            MAX_TFLOPS = 2250
            system_name = 'b200'
        else:
            raise ValueError("Unsupported device")
    except:
        pass

    try:
        system = torch.musa.get_device_name(0)
        device = 'musa'
        MAX_TFLOPS = None
    except:
        pass
    # assert system is not None and device is not None and MAX_TFLOPS is not None, "Unsupported device"
    if os.environ.get('MAX_TFLOPS', None) is not None:
        MAX_TFLOPS = float(os.environ.get('MAX_TFLOPS'))

    if system is None or device is None or MAX_TFLOPS is None:
        raise RuntimeError(
            "Unsupported device for efficiency measurement. "
            "Detected system=%r, device=%r, MAX_TFLOPS=%r. "
            "Use a supported accelerator or set MAX_TFLOPS explicitly for a detected CUDA/MUSA device."
            % (system, device, MAX_TFLOPS)
        )

    print(f'System: {system}, Device: {device}, Max TFLOPS: {MAX_TFLOPS}')
    return system, device, MAX_TFLOPS


def get_efficiency_save_root(system_name: str, suffix: str) -> str:
    root_dir = os.path.dirname(__file__)
    cache_tag = os.environ.get("EFFICIENCY_CACHE_TAG", "").strip()
    if cache_tag:
        return os.path.join(root_dir, f"{system_name}_{cache_tag}_{suffix}")
    return os.path.join(root_dir, f"{system_name}_{suffix}")


def get_system_runtime_info():
    system, device, max_tflops = get_system_name()

    if device == "cuda":
        device_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
    elif device == "musa":
        device_count = torch.musa.device_count()
        props = torch.musa.get_device_properties(0)
    else:
        raise RuntimeError(f"Unsupported runtime device {device!r}")

    total_memory = getattr(props, "total_memory", None)
    mem_gbs = None if total_memory is None else math.ceil(total_memory / (1024**3))

    if os.environ.get("NUM_PER_NODE") is not None:
        device_count = int(os.environ["NUM_PER_NODE"])
    if os.environ.get("MEM_GBS") is not None:
        mem_gbs = int(os.environ["MEM_GBS"])

    return {
        "system": system,
        "device": device,
        "max_tflops": max_tflops,
        "num_per_node": device_count,
        "mem_gbs": mem_gbs,
    }
