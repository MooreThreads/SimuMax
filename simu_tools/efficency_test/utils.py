import os
import time
import json
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
DEFAULT_TP_LIST = [1, 2, 4, 8]
DEFAULT_EP_LIST = [1, 2, 4, 8, 16, 64]

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
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, skip_first=2,repeat=100),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            with_modules=True,
        )
    return profiler

def sync_device(device=None):
    torch.cuda.synchronize()

def get_system_name():
    system = None
    device = None
    MAX_TFLOPS = None
    
    try:
        system = torch.cuda.get_device_name(0)
        device = 'cuda'
        if 'A100' in system:
            MAX_TFLOPS = 312
        elif 'H100' in system:
            MAX_TFLOPS = 1979
    except Exception as e:
        print(f"[ERROR] Info: {e}") 
        exit()
    
    if os.environ.get('MAX_TFLOPS', None) is not None:
        MAX_TFLOPS = float(os.environ.get('MAX_TFLOPS'))
    
    assert MAX_TFLOPS is not None, f"MAX_TFLOPS is None, exit!"
    assert device is not None, f"device is None, exit!"
    
    print(f'System: {system}, Device: {device}, Max TFLOPS: {MAX_TFLOPS}')
    return system, device, MAX_TFLOPS