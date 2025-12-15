import os
import time
from tabulate import tabulate
import importlib.util
spec = importlib.util.find_spec("simumax")
root = os.path.dirname(spec.submodule_search_locations[0])

def init_config_map(root):
    def get_config_files(config_dir):
        config_files = {}
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                if file.endswith('.json'):
                    config_files[file[:-5]] = os.path.join(root, file)
        config_files['root'] = config_dir
        return config_files
    model_config_dir = os.path.join(root, 'models')
    strategy_config_dir = os.path.join(root, 'strategy')
    system_config_dir = os.path.join(root, 'system')    

    models = get_config_files(model_config_dir)
    strategy = get_config_files(strategy_config_dir)
    systems = get_config_files(system_config_dir)   
    return models, strategy, systems

RELEASE_MODELS, RELEASE_STRATEGY, RELEASE_SYSTEM = init_config_map(os.path.join(root, 'configs'))
DEV_MODELS, DEV_STRATEGY, DEV_SYSTEM = init_config_map(os.path.join(root, 'develop/configs'))

def get_config(key, version, r_maps:dict, d_maps:dict, m_type):
    if version == 'release':
        assert key in r_maps.keys(), f"{key} not found, please add {m_type} config in {r_maps['root']}"
        return r_maps[key]
    elif version == 'dev':
        assert key in d_maps.keys(), f"{key} not found, please add {m_type} config in {d_maps['root']}"
        return d_maps[key]
    else:
        raise ValueError('type must be release or dev')
    
def get_simu_model_config(model_name, version='release'):
    return get_config(model_name, version, RELEASE_MODELS, DEV_MODELS, 'model')
    
def get_simu_strategy_config(strategy_name, version='release'):
    return get_config(strategy_name, version, RELEASE_STRATEGY, DEV_STRATEGY, 'strategy')

def get_simu_system_config(system_name, version='release'):
    return get_config(system_name, version, RELEASE_SYSTEM, DEV_SYSTEM, 'system')    

def show_dict(maps:dict, headers):
    print(tabulate([[k, v] for i, (k, v) in enumerate(maps.items()) if k != 'root'], headers=headers))

def show_simu_models(version='release'):
    headers = ['Model', 'Model Config Path']
    if version == 'release':
        show_dict(RELEASE_MODELS, headers)
    elif version == 'dev':  
        show_dict(DEV_MODELS, headers)
    else:
        raise ValueError('type must be release or dev')
    
def show_simu_strategy(version='release'):
    headers = ['Strategy', 'Strategy Config Path']
    if version == 'release':    
        show_dict(RELEASE_STRATEGY, headers)
    elif version == 'dev':  
        show_dict(DEV_STRATEGY, headers)
    else:
        raise ValueError('type must be release or dev')

def show_simu_system(version='release'):
    headers = ['System', 'System Config Path']
    if version == 'release':
        show_dict(RELEASE_SYSTEM, headers)
    elif version == 'dev':  
        show_dict(DEV_SYSTEM, headers)
    else:
        raise ValueError('type must be release or dev')


def parse_parallel_config(s):
    parts = s.split('_')
    config = {}
    for part in parts:
        if part.startswith('ep'):
            config['ep_size'] = int(part[2:])
        elif part.startswith('tp'):
            config['tp_size'] = int(part[2:])
        elif part.startswith('pp'):
            config['pp_size'] = int(part[2:])
    return config

def create_default_strategy(strs="", seq_len=4096, micro_batch_size=1, micro_batch_num=81, dtype='bf16'):
    default_config = {
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "micro_batch_num": micro_batch_num,
        "dtype": "bf16",
        "world_size": 8,
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "etp_size": 1,
        "moe_dispatcher_policy": "all2all",
        "enable_sequence_parallel": True,
        "interleaving_size": 1,
        "zero_state": 1,
        "enable_dropout": False,
        "use_fused_norm": True,
        "use_math_sdp": False,
        "use_flash_sdp": True,
        "use_fp32_accum_grad": True,
        "enable_recompute": True,
        "mem_factor": 0.94,
        "fp8" : True if dtype == 'fp8' else False,
    }
    default_config.update(parse_parallel_config(strs))
    return default_config


    