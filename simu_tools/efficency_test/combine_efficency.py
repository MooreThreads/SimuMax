import os
import json
from simu_tools.efficency_test.utils import get_system_name

system, device, MAX_TFLOPS = get_system_name()
gemm_save_root =  f'{system}_gemm_efficency/gemm_efficency.json'
groupgemm_save_root =  f'{system}_grouped_gemm_efficency/grouped_gemm_efficency.json'
fa_save_root =  f'{system}_fa_efficency/fa_efficency_test.json'

gemm_ops = json.load(open(gemm_save_root, 'r'))
groupgemm_ops = json.load(open(groupgemm_save_root, 'r'))
fa_ops = json.load(open(fa_save_root, 'r'))

sys_name = os.environ.get('SYS_NAME', 'A100_pcie')
intra_with_pcie = int(os.environ.get('PICE_INTRA_LINK', '0'))
FC8 = bool(int(os.environ.get('FC8_MODE', '0')))

networks = {
        "intra_with_pcie": False,
        "low_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 1,
                "gbps": 21,
                "latency_us": 130
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1
                }
            }
        },
        "high_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 50,
                "latency_us": 35,
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0,
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                }
            }
        },
        "inter_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 1,
                "gbps": 200,
                "latency_us": 35
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                }
            }
        }
    }
pcie_networks = {
        "intra_with_pcie":True,
        "intra_node_pcie_8x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.4176,
                    "latency_us": 8.98
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.4497, 
                    "latency_us": 4.83,
                    "dp_fixed_bw": {
                        "2": 4,
                        "4": 7.65
                    }
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.3688,
                    "latency_us": 15.64,
                    "dp_fixed_bw":{
                        "2": 3.9,
                        "4": 7
                    } 
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.2754,
                    "latency_us": 30.04
                }
            }
        },
        "intra_node_pcie_4x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.6924,
                    "latency_us": 3.45
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6915,
                    "latency_us": 8.93
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6845,
                    "latency_us": 29.88
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6380,
                    "latency_us": 13.84
                }
            }

        },
        "intra_node_pcie_2x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.6965,
                    "latency_us": 15.51
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6866,
                    "latency_us": 24.84
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6419,
                    "latency_us": 131.30
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6969,
                    "latency_us": 24.07
                }
            }

        },
        "inter_node": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.6,
                "gbps": 200,
                "latency_us": 130
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1
                }
            }
        }
    }

system_template = {
    "sys_name": sys_name,
    "num_per_node": 8,
    "accelerator": {
        "backend": "cuda",
        "mem_gbs": 80,
        "op" : {

        },
        "bandwidth": {
            "default": {
                "efficient_factor": 0.666,
                "gbps":   1600,
                "latency_us": 30
            },
            "permute_fwd":{
                "efficient_factor": 0.46,
                "gbps": 1600,
                "latency_us": 30
            },
            "permute_bwd":{
                "efficient_factor": 0.2175,
                "gbps": 1600,
                "latency_us": 30
            },
            "ce":{
                "efficient_factor": 0.73,
                "gbps": 1600,
                "latency_us": 30
            }
        },
        "mode": "roofline"
    },
    "FC8":FC8,
}
if intra_with_pcie:
    system_template['networks'] = pcie_networks
else:
    system_template['networks'] = networks

system_template["accelerator"]['op'].update(gemm_ops)
system_template["accelerator"]['op'].update(groupgemm_ops)
system_template["accelerator"]['op'].update(fa_ops)

bf16_efficient_factor = [v['efficient_factor'] for k, v in system_template["accelerator"]['op'].items() if 'fp8' not in k and 'sdp' not in k]
avg_efficient_factor = sum(bf16_efficient_factor) / len(bf16_efficient_factor)
system_template["accelerator"]['op']['default'] = {
    'tflops' : MAX_TFLOPS,
    'efficient_factor': avg_efficient_factor,
}
json.dump(system_template, open(f"{sys_name}.json", "w"), indent=4)