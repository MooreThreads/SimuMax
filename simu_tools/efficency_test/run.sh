SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MAX_TFLOPS=312 # target device peak TFLOPS
export SYS_NAME="my_system" # output system name
# export NUM_PER_NODE=8 # optional override; leave unset to use visible device count
# export MEM_GBS=80 # optional override; leave unset to use detected device memory
export PICE_INTRA_LINK=0 # 0: non-PCIe intra-node interconnect, 1: PCIe intra-node interconnect
export FC8_MODE=0 # whether the intra-node link behaves like FC8
export PARAM_FILE="${SCRIPT_DIR}/run_params.json" # 测试超参数，指定测试模型的列表，mbs区间，seq_len区间等, 如果不指定，则使用默认超参。
python "${SCRIPT_DIR}/test_gemm_efficiency.py"
python "${SCRIPT_DIR}/test_grouped_gemm_efficiency.py"
python "${SCRIPT_DIR}/test_fa_efficiency.py"
python "${SCRIPT_DIR}/combine_efficiency.py"
