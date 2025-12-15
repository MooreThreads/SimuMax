export MAX_TFLOPS=312 # 指定机器的标称算力
export SYS_NAME="A100_PCIE" #指定system名称
export PICE_INTRA_LINK=1 #指定机内卡间是否为PCIE互联
export FC8_MODE=0 #指定机内卡间通信连接是否为FC8模式
export PARAM_FILE="./run_params.json" # 测试超参数，指定测试模型的列表，mbs区间，seq_len区间等, 如果不指定，则使用默认超参。
python test_gemm_efficency.py
python test_grouped_gemm_efficency.py
python test_fa_efficency.py
python combine_efficency.py