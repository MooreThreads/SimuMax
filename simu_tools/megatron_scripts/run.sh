#!/bin/bash

mbc_list=(4 8 32)

# 遍历每个 mbc 值
for mbc in "${mbc_list[@]}"; do
    bash run_deepseekv2.sh 1 $mbc 8 1 deepseek_v2 test 4
    bash run_deepseekv2.sh 1 $mbc 4 2 deepseek_v2 test 4
    bash run_llama3.sh 1 $mbc 1 2 llama3_8b test 32 
    bash run_llama3.sh 1 $mbc 2 1 llama3_8b test 32 
    bash run_llama3.sh 1 $mbc 4 1 llama3_8b test 32 
    bash run_llama3.sh 1 $mbc 8 1 llama3_8b test 32 

    bash run_llama3.sh 1 $mbc 1 2 llama3_70b test 12 
    bash run_llama3.sh 1 $mbc 2 1 llama3_70b test 12 
    bash run_llama3.sh 1 $mbc 4 1 llama3_70b test 12 
    bash run_llama3.sh 1 $mbc 8 1 llama3_70b test 12 

    # bash run_deepseekv2.sh 1 $mbc 8 1 deepseek_v2 test 4 fp8
done
