#!/bin/bash
LOCK_FILE="/tmp/$(basename "$0").lock"
exec 200>"$LOCK_FILE"
flock -n 200 || {
    echo "等待前一个实例完成..."
    flock 200
    echo "开始执行..."
}

tp_size=1
world_size=8
micro_batch_size=$1
num_microbatches=$2
ep_size=$3
pp_size=$4
model_type=$5
test_type=$6
num_layer=$7
dtype=${8:-"bf16"}
(( dp_size = $world_size / ($tp_size * $pp_size) ))
(( global_batch_size = $micro_batch_size * $num_microbatches * $dp_size ))

# echo -e "\033[32mdp_size: $dp_size, pp_size: $pp_size, ep_size: $ep_size, \
# world_size: $world_size, micro_batch_size: $micro_batch_size, \
# num_microbatches: $num_microbatches, global_batch_size: $global_batch_size, model_type=$model_type, est_type=$test_type, num_layer=$num_layer, dtype=$dtype\033[0m"
current_time=$(date "+%Y%m%d_%H%M%S")
output_dir="./output_${model_type}/$current_time" 
mkdir -p ${output_dir}

set -u
  work_home=$(realpath ./)
  patch_home=""
  megatron_home="./Megatron-LM"
  example="tp${tp_size}_pp${pp_size}_dp${dp_size}_mbs${micro_batch_size}_mbc${num_microbatches}_gbs${global_batch_size}_gpus${world_size}"
  data_path=${data_path:-"/home/dist/yehua/llama2_dataset/llama_00_text_document"}
  hostfile=./hostfile
  log_file=${output_dir}/$example.log
  script_file=./scripts/run_pretrain_deepseekv2.sh
  rdzv_id=$current_time
  master_port=12345
set +u

cmd_env=(
    "WORK_HOME=${work_home}"
    "PATCH_HOME=${patch_home}"
    "MEGATRON_HOME=${megatron_home}"
    "EXAMPLE=${example}"
    "HOSTFILE=${hostfile}"
    "TP_SIZE=${tp_size}"
    "PP_SIZE=${pp_size}"
    "EP_SIZE=${ep_size}"
    "MICRO_BATCH_SIZE=${micro_batch_size}"
    "GLOBAL_BATCH_SIZE=${global_batch_size}"
    "RDZV_ID=${rdzv_id}"
    "MASTER_PORT=${master_port}"
    "TEST_TYPE=${test_type}"
    "NUM_LAYER=${num_layer}"
    "MODEL_TYPE=${model_type}"
    "DTYPE=${dtype}"
    "OUTPUT_DIR=${output_dir}"
)

# 执行
cmd="cd ${work_home} && ${cmd_env[@]} bash ${script_file}"

hostlist=$(grep -v '^#\|^$' $hostfile | awk '{print $1}' | xargs)

# Check if hostlist is empty
if [ -z "$hostlist" ]; then
  echo "Error: hostlist is empty. Please add IP addresses to the hostfile."
  exit 1
fi

for host in ${hostlist[@]}; do
  # cmd_ssh="$cmd > ${output_dir}/$host.log 2>&1"
  cmd_ssh="$cmd > ${output_dir}/$host.log"
  echo $cmd_ssh
  ssh -n $host $cmd_ssh &
  # eval $cmd_ssh
done

wait
bash stop_all.sh 