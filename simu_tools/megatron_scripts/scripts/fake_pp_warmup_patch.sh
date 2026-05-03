#!/bin/bash

fake_pp_warmup_patch_marker() {
    echo "MEGATRON_FAKE_PP_SIZE_FOR_WARMUP"
}

fake_pp_warmup_patch_is_applied() {
    local megatron_home=$1
    grep -q "$(fake_pp_warmup_patch_marker)" \
        "${megatron_home}/megatron/core/pipeline_parallel/schedules.py"
}

apply_fake_pp_warmup_patch() {
    local megatron_home=$1
    local patch_file=$2
    patch --quiet -p1 -d "${megatron_home}" < "${patch_file}"
}

revert_fake_pp_warmup_patch() {
    local megatron_home=$1
    local patch_file=$2
    patch --quiet -R -p1 -d "${megatron_home}" < "${patch_file}"
}
