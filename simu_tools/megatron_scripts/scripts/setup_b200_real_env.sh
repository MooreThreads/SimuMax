#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
work_root="${repo_root}/simu_tools/megatron_scripts"
megatron_home="${work_root}/Megatron-LM"
patch_file="${repo_root}/simu_tools/megatron_scripts/patches/megatron_fake_pp_warmup.patch"
memory_patch_file="${repo_root}/simu_tools/megatron_scripts/patches/megatron_b200_memory_logging.patch"
megatron_repo="https://github.com/NVIDIA/Megatron-LM.git"
megatron_commit="23e00ed0963c35382dfe8a5a94fb3cda4d21e133"
expected_te="2.11.0+c188b533"
expected_torch="2.10.0a0+a36e1d39eb.nv26.01.42222806"
apply_fake_pp_patch="${APPLY_FAKE_PP_PATCH:-0}"
apply_memory_log_patch="${APPLY_MEMORY_LOG_PATCH:-1}"
sync_megatron_remote="${SYNC_MEGATRON_REMOTE:-0}"

if [[ -x /usr/local/cuda-13.1/bin/ptxas ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.1/bin/ptxas}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
fi

if [[ -d "${CUDA_HOME}/include" ]]; then
  export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
  export C_INCLUDE_PATH="${CUDA_HOME}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
if [[ -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

printf '== Runtime check ==\n'
python - <<'PY'
import platform, sys
import torch
import transformer_engine
print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('te:', transformer_engine.__version__)
print('platform:', platform.platform())
PY

printf '\n== CUDA/Triton env ==\n'
printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
printf 'TRITON_PTXAS_PATH=%s\n' "${TRITON_PTXAS_PATH}"
printf 'CPATH=%s\n' "${CPATH:-}"

mkdir -p "${work_root}"
if [[ ! -d "${megatron_home}/.git" ]]; then
  printf '\n== Cloning Megatron-LM ==\n'
  git clone "${megatron_repo}" "${megatron_home}"
fi

printf '\n== Sync Megatron-LM checkout ==\n'
if [[ "${sync_megatron_remote}" == "1" ]]; then
  git -C "${megatron_home}" fetch origin --tags --prune
elif ! git -C "${megatron_home}" cat-file -e "${megatron_commit}^{commit}" 2>/dev/null; then
  git -C "${megatron_home}" fetch origin --tags --prune
fi
git -C "${megatron_home}" checkout "${megatron_commit}"
actual_commit="$(git -C "${megatron_home}" rev-parse HEAD)"
printf 'Megatron-LM commit: %s\n' "${actual_commit}"

if [[ "${apply_memory_log_patch}" == "1" ]]; then
  printf '\n== Applying B200 memory logging patch ==\n'
  if patch --dry-run -p1 -d "${megatron_home}" < "${memory_patch_file}" >/dev/null 2>&1; then
    patch --quiet -p1 -d "${megatron_home}" < "${memory_patch_file}"
    printf 'Applied: %s\n' "${memory_patch_file}"
  else
    printf 'Patch already applied or not cleanly applicable: %s\n' "${memory_patch_file}"
  fi
else
  printf '\n== B200 memory logging patch ==\n'
  printf 'Skipped. Set APPLY_MEMORY_LOG_PATCH=1 to apply:\n'
  printf '  %s\n' "${memory_patch_file}"
fi

if [[ "${apply_fake_pp_patch}" == "1" ]]; then
  printf '\n== Applying fake-PP warmup patch ==\n'
  if patch --dry-run -p1 -d "${megatron_home}" < "${patch_file}" >/dev/null 2>&1; then
    patch --quiet -p1 -d "${megatron_home}" < "${patch_file}"
    printf 'Applied: %s\n' "${patch_file}"
  else
    printf 'Patch already applied or not cleanly applicable: %s\n' "${patch_file}"
  fi
else
  printf '\n== Fake-PP patch ==\n'
  printf 'Skipped by default. Set APPLY_FAKE_PP_PATCH=1 to apply:\n'
  printf '  %s\n' "${patch_file}"
fi

printf '\n== Recommended next steps ==\n'
printf '1. Verify TE version matches retained baseline: %s\n' "${expected_te}"
printf '2. Verify Torch version matches retained baseline: %s\n' "${expected_torch}"
printf '3. Set SYNC_MEGATRON_REMOTE=1 if you want this script to refresh the remote checkout.\n'
printf '4. Set APPLY_MEMORY_LOG_PATCH=0 only if you intentionally do not need formal memory reproduction.\n'
printf '5. Run canonical launcher scripts from: %s\n' "${work_root}"
printf '6. Read reproduction guide: %s\n' "${repo_root}/docs/b200/b200_real_repro_guide.md"
