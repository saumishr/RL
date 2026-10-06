#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sourced by lightning35_launch.sh through EXTERNAL_VLLM_POOL_OVERRIDES, after
# it has defined the GenRM and NL2Bash pools. Both helpers append, and the
# replica exports env assignments and builds vLLM arguments in order, so these
# values win over the Lightning defaults.

# GenRM runs at TP=8 on 4-GPU nodes, so its tensor-parallel group spans two
# nodes and the allreduce is multi-node. FlashInfer's trtllm backend, which the
# Lightning defaults select, refuses that and names its own replacement:
#   "Flashinfer allreduce is not supported for multi-node allreduce with
#    'trtllm' backend. Please use 'mnnvl' backend instead."
external_vllm_pool_env GENRM \
  "VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl"

# vLLM's BF16x3 router GEMM warmup and its CuteDSL warmup both import
# cutlass.base_dsl, which the nvidia-cutlass-dsl build in this container does
# not provide, so engine startup dies with
#   "No module named 'cutlass.base_dsl.enums'".
# Only the sm100 branch of kernel_warmup() reaches the router GEMM, so this is
# specific to running the judges on Blackwell. Warmup only moves JIT cost off
# the first request, so skipping it costs judge warm-up latency and nothing
# else; drop these once the container's cutlass-dsl skew is fixed.
_nt4_kernel_config='{"enable_jit_warmup": false, "enable_cutedsl_warmup": false}'
external_vllm_pool_args GENRM --kernel-config "${_nt4_kernel_config}"
external_vllm_pool_args NL2BASH --kernel-config "${_nt4_kernel_config}"
unset _nt4_kernel_config

# Most of a judge's startup is its "initial profiling run", and most of that is
# FlashInfer building JIT modules. FlashInfer derives its cache from
# FLASHINFER_WORKSPACE_BASE, which Lightning pins to /tmp, so every replica
# rebuilds the same modules from scratch on every run. Point each pool at a
# per-pool directory on shared storage so the builds survive across jobs. The
# first run still pays for them; later runs reuse them, which is what keeps the
# judges inside the idle-GPU budget. Per-pool paths keep GenRM and NL2Bash from
# contending over the same build locks.
for _nt4_pool in GENRM NL2BASH; do
  external_vllm_pool_env "${_nt4_pool}" \
    "FLASHINFER_WORKSPACE_BASE=${PERSISTENT_CACHE}/judges/${_nt4_pool,,}" \
    "VLLM_CACHE_ROOT=${PERSISTENT_CACHE}/judges/${_nt4_pool,,}/vllm"
  mkdir -p "${PERSISTENT_CACHE}/judges/${_nt4_pool,,}/vllm"
done
unset _nt4_pool
