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
