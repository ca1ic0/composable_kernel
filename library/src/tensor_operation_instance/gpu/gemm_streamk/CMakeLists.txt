# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
add_instance_library(device_gemm_streamk_instance
   device_gemm_xdl_streamk_f16_f16_f16_mk_kn_mn_instance.cpp
)
