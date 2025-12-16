// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, g, c, hi, wi] * wei[g, k, c, y, x] = out[n, g, k, ho, wo]
void add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NGCHW,
                                                           GKCYX,
                                                           NGKHW,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f32_merge_instances<
            2,
            NGCHW,
            GKCYX,
            NGKHW,
            ConvBwdWeightDefault,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v1>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f32_merge_instances<
            2,
            NGCHW,
            GKCYX,
            NGKHW,
            ConvBwdWeightDefault,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v2>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f32_merge_instances<
            2,
            NGCHW,
            GKCYX,
            NGKHW,
            ConvBwdWeightDefault,
            BlockGemmPipelineScheduler::Interwave,
            BlockGemmPipelineVersion::v1>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f32_merge_instances<
            2,
            NGCHW,
            GKCYX,
            NGKHW,
            ConvBwdWeightDefault,
            BlockGemmPipelineScheduler::Interwave,
            BlockGemmPipelineVersion::v2>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
