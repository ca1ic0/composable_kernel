// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include "impl/conv_signature_types.hpp"
#include "impl/conv_algorithm_types.hpp"

#include <gtest/gtest.h>

namespace ckb = ck_tile::builder;

// This test demonstrates how to specify a convolution kernel using the CK Builder API.
//
// STRATEGY:
// We are targeting the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 factory specialization.
// This factory is selected when the algorithm satisfies the
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 concept, which requires:
//   - ThreadBlock specification (block_size, tile_size)
//   - GridwiseXdlGemm parameters (XDL-specific tuning)
//   - TransferABC configuration (memory transfer patterns for A, B, C tensors)
//   - ConvFwdSpecialization and GemmSpecialization
//   - BlockGemm configuration (pipeline version and scheduler)
//
// The signature specifies:
//   - 2D spatial convolution
//   - GNHWC_GKYXC_GNHWK layout (channels-last for input/output)
//   - FP16 data type
//   - Forward direction (default when not specified)
//
// This combination will instantiate:
//   ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
TEST(BuilderExample, SimpleConvolutionExample)
{
    // Define a struct to specify the signature
    struct Signature
    {
        int spatial_dim = 2;
        // TOOD: This direction should be OK as defualt, but the factory fails.
        ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
        ckb::GroupConvLayout layout  = ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
        ckb::DataType data_type      = ckb::DataType::FP16;
    };
    // Verify that the signature conforms to the expected descriptor
    static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // Specify the signature in a constexpr value
    constexpr Signature kSignature{};
    // Verify the signature value is valid
    static_assert(ckb::ValidConvSignature<kSignature>);

    // Define the algorithm specification using the same structure as test_conv_description.cpp
    // TODO: We should be able to build this struct without the ckb::test helpers.
    struct DefaultAlgorithm
    {
        ckb::test::ThreadBlock thread_block{.block_size = 256,
                                            .tile_size  = {.m = 256, .n = 256, .k = 32}};

        ckb::test::GridwiseXdlGemm gridwise_gemm{.ak1            = 8,
                                                 .bk1            = 8,
                                                 .m_per_xdl      = 16,
                                                 .n_per_xdl      = 16,
                                                 .m_xdl_per_wave = 4,
                                                 .n_xdl_per_wave = 4};

        ckb::test::TransferABC transfer{
            .a =
                {
                    .block_transfer              = {.k0 = 4, .m_n = 256, .k1 = 8},
                    .lds_transfer                = {.src_vector_dim            = 2,
                                                    .src_scalar_per_vector     = 8,
                                                    .lds_dst_scalar_per_vector = 8,
                                                    .is_direct_load            = true,
                                                    .lds_padding               = false},
                    .block_transfer_access_order = {.order = {0, 1, 2}},
                    .src_access_order            = {.order = {0, 1, 2}},

                },
            .b =
                {
                    .block_transfer              = {.k0 = 4, .m_n = 256, .k1 = 8},
                    .lds_transfer                = {.src_vector_dim            = 2,
                                                    .src_scalar_per_vector     = 8,
                                                    .lds_dst_scalar_per_vector = 8,
                                                    .is_direct_load            = true,
                                                    .lds_padding               = false},
                    .block_transfer_access_order = {.order = {0, 1, 2}},
                    .src_access_order            = {.order = {0, 1, 2}},
                },
            .c =
                {
                    .thread_cluster_dims =
                        {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
                    .epilogue = {.m_per_wave_per_shuffle = 1,
                                 .n_per_wave_per_shuffle = 1,
                                 .scalar_per_vector      = 8},
                },
        };

        ckb::ConvFwdSpecialization fwd_specialization = ckb::ConvFwdSpecialization::DEFAULT;
        ckb::GemmSpecialization gemm_specialization   = ckb::GemmSpecialization::Default;
        ckb::test::BlockGemm block_gemm{.pipeline_version = ckb::PipelineVersion::V4,
                                        .scheduler        = ckb::PipelineScheduler::INTRAWAVE};
    };
    static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

    // Verify the algorithm satisfies the V3 concept
    // TODO: This looks wrong users shouldn't need to do this.
    static_assert(ckb::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<DefaultAlgorithm>,
                  "Algorithm must satisfy DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 concept");

    // Create constexpr instances for use with ConvBuilder
    static constexpr const DefaultAlgorithm kAlgorithm;

    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<kSignature, kAlgorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");
}
