// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>

#include <gtest/gtest.h>

namespace ckb = ck_tile::builder;

// This test demonstrates how to specify a convolution kernel using the CK Builder API.
//
// STRATEGY:
// This targets the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 operation template.
// This factory is selected when the algorithm satisfies the
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 concept, which requires:
//   - ThreadBlock specification (block_size, tile_size)
//   - GridwiseXdlGemm parameters (XDL-specific tuning)
//   - TransferABC configuration (memory transfer patterns for A, B, C tensors)
//   - ConvFwdSpecialization and GemmSpecialization
//   - BlockGemm configuration (pipeline version and scheduler)
// Note: we are going to revert to explicitly stating the operation template.
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
        // TODO: This direction should be OK as default, but the factory fails.
        ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
        ckb::GroupConvLayout layout  = ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
        ckb::DataType data_type      = ckb::DataType::FP16;
    };
    // Verify that the signature structure conforms to the signature concept.
    static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // Specify the signature in a constexpr value
    constexpr Signature kSignature{};
    // Verify the signature value is valid
    static_assert(ckb::ValidConvSignature<kSignature>);

    // Define a struct to specify the algorithm.
    // TODO improve CK Builder schema to reduce duplication and simplify.
    struct DefaultAlgorithm
    {
        using ConvSpecial = ckb::ConvFwdSpecialization;
        using GemmSpecial = ckb::GemmSpecialization;
        using PipeVers    = ckb::PipelineVersion;
        using PipeSched   = ckb::PipelineScheduler;

        struct ThreadBlock
        {
            int block_size = 256;
            struct TileSize
            {
                int m = 256;
                int n = 256;
                int k = 32;
            } tile_size;
        } thread_block;

        struct GridwiseGemm
        {
            int ak1            = 8;
            int bk1            = 8;
            int m_per_xdl      = 16;
            int n_per_xdl      = 16;
            int m_xdl_per_wave = 4;
            int n_xdl_per_wave = 4;
        } gridwise_gemm;

        struct TransferABC
        {
            struct TransferAB
            {
                struct BlockTransfer
                {
                    int k0  = 4;
                    int m_n = 256;
                    int k1  = 8;
                } block_transfer;
                struct LdsTransfer
                {
                    int src_vector_dim            = 2;
                    int src_scalar_per_vector     = 8;
                    int lds_dst_scalar_per_vector = 8;
                    bool is_direct_load           = true;
                    bool lds_padding              = false;
                } lds_transfer;
                struct BlockTransferAccessOrder
                {
                    std::array<size_t, 3> order{0, 1, 2};
                } block_transfer_access_order;
                struct SrcAccessOrder
                {
                    std::array<size_t, 3> order{0, 1, 2};
                } src_access_order;
            };
            TransferAB a;
            TransferAB b;
            struct TransferC
            {
                struct ThreadClusterDims
                {
                    int m_block        = 1;
                    int m_wave_per_xdl = 32;
                    int n_block        = 1;
                    int n_wave_per_xdl = 8;
                } thread_cluster_dims;
                struct Epilogue
                {
                    int m_per_wave_per_shuffle = 1;
                    int n_per_wave_per_shuffle = 1;
                    int scalar_per_vector      = 8;
                } epilogue;
            } c;
        } transfer;

        // TODO: Fix CK Builder schema to not require these defaults.
        ConvSpecial fwd_specialization  = ConvSpecial::DEFAULT;
        GemmSpecial gemm_specialization = GemmSpecial::Default;
        struct BlockGemm
        {
            PipeVers pipeline_version = PipeVers::V4;
            PipeSched scheduler       = PipeSched::INTRAWAVE;
        } block_gemm;
    };
    //     // Verify that the signature conforms to the expected descriptor
    // static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // // Specify the signature in a constexpr value
    // constexpr Signature kSignature{};
    // // Verify the signature value is valid
    // static_assert(ckb::ValidConvSignature<kSignature>);
    // Verify that the algorithm conforms to the algorithm concept
    static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

    // Specify the algorithm in a constexpr value
    static constexpr const DefaultAlgorithm kAlgorithm;

    // TODO: Verify the algorithm value is valid.

    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<kSignature, kAlgorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");
}
