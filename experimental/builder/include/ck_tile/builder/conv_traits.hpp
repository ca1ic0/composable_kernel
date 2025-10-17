// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/instance_traits.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>

namespace ck_tile::reflect {

// Helper structures for organizing trait data with domain-specific naming
struct ConvBlock
{
    int block_size;
    struct
    {
        int m;
        int n;
        int k;
    } per_block;
};

struct ConvTuning
{
    int ak1;
    int bk1;
    int m_per_xdl;
    int n_per_dxl;
    int m_xdl_per_wave;
    int n_xdl_per_wave;
};

struct BlockTransfer
{
    ck::Array<int, 3> thread_cluster_dims; // k0, m/n, k1
    int src_vector_dim;
    int src_scalar_per_vector;
    int dst_scalar_per_vector_k1;
    int lds_extra;
};

struct CBlockTransfer
{
    int m_xdl_per_wave_per_shuffle;
    int n_xdl_per_wave_per_shuffle;
    ck::Array<int, 4> thread_cluster_dims; // m_block, m_wave_per_xdl, n_block, n_wave_per_xdl
    int scalar_per_vector;
};

// Helper metafunctions to derive signature information from Instance types

// Derive ConvDirection from device kernel type
template <typename Instance>
constexpr builder::ConvDirection conv_direction()
{
    using InstTraits = InstanceTraits<Instance>;
    
    // Check if conv_forward_specialization exists
    if constexpr (requires { &InstTraits::conv_forward_specialization; }) {
        return builder::ConvDirection::FORWARD;
    }
    // Check if conv_bwd_data_specialization exists  
    else if constexpr (requires { &InstTraits::conv_bwd_data_specialization; }) {
        return builder::ConvDirection::BACKWARD_DATA;
    }
    else {
        return builder::ConvDirection::FORWARD; // Default fallback
    }
}

// Derive GroupConvLayout from layout types
template <typename Instance>
constexpr builder::GroupConvLayout conv_layout()
{
    using InstTraits = InstanceTraits<Instance>;
    using ALayout = typename InstTraits::a_layout;
    
    namespace ctc = ck::tensor_layout::convolution;
    
    // Check for channels-last layouts
    if constexpr(std::is_same_v<ALayout, ctc::NHWGC> ||
                 std::is_same_v<ALayout, ctc::NDHWGC> ||
                 std::is_same_v<ALayout, ctc::GNHWC> ||
                 std::is_same_v<ALayout, ctc::GNDHWC> ||
                 std::is_same_v<ALayout, ctc::NWGC> ||
                 std::is_same_v<ALayout, ctc::GNWC>)
    {
        return builder::GroupConvLayout::CHANNELS_LAST;
    }
    // Check for channels-first layouts
    else if constexpr(std::is_same_v<ALayout, ctc::NGCHW> ||
                      std::is_same_v<ALayout, ctc::NGCDHW> ||
                      std::is_same_v<ALayout, ctc::GNCW>)
    {
        return builder::GroupConvLayout::CHANNELS_FIRST;
    }
    else
    {
        // Default fallback
        return builder::GroupConvLayout::CHANNELS_LAST;
    }
}

// Derive DataType from data type
template <typename Instance>
constexpr builder::DataType conv_data_type()
{
    using InstTraits = InstanceTraits<Instance>;
    using ADataType = typename InstTraits::a_data_type;
    
    if constexpr(std::is_same_v<ADataType, ck::half_t>)
    {
        return builder::DataType::FP16;
    }
    else if constexpr(std::is_same_v<ADataType, ck::bhalf_t>)
    {
        return builder::DataType::BF16;
    }
    else if constexpr(std::is_same_v<ADataType, float>)
    {
        return builder::DataType::FP32;
    }
    else if constexpr(std::is_same_v<ADataType, double>)
    {
        return builder::DataType::FP64;
    }
    else if constexpr(std::is_same_v<ADataType, int8_t>)
    {
        return builder::DataType::S8;
    }
    else
    {
        // Default fallback
        return builder::DataType::FP32;
    }
}

// Helper to extract values from Sequence types at compile time
template <typename Seq, ck::index_t Idx>
struct SequenceAt;

template <ck::index_t... Is, ck::index_t Idx>
struct SequenceAt<ck::Sequence<Is...>, Idx>
{
    static constexpr int value = ck::Sequence<Is...>::At(Idx);
};

// Primary template for ConvTraits
template <typename T>
struct ConvTraits;

// Specialization 1: Direct from Instance (Primary use case)
template <typename Instance>
    requires requires { typename InstanceTraits<Instance>; }
struct ConvTraits<Instance>
{
    using InstTraits = InstanceTraits<Instance>;

    // Signature information (derived from Instance template parameters)
    static constexpr int spatial_dim = InstTraits::spatial_dim;
    static constexpr builder::ConvDirection direction = conv_direction<Instance>();
    static constexpr builder::GroupConvLayout layout = conv_layout<Instance>();
    static constexpr builder::DataType data_type = conv_data_type<Instance>();

    // Algorithm information (extracted from Instance template parameters)
    static constexpr ConvBlock block = {
        .block_size = InstTraits::block_size,
        .per_block = {
            .m = InstTraits::m_per_block,
            .n = InstTraits::n_per_block,
            .k = InstTraits::k_per_block
        }
    };

    static constexpr ConvTuning tuning = {
        .ak1 = InstTraits::ak1,
        .bk1 = InstTraits::bk1,
        .m_per_xdl = InstTraits::m_per_xdl,
        .n_per_dxl = InstTraits::n_per_xdl,
        .m_xdl_per_wave = InstTraits::m_xdl_per_wave,
        .n_xdl_per_wave = InstTraits::n_xdl_per_wave
    };

    static constexpr BlockTransfer a_block_transfer = {
        .thread_cluster_dims = {
            SequenceAt<typename InstTraits::a_thread_cluster_lengths, 0>::value,
            SequenceAt<typename InstTraits::a_thread_cluster_lengths, 1>::value,
            SequenceAt<typename InstTraits::a_thread_cluster_lengths, 2>::value
        },
        .src_vector_dim = InstTraits::a_block_transfer_src_vector_dim,
        .src_scalar_per_vector = InstTraits::a_block_transfer_src_scalar_per_vector,
        .dst_scalar_per_vector_k1 = InstTraits::a_block_transfer_dst_scalar_per_vector_k1,
        .lds_extra = InstTraits::a_block_lds_extra_m
    };

    static constexpr BlockTransfer b_block_transfer = {
        .thread_cluster_dims = {
            SequenceAt<typename InstTraits::b_thread_cluster_lengths, 0>::value,
            SequenceAt<typename InstTraits::b_thread_cluster_lengths, 1>::value,
            SequenceAt<typename InstTraits::b_thread_cluster_lengths, 2>::value
        },
        .src_vector_dim = InstTraits::b_block_transfer_src_vector_dim,
        .src_scalar_per_vector = InstTraits::b_block_transfer_src_scalar_per_vector,
        .dst_scalar_per_vector_k1 = InstTraits::b_block_transfer_dst_scalar_per_vector_k1,
        .lds_extra = InstTraits::b_block_lds_extra_n
    };

    static constexpr CBlockTransfer c_block_transfer = {
        .m_xdl_per_wave_per_shuffle = InstTraits::c_shuffle_m_xdl_per_wave_per_shuffle,
        .n_xdl_per_wave_per_shuffle = InstTraits::c_shuffle_n_xdl_per_wave_per_shuffle,
        .thread_cluster_dims = {
            SequenceAt<typename InstTraits::c_thread_cluster_lengths, 0>::value,
            SequenceAt<typename InstTraits::c_thread_cluster_lengths, 1>::value,
            SequenceAt<typename InstTraits::c_thread_cluster_lengths, 2>::value,
            SequenceAt<typename InstTraits::c_thread_cluster_lengths, 3>::value
        },
        .scalar_per_vector = InstTraits::c_block_transfer_scalar_per_vector
    };

    // Pipeline version (only available for forward convolutions)
    // For backward data, this member doesn't exist in InstanceTraits
    template <typename T = InstTraits>
    static constexpr auto get_pipeline_version() {
        if constexpr (requires { T::pipeline_version; }) {
            return T::pipeline_version;
        } else {
            // Return a default or indicate not available
            return ck::BlockGemmPipelineVersion::v1;
        }
    }
    static constexpr auto pipeline_version = get_pipeline_version();
};

// Specialization 2: From Builder (Backward compatibility)
template <builder::ConvSignatureDescriptor auto SIGNATURE,
          builder::ConvAlgorithmDescriptor auto ALGORITHM,
          builder::StringLiteral VERSION>
struct ConvTraits<builder::ConvBuilder<SIGNATURE, ALGORITHM, VERSION>>
{
    using Factory = builder::ConvFactory<SIGNATURE, ALGORITHM, VERSION>;
    using Instance = typename Factory::Instance;
    
    // Delegate to Instance-based ConvTraits
    using InstanceConvTraits = ConvTraits<Instance>;
    
    // Forward all members from Instance-based traits
    static constexpr int spatial_dim = InstanceConvTraits::spatial_dim;
    static constexpr builder::ConvDirection direction = InstanceConvTraits::direction;
    static constexpr builder::GroupConvLayout layout = InstanceConvTraits::layout;
    static constexpr builder::DataType data_type = InstanceConvTraits::data_type;
    
    static constexpr auto block = InstanceConvTraits::block;
    static constexpr auto tuning = InstanceConvTraits::tuning;
    static constexpr auto a_block_transfer = InstanceConvTraits::a_block_transfer;
    static constexpr auto b_block_transfer = InstanceConvTraits::b_block_transfer;
    static constexpr auto c_block_transfer = InstanceConvTraits::c_block_transfer;
    static constexpr auto pipeline_version = InstanceConvTraits::pipeline_version;
};

} // namespace ck_tile::reflect
