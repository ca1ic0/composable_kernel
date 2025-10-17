// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp>

namespace ck_tile::reflect {

// Primary template for InstanceTraits - extracts compile-time information directly from
// device kernel instances (e.g., DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3)
template <typename Instance>
struct InstanceTraits;

// Specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AComputeDataType,
          typename BComputeDataType>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
    NDimSpatial,
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    DsDataType,
    EDataType,
    AElementwiseOperation,
    BElementwiseOperation,
    CDEElementwiseOperation,
    ConvForwardSpecialization,
    GemmSpec,
    BlockSize,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    AK1,
    BK1,
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_AK1,
    ABlockLdsExtraM,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_BK1,
    BBlockLdsExtraN,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CDEBlockTransferScalarPerVector_NPerBlock,
    BlkGemmPipeSched,
    BlkGemmPipelineVer,
    AComputeDataType,
    BComputeDataType>>
{
    // Spatial dimension
    static constexpr int spatial_dim = NDimSpatial;

    // Layout types
    using a_layout = ALayout;
    using b_layout = BLayout;
    using e_layout = ELayout;

    // Data types
    using a_data_type = ADataType;
    using b_data_type = BDataType;
    using acc_data_type = AccDataType;
    using e_data_type = EDataType;

    // Block configuration
    static constexpr int block_size = BlockSize;
    static constexpr int m_per_block = MPerBlock;
    static constexpr int n_per_block = NPerBlock;
    static constexpr int k_per_block = KPerBlock;

    // Tuning parameters
    static constexpr int ak1 = AK1;
    static constexpr int bk1 = BK1;
    static constexpr int m_per_xdl = MPerXDL;
    static constexpr int n_per_xdl = NPerXDL;
    static constexpr int m_xdl_per_wave = MXdlPerWave;
    static constexpr int n_xdl_per_wave = NXdlPerWave;

    // A block transfer thread cluster dimensions
    using a_thread_cluster_lengths = ABlockTransferThreadClusterLengths_AK0_M_AK1;
    static constexpr int a_block_transfer_src_vector_dim = ABlockTransferSrcVectorDim;
    static constexpr int a_block_transfer_src_scalar_per_vector = ABlockTransferSrcScalarPerVector;
    static constexpr int a_block_transfer_dst_scalar_per_vector_k1 =
        ABlockTransferDstScalarPerVector_AK1;
    static constexpr int a_block_lds_extra_m = ABlockLdsExtraM;

    // B block transfer thread cluster dimensions
    using b_thread_cluster_lengths = BBlockTransferThreadClusterLengths_BK0_N_BK1;
    static constexpr int b_block_transfer_src_vector_dim = BBlockTransferSrcVectorDim;
    static constexpr int b_block_transfer_src_scalar_per_vector = BBlockTransferSrcScalarPerVector;
    static constexpr int b_block_transfer_dst_scalar_per_vector_k1 =
        BBlockTransferDstScalarPerVector_BK1;
    static constexpr int b_block_lds_extra_n = BBlockLdsExtraN;

    // C shuffle parameters
    static constexpr int c_shuffle_m_xdl_per_wave_per_shuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr int c_shuffle_n_xdl_per_wave_per_shuffle = CShuffleNXdlPerWavePerShuffle;
    using c_thread_cluster_lengths = CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
    static constexpr int c_block_transfer_scalar_per_vector = CDEBlockTransferScalarPerVector_NPerBlock;

    // Pipeline configuration
    static constexpr ck::BlockGemmPipelineScheduler pipeline_scheduler = BlkGemmPipeSched;
    static constexpr ck::BlockGemmPipelineVersion pipeline_version = BlkGemmPipelineVer;

    // Specialization
    static constexpr auto conv_forward_specialization = ConvForwardSpecialization;
    static constexpr auto gemm_specialization = GemmSpec;
};

// Specialization for DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization,
          bool DoPadGemmM,
          bool DoPadGemmN,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
    NDimSpatial,
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    DsDataType,
    EDataType,
    AElementwiseOperation,
    BElementwiseOperation,
    CDEElementwiseOperation,
    ConvBwdDataSpecialization,
    DoPadGemmM,
    DoPadGemmN,
    NumGemmKPrefetchStage,
    BlockSize,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    AK1,
    BK1,
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_AK1,
    ABlockLdsExtraM,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_BK1,
    BBlockLdsExtraN,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CDEBlockTransferScalarPerVector_NPerBlock>>
{
    // Spatial dimension
    static constexpr int spatial_dim = NDimSpatial;

    // Layout types
    using a_layout = ALayout;
    using b_layout = BLayout;
    using e_layout = ELayout;

    // Data types
    using a_data_type = ADataType;
    using b_data_type = BDataType;
    using acc_data_type = AccDataType;
    using e_data_type = EDataType;

    // Block configuration
    static constexpr int block_size = BlockSize;
    static constexpr int m_per_block = MPerBlock;
    static constexpr int n_per_block = NPerBlock;
    static constexpr int k_per_block = KPerBlock;

    // Tuning parameters
    static constexpr int ak1 = AK1;
    static constexpr int bk1 = BK1;
    static constexpr int m_per_xdl = MPerXDL;
    static constexpr int n_per_xdl = NPerXDL;
    static constexpr int m_xdl_per_wave = MXdlPerWave;
    static constexpr int n_xdl_per_wave = NXdlPerWave;

    // A block transfer thread cluster dimensions
    using a_thread_cluster_lengths = ABlockTransferThreadClusterLengths_AK0_M_AK1;
    static constexpr int a_block_transfer_src_vector_dim = ABlockTransferSrcVectorDim;
    static constexpr int a_block_transfer_src_scalar_per_vector = ABlockTransferSrcScalarPerVector;
    static constexpr int a_block_transfer_dst_scalar_per_vector_k1 =
        ABlockTransferDstScalarPerVector_AK1;
    static constexpr int a_block_lds_extra_m = ABlockLdsExtraM;

    // B block transfer thread cluster dimensions
    using b_thread_cluster_lengths = BBlockTransferThreadClusterLengths_BK0_N_BK1;
    static constexpr int b_block_transfer_src_vector_dim = BBlockTransferSrcVectorDim;
    static constexpr int b_block_transfer_src_scalar_per_vector = BBlockTransferSrcScalarPerVector;
    static constexpr int b_block_transfer_dst_scalar_per_vector_k1 =
        BBlockTransferDstScalarPerVector_BK1;
    static constexpr int b_block_lds_extra_n = BBlockLdsExtraN;

    // C shuffle parameters
    static constexpr int c_shuffle_m_xdl_per_wave_per_shuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr int c_shuffle_n_xdl_per_wave_per_shuffle = CShuffleNXdlPerWavePerShuffle;
    using c_thread_cluster_lengths = CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
    static constexpr int c_block_transfer_scalar_per_vector = CDEBlockTransferScalarPerVector_NPerBlock;

    // Note: Backward data kernel does not have pipeline_scheduler or pipeline_version
    // as template parameters, so we don't expose them here

    // Specialization
    static constexpr auto conv_bwd_data_specialization = ConvBwdDataSpecialization;
    static constexpr bool do_pad_gemm_m = DoPadGemmM;
    static constexpr bool do_pad_gemm_n = DoPadGemmN;
    static constexpr int num_gemm_k_prefetch_stage = NumGemmKPrefetchStage;
};

} // namespace ck_tile::reflect
