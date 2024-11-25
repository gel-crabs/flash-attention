// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "device_gemm_trait.hpp"

namespace fwd_device_gemm {
namespace device_op =
    ck::tensor_operation::device; // namespace alias for internal use
#if defined(__WMMA__)
namespace wmma {
template <typename DeviceGemmTraits>
using DeviceGemmBatchedMQA32 = device_op::DeviceMultiQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec>;

template <typename DeviceGemmTraits>
using DeviceGemmBatchedMQA64 = device_op::DeviceMultiQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec>;

template <typename DeviceGemmTraits>
using DeviceGemmBatchedMQA128 = device_op::DeviceMultiQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec>;

template <typename DeviceGemmTraits>
using DeviceGemmBatchedGQA32 = device_op::DeviceGroupedQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    DeviceGemmTraits::QueryGroupNumber,
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec,
    ck::make_default_loop_scheduler(),
    ck::PipelineVersion::v2>;

template <typename DeviceGemmTraits>
using DeviceGemmBatchedGQA64 = device_op::DeviceGroupedQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    DeviceGemmTraits::QueryGroupNumber,
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec,
    ck::make_default_loop_scheduler(),
    ck::PipelineVersion::v2>;

template <typename DeviceGemmTraits>
using DeviceGemmBatchedGQA128 = device_op::DeviceGroupedQueryAttentionForward_Wmma<
    DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
    DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
    DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::QDataType,
    typename DeviceGemmTraits::KDataType, typename DeviceGemmTraits::VDataType,
    typename DeviceGemmTraits::OutDataType,
    typename DeviceGemmTraits::Acc0BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::Acc1BiasDataType,
    typename DeviceGemmTraits::AccDataType,
    typename DeviceGemmTraits::OutShuffleDataType,
    typename DeviceGemmTraits::QElementOp,
    typename DeviceGemmTraits::KElementOp,
    typename DeviceGemmTraits::Acc0ElementOp,
    typename DeviceGemmTraits::VElementOp,
    typename DeviceGemmTraits::OutElementOp,
    DeviceGemmTraits::kGemmSpec,
    DeviceGemmTraits::kTensorSpecQ,
    DeviceGemmTraits::kTensorSpecK,
    DeviceGemmTraits::kTensorSpecV,
    DeviceGemmTraits::kTensorSpecOut,
    1, // NumPrefetch
    DeviceGemmTraits::QueryGroupNumber,
    256, // BlockSize
    //      Gemm 0
    128, // MPerBlock
    128, // LPerBlock
    64, // KPerBlock
    8, // AK1
    8, // BK1
    //      Gemm 1
    64, // NPerBlock
    64, // LTilePerBlock
    8, // L1
    16, // MPerWmma
    16, // LPerWmma
    16, // NPerWmma
    // Per repeat = wave_m = wave_num, wave_n = 1
    1, // MRepeat
    8, // LRepeat
    4, // NRepeat
    // ABlockTransfer MK -> K0 M K1
    device_gemm_trait::S<2, 128, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // ABlockTransferSrcAccessOrder
    2, // ABlockTransferSrcVectorDim
    8, // ABlockTransferSrcScalarPerVector
    8, // ABlockTransferDstScalarPerVector_K1
    true, // ABlockLdsAddExtraM
    // B0BlockTransfer LK -> K0 L K1
    device_gemm_trait::S<8, 32, 1>, // B0BlockTransferThreadClusterLengths_K0_L_K1
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<1, 0, 2>, // B0BlockTransferSrcAccessOrder
    2, // B0BlockTransferSrcVectorDim
    8, // B0BlockTransferSrcScalarPerVector
    8, // B0BlockTransferDstScalarPerVector_K1
    true, // B0BlockLdsAddExtraL
    // B1BlockTransfer NL -> L0 N L1
    device_gemm_trait::S<2, 16, 8>, // B1BlockTransferThreadClusterLengths_L0_N_L1
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferThreadClusterArrangeOrder
    device_gemm_trait::S<0, 2, 1>, // B1BlockTransferSrcAccessOrder
    1, // B1BlockTransferSrcVectorDim
    1, // B1BlockTransferSrcScalarPerVector
    1, // B1BlockTransferDstScalarPerVector_L1
    false, // B1BlockLdsAddExtraN
    // CShuffleBlockTransfer MN
    1, // CShuffleMRepeatPerShuffle
    1, // CShuffleNRepeatPerShuffle
    device_gemm_trait::S<1, 128, 1, 2>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8, // CShuffleBlockTransferScalarPerVector_NPerBlock
    DeviceGemmTraits::kMaskingSpec,
    ck::make_default_loop_scheduler(),
    ck::PipelineVersion::v2>;
} // namespace wmma
#endif
// TODO: add default implementation or error handling
} // namespace fwd_device_gemm