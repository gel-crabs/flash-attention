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

#include "fwd_device_gemm_template.hpp"
#include "params.hpp"

namespace fwd_device_gemm {
#if defined(__WMMA__)
namespace wmma {
template <template <typename> typename DeviceGemmTemplate,
          typename DeviceGemmTraits>
class DeviceGemmInvoker {
  using Gemm = DeviceGemmTemplate<DeviceGemmTraits>;

public:
  // constructor for batched gemm
  explicit DeviceGemmInvoker(FlashFwdBatchedParams &params,
                             hipStream_t &stream) {
    auto gemm_ptr = std::make_unique<Gemm>();
    auto invoker = gemm_ptr->MakeInvoker();

    auto argument = gemm_ptr->MakeArgument(
        reinterpret_cast<const params.q.dtype>(params.q_ptr), params.k_ptr, params.v_ptr, params.out_ptr,
        params.max_seqlen_q, params.max_seqlen_kv, params.d, params.b,
        params.h_q, params.h_kv, params.softmax_scale, true, true);

    if (!gemm_ptr->IsSupportedArgument(argument)) {
      throw std::runtime_error(gemm_ptr->GetTypeString() +
                               " does not support this problem");
    }
    auto time_kernel = get_env_("FLASH_ATTENTION_INTERNAL_ENABLE_TIME_KERNEL");
    auto avg_time = invoker.Run(argument, StreamConfig{stream, time_kernel});

    if (time_kernel) {
      std::cout << "time elpase is " << avg_time << " ms" << std::endl;
    }
  }
};
} // namespace wmma
#endif
// TODO: add default implementation or error handling
} // namespace fwd_device_gemm