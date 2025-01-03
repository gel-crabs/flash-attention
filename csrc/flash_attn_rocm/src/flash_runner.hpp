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

#include "fwd_device_gemm_invoker.hpp"

#include "static_switch.hpp"

class FlashRunner {
public:
  template <typename FlashParams>
  void Run(FlashParams &params, hipStream_t &stream) {
      BF16_SWITCH(params.is_bf16, [&] {
        BOOL_SWITCH(params.is_mnko_padding, kIsPadding, [&] {
          BOOL_SWITCH(params.is_causal, kIsCausal, [&] {
            this->template run_<FlashParams, T, kIsPadding, kIsCausal>(
                params, stream);
          });
        });
      });
  }

private:
  template <typename FlashParams, typename T, bool kIsPadding,
            bool kIsCausal>
  void run_(FlashParams &params, hipStream_t &stream);

  template <typename FlashFwdParams,
            template <typename> typename DeviceGemmTemplate, typename T,
            device_gemm_trait::GemmSpec kGemmSpec,
            device_gemm_trait::MaskingSpec kMaskingSpec, int kQueryGroupNumber, bool kIsDeterministic>
  void run_fwd_(FlashFwdParams &params, hipStream_t &stream) {
    // input, output, gemm, dropout, cshuffle, masking specialization,
    using DeviceGemmTraits =
        device_gemm_trait::Forward<T, kGemmSpec, kMaskingSpec, kQueryGroupNumber>;
    using Invoker = fwd_device_gemm::wmma::DeviceGemmInvoker<DeviceGemmTemplate,
                                                             DeviceGemmTraits>;
    Invoker(params, stream);
  }
};
