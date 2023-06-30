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

#include "run_fmha_fwd.gfx90a.h"

#include <memory>

#include "device_gemm_trait.h"
#include "switch_wrapper.h"

namespace fwd_device_gemm {
void FmhaFwdRunner::Run() {
  headdim_switch(params_.d,
  bf16_switch(is_bf16_, 
  causal_switch(is_causal_, 
  deterministic_switch(is_deterministic_, [&]() {
    // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
    using FwdDeviceGemmTraits = device_gemm_trait::Forward<DataType, kMaskingSpec, kIsDeterministic>;
    using FwdDeviceGemmTemplate = FwdDeviceGemm<FwdDeviceGemmTraits>;
    auto fwd_device_gemm_instance_launcher_ptr = std::make_unique<FwdDeviceGemmInstanceLauncher<FwdDeviceGemmTemplate>>();
    fwd_device_gemm_instance_launcher_ptr->Launch(params_);
  }))));
} // FmhaFwdRunner::Run()
} // namespace fwd_device_gemm