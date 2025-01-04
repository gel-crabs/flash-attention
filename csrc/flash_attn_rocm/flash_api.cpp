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
// specific prior written permission.

#include "flash_common.hpp"

#include "flash_runner.hpp"
#include "mask.hpp"


std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &k,              // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &v,              // batch_size x seqlen_k x num_heads_k x head_size
    c10::optional<at::Tensor> &out_,  // batch_size x seqlen_q x num_heads x head_size
    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float /*softcap*/,
    const bool return_dropout_randval,
    c10::optional<at::Generator> gen_)
{
  TORCH_CHECK(
      ck::is_xdl_supported() || ck::is_gfx11_supported(),
      "FlashAttention currently only supports MI100 and RX7000 and above");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");

  TORCH_CHECK(k.dtype() == q_dtype, "Query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "Query and value must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be postive");
  TORCH_CHECK(head_size <= 512, "FlashAttention forward only supports head dimension at most 512");
  TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in Query");

  if (window_size_left >= seqlen_k) { window_size_left = -1; }
  if (window_size_right >= seqlen_k) { window_size_right = -1; }

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

  mask_info mask;
  if (is_causal) {
      // Causal is the special case where window_size_right == 0 and window_size_left < 0.
      window_size_right = 0;
      std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
      mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
  }
  else if (window_size_left == -1 && window_size_right == -1) {
      mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
  }
  else {
      // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
      std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
      mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
  }

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
  // H/t Daniel Haziza
  const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
  const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
      q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
      seqlen_q = ngroups;
      num_heads = num_heads_k;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
    if (seqlenq_ngroups_swapped) {
        out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
    }
  } else {
    out = torch::empty_like(q);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{q.device()};

  auto opts = q.options();
  bool has_lse = true;
  bool has_dropout = p_dropout > 0.0f;

  at::Tensor softmax_lse;
  softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_dropout_randval) {
    TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
    p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(torch::kUInt8));
  }
  else {
    p = torch::empty({ 0 }, opts);
  }

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  int64_t counter_offset = batch_size * num_heads * 32;
  auto options = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (p_dropout > 0.0) {
      auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
          gen_, at::cuda::detail::getDefaultCUDAGenerator());
      std::lock_guard<std::mutex> lock(gen->mutex_);
      auto philox_args = gen->philox_cuda_state(counter_offset);
      hipLaunchKernelGGL(
          flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
  }

  if (seqlen_k > 0) {
      auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
      auto stream = at::cuda::getCurrentHIPStream().stream();
      FlashFwdBatchedParams params(batch_size, seqlen_q, seqlen_k, num_heads,
                               num_heads_k, head_size, q, k,
                               v, out, p, softmax_lse, p_dropout,
                               softmax_scale, is_causal, return_dropout_randval);
      FlashRunner flash_runner;
      flash_runner.Run(params, stream);
  }
  else {
      // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
      out.zero_();
      softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  if (seqlenq_ngroups_swapped) {
      out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
      q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
      softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }

  return {out,        softmax_lse, p,        rng_state};
}

void dummy_varlen_fwd() {
  throw std::runtime_error("Function 'varlen_fwd' is not available when __WMMA__ is defined.");
}

void dummy_bwd() {
  throw std::runtime_error("Function 'bwd' is not available when __WMMA__ is defined.");
}

void dummy_varlen_bwd() {
  throw std::runtime_error("Function 'varlen_bwd' is not available when __WMMA__ is defined.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashAttention";
  m.def("fwd", &mha_fwd, "Forward pass");
  m.def("varlen_fwd", &dummy_varlen_fwd, "Forware pass (variable length, dummy)");
  m.def("bwd", &dummy_bwd, "Backward pass (dummy)");
  m.def("varlen_bwd", &dummy_varlen_bwd, "Backward pass (variable length, dummy)");
}
