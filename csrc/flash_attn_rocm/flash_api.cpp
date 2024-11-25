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

#include "flash_runner.hpp"

std::vector<torch::Tensor> mha_fwd(
    const torch::Tensor &q, // batch_size x seqlen_q x num_heads_q x head_size
    const torch::Tensor &k, // batch_size x seqlen_kv x num_heads_kv x head_size
    const torch::Tensor &v, // batch_size x seqlen_kv x num_heads_kv x head_size
    c10::optional<torch::Tensor>
        &out_, // batch_size x seqlen_q x num_heads_q x head_size
    const float p_dropout, const float softmax_scale, const bool is_causal,
    const bool return_softmax, c10::optional<at::Generator> gen_) {

  TORCH_CHECK(
      ck::is_xdl_supported() || ck::is_gfx11_supported(),
      "FlashAttention currently only supports MI100 and RX7000 and above");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");

  TORCH_CHECK(k.dtype() == q_dtype, "Query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "Query and value must have the same dtype");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_kv = k.size(1);
  const int num_heads_kv = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be postive");
  TORCH_CHECK(
      head_size_og <= 512,
      "FlashAttention forward only supports head dimension at most 512");
  TORCH_CHECK(
      num_heads_q % num_heads_kv == 0,
      "Number of heads in key/value must divide number of heads in Query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads_q, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_kv, num_heads_kv, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_kv, num_heads_kv, head_size_og);

  torch::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(
        q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(
        k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(
        v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  torch::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    TORCH_CHECK(out.is_cuda(), "Output tensor must be on ROCm device");
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads_q, head_size_og);
    if (head_size_og % 8 != 0) {
      out = torch::empty_like(q_padded);
    }
  } else {
    out = torch::empty_like(q_padded);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads_q, seqlen_q},
                                  opts.dtype(torch::kFloat32));
  torch::Tensor z;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when
    // p_dropout > 0.0");
    z = torch::empty({batch_size, num_heads_q, seqlen_q, seqlen_kv},
                     opts.dtype(torch::kUInt8));
  }

  FlashFwdBatchedParams params(batch_size, seqlen_q, seqlen_kv, num_heads_q,
                               num_heads_kv, head_size, q_padded, k_padded,
                               v_padded, out, z, softmax_lse, p_dropout,
                               softmax_scale, is_causal, return_softmax);

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  int64_t counter_offset = params.b * params.h_q * 32;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  auto philox_args = gen->philox_cuda_state(counter_offset);

  if (params.is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    params.seeds = unpack(philox_args);

    // pass to backward
    auto rng_state_ptr = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
    std::tie(rng_state_ptr[0], rng_state_ptr[1]) = params.seeds;
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  FlashRunner flash_runner;
  flash_runner.Run(params, stream);

  torch::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  return {out,        q_padded,    k_padded, v_padded,
          out_padded, softmax_lse, z,        rng_state};
}

std::vector<torch::Tensor> mha_varlen_fwd(
    const torch::Tensor
        &q, // total_q x num_heads_q x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor &k, // total_kv x num_heads_kv x head_size, total_kv :=
                            // \sum_{i=0}^{b} s_i
    const torch::Tensor &v, // total_kv x num_heads_kv x head_size, total_kv :=
                            // \sum_{i=0}^{b} s_i
    c10::optional<torch::Tensor> &out_, // total_q x num_heads_q x head_size,
                                        // total_kv := \sum_{i=0}^{b} s_i
    const torch::Tensor &cu_seqlens_q,  // b+1
    const torch::Tensor &cu_seqlens_kv, // b+1
    const int max_seqlen_q, const int max_seqlen_kv, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    const bool return_softmax, // in rocm ,this will return the random number
                               // matrix when doing dropout
    c10::optional<at::Generator> gen_) {

  TORCH_CHECK(
      ck::is_xdl_supported() || ck::is_gfx11_supported(),
      "FlashAttention currently only supports MI100 and RX7000 and above");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");

  TORCH_CHECK(k.dtype() == q_dtype, "Query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "Query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
              "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_kv.dtype() == torch::kInt32,
              "cu_seqlens_kv must have dtype int32");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be on ROCm device");
  TORCH_CHECK(cu_seqlens_kv.is_cuda(), "cu_seqlens_kv must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_kv.is_contiguous(),
              "cu_seqlens_kv must be contiguous");

  const auto sizes = q.sizes();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads_q = sizes[1];
  const int head_size_og = sizes[2];
  const int total_kv = k.size(0);
  const int num_heads_kv = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size_og <= 128,
      "FlashAttention forward only supports head dimension at most 128");
  TORCH_CHECK(
      num_heads_q % num_heads_kv == 0,
      "Number of heads in key/value must divide number of heads in Query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, total_q, num_heads_q, head_size_og);
  CHECK_SHAPE(k, total_kv, num_heads_kv, head_size_og);
  CHECK_SHAPE(v, total_kv, num_heads_kv, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_kv, batch_size + 1);

  torch::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(
        q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(
        k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(
        v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  torch::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    TORCH_CHECK(out.is_cuda(), "Output tensor must be on ROCm device");
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, total_q, num_heads_q, head_size_og);
    if (head_size_og % 8 != 0) {
      out = torch::empty_like(q_padded);
    }
  } else {
    out = torch::empty_like(q_padded);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_lse = torch::empty({batch_size, num_heads_q, max_seqlen_q},
                                  opts.dtype(torch::kFloat32));

  std::vector<torch::Tensor> z_vec;
  if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    z_vec.reserve(batch_size);
  }

  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    // if (return_softmax) {z.zero_();}
  }

  FlashFwdGroupedParams params(
      batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv,
      head_size, q_padded, k_padded, v_padded, out, cu_seqlens_q.data_ptr(),
      cu_seqlens_kv.data_ptr(), z_vec, softmax_lse, p_dropout, softmax_scale,
      is_causal, return_softmax);

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  int64_t counter_offset = params.b * params.h_q * 32;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  auto philox_args = gen->philox_cuda_state(counter_offset);

  if (params.is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    params.seeds = unpack(philox_args);

    // pass to backward
    auto rng_state_ptr = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
    std::tie(rng_state_ptr[0], rng_state_ptr[1]) = params.seeds;
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  FlashRunner flash_runner;
  flash_runner.Run(params, stream);

  torch::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  torch::Tensor z;
  if (return_softmax) {
    for (auto &z : z_vec) {
      auto pad_options = torch::nn::functional::PadFuncOptions(
          {0, max_seqlen_kv - z.size(-1), 0, max_seqlen_q - z.size(-2)});
      z = torch::nn::functional::pad(z, pad_options);
    }
    z = torch::cat(z_vec, 0);
  }

  return {out,        q_padded,    k_padded, v_padded,
          out_padded, softmax_lse, z,        rng_state};
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
  m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
}
