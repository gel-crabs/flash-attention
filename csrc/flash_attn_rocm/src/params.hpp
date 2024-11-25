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

#include <memory>
#include <vector>

#include "utils.hpp"

// TODO: Use shared_ptr to use the same memory of BaseParams when calling
// forward/backward parameters
// TODO: Fix input constness
// Common argements used by both batched & grouped gemms
struct BaseParams {
  explicit BaseParams(const Index b, const Index max_seqlen_q,
                      const Index max_seqlen_kv, const Index h_q,
                      const Index h_kv, const Index d, const torch::Tensor &q,
                      const torch::Tensor &k, const torch::Tensor &v,
                      torch::Tensor &out, torch::Tensor &softmax_lse,
                      const float p_dropout, const float softmax_scale,
                      const bool is_causal)
      : b(b), max_seqlen_q(max_seqlen_q), max_seqlen_kv(max_seqlen_kv),
        h_q(h_q), h_kv(h_kv), d(d), p_dropout(p_dropout),
        softmax_scale(softmax_scale), is_bf16(q.dtype() == torch::kBFloat16),
        is_dropout(p_dropout > 0.0f), is_mnko_padding(false),
        is_causal(is_causal), q_seq_stride(q.stride(-3)),
        kv_seq_stride(k.stride(-3)), out_seq_stride(out.stride(-3)),
        q_head_stride(q.stride(-2)), kv_head_stride(k.stride(-2)),
        out_head_stride(out.stride(-2)),
        softmax_lse_batch_stride(softmax_lse.stride(0)) {
    TORCH_CHECK(p_dropout < 1.f);
    is_mnko_padding = ((d % 32) != 0) || (d == 96);
    if (d > 512) {
      std::cout << "Unsupported head dimension" << std::endl;
    }
  }
  // The dimensions.
  Index b, max_seqlen_q, max_seqlen_kv, d;

  // The number of heads.
  Index h_q, h_kv;

  // The scaling factors for the kernel.
  float softmax_scale;
  // float softmax_scale_log2;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint8_t p_dropout_in_uint8_t;

  // seeds
  std::tuple<uint64_t, uint64_t> seeds;

  bool is_bf16;
  bool is_dropout;
  bool is_mnko_padding;
  bool is_causal;

  Index q_seq_stride;
  Index kv_seq_stride;
  Index out_seq_stride;

  Index q_head_stride;
  Index kv_head_stride;
  Index out_head_stride;

  Index softmax_lse_batch_stride;

  static inline const bool kIsUnitTestMode =
      get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE");
  static inline const bool kIsDeterministic =
      get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC");
};

// Common Batched Arguments
struct BatchedParams : public BaseParams {
  explicit BatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, const torch::Tensor &q,
      const torch::Tensor &k, const torch::Tensor &v, torch::Tensor &out,
      torch::Tensor
          &softmax_lse, // TODO: forward reference, backward const reference
      const float p_dropout, const float softmax_scale, const bool is_causal)
      : BaseParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v, out,
                   softmax_lse, p_dropout, softmax_scale, is_causal),
        softmax_lse_ptr(softmax_lse.data_ptr()),
        q_batch_stride(q.stride(0)), kv_batch_stride(k.stride(0)),
        out_batch_stride(out.stride(0)) {
    if (!is_mnko_padding && d <= 32) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 64) {
      if (is_dropout) {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                     : true);
      } else {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 256) == 0 ? false
                                                                     : true);
      }
    } else if (!is_mnko_padding && d <= 128) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 256) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 512) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    }

    // TODO: Change to tensor.shape()
    // Q layout [b, max_seqlen_q, h_q, d]
    q_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    q_strides =
        std::vector<Index>{q_batch_stride, q_head_stride, q_seq_stride, 1};

    // K layout [b, max_seqlen_kv, h_kv, d]
    k_lengths = std::vector<Index>{b, h_kv, max_seqlen_kv, d};
    k_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

    // V layout [b, max_seqlen_kv, h_kv, d]
    v_lengths = std::vector<Index>{b, h_kv, d, max_seqlen_kv};
    v_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

    // Y layout [b, max_seqlen_q, h_q, d]
    out_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    out_strides = std::vector<Index>{out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

    // LSE layout [b, h_q, max_seqlen_q]
    lse_lengths = std::vector<Index>{b, h_q, max_seqlen_q};
    // std::vector<Index> lse_strides{h_q*max_seqlen_q, max_seqlen_q, 1};
  }

  const void* q_ptr = reinterpret_cast<const void *>(q.data_ptr());
  const void* k_ptr = reinterpret_cast<const void *>(k.data_ptr());
  const void* v_ptr = reinterpret_cast<const void *>(v.data_ptr());
  void* out_ptr = reinterpret_cast<void *>(out.data_ptr());

  void *__restrict__ z_ptr;
  void *__restrict__ softmax_lse_ptr;
  //void *__restrict__ q_ptr;
  //void *__restrict__ k_ptr;
  //void *__restrict__ v_ptr;

  //void *__restrict__ out_ptr;


  Index q_batch_stride;
  Index kv_batch_stride;
  Index out_batch_stride;
  Index softmax_lse_batch_stride;

  std::vector<Index> q_lengths;
  std::vector<Index> q_strides;
  std::vector<Index> k_lengths;
  std::vector<Index> k_strides;
  std::vector<Index> v_lengths;
  std::vector<Index> v_strides;
  std::vector<Index> z_lengths;
  std::vector<Index> z_strides;
  std::vector<Index> out_lengths;
  std::vector<Index> out_strides;
  std::vector<Index> lse_lengths;
  // std::vector<Index> lse_strides;
};

// Forward Batched Arguments
struct FlashFwdBatchedParams : public BatchedParams {
  explicit FlashFwdBatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, const torch::Tensor &q,
      const torch::Tensor &k, const torch::Tensor &v, torch::Tensor &out,
      torch::Tensor &z,
      torch::Tensor
          &softmax_lse, // TODO: forward reference, backward const reference
      const float p_dropout, const float softmax_scale, const bool is_causal,
      const bool return_softmax)
      : BatchedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, softmax_lse, p_dropout, softmax_scale, is_causal) {
    z_ptr = return_softmax ? z.data_ptr() : nullptr;

    // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
    z_lengths = std::vector<Index>{b, h_q, max_seqlen_q, max_seqlen_kv};
    z_strides =
        std::vector<Index>{h_q * max_seqlen_q * max_seqlen_kv,
                           max_seqlen_q * max_seqlen_kv, max_seqlen_kv, 1};
  }

  bool return_softmax;
};

