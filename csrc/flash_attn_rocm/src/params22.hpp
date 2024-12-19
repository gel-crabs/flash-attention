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

#include "mask.hpp"
#include "utils.hpp"

// TODO: Use shared_ptr to use the same memory of BaseParams when calling
// forward/backward parameters
// TODO: Fix input constness
// Common argements used by both batched & grouped gemms
struct BaseParams {
  explicit BaseParams(bool has_lse,
                      bool has_dropout_randval,
                      const mask_info &mask,
                      // sizes
                      const Index b,
                      const Index seqlen_q,
                      const Index seqlen_k,
                      const Index h,
                      const Index h_k,
                      const Index d,
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      c10::optional<at::Tensor> &alibi_slopes_,
                      at::Tensor out,
                      at::Tensor softmax_lse,
                      at::Tensor dropout_randval,
                      float softmax_scale,
                      float p_dropout,
                      std::pair<uint64_t*, uint64_t*> drop_seed_offset) :
        has_lse(has_lse),
        has_dropout_randval(has_dropout_randval),
        b(b),
        seqlen_q(seqlen_q),
        seqlen_k(seqlen_k),
        h(h),
        h_k(h_k),
        d(d),
        q(q),
        k(k),
        v(v),
        out(out),
        softmax_lse(softmax_lse),
        dropout_randval(dropout_randval),
        softmax_scale(softmax_scale),
        p_dropout(p_dropout),
        is_bf16(q.dtype() == torch::kBFloat16),
        is_dropout(p_dropout > 0.0f),
        is_mnko_padding(false),
        is_causal(is_causal),
        q_seq_stride(q.stride(-3)),
        kv_seq_stride(k.stride(-3)),
        out_seq_stride(out.stride(-3)),
        q_head_stride(q.stride(-2)),
        kv_head_stride(k.stride(-2)),
        out_head_stride(out.stride(-2)),
        softmax_lse_batch_stride(softmax_lse.stride(0)) {
    TORCH_CHECK(p_dropout < 1.f);
    is_mnko_padding = ((d % 32) != 0) || (d == 96);
    if (d > 512) {
      std::cout << "Unsupported head dimension" << std::endl;
    }
  }
  // The dimensions.
  Index b, seqlen_q, seqlen_k, d;

  // The number of heads.
  Index h, h_k;

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
      bool has_lse,
      bool has_dropout_randval,
      const mask_info &mask,
      const Index b,
      const Index seqlen_q,
      const Index seqlen_k,
      const Index h,
      const Index h_k,
      const Index d,
      const at::Tensor q,
      const at::Tensor k,
      const at::Tensor v,
      c10::optional<at::Tensor> &alibi_slopes_,
      at::Tensor out,
      at::Tensor softmax_lse,
      at::Tensor dropout_randval,
      const float softmax_scale,
      const float p_dropout,
      std::pair<uint64_t*, uint64_t*> drop_seed_offset)
      : BaseParams(has_lse, has_dropout_randval, mask, b, seqlen_q, seqlen_k, h, h_k, d, q, k, v, out,
                   softmax_lse, dropout_randval, softmax_scale, p_dropout, drop_seed_offset),
        q_ptr(q.data_ptr()),
        k_ptr(k.data_ptr()),
        v_ptr(v.data_ptr()),
        out_ptr(out.data_ptr()),
        softmax_lse_ptr(softmax_lse.data_ptr()),
        q_batch_stride(q.stride(0)),
        kv_batch_stride(k.stride(0)),
        out_batch_stride(out.stride(0)) {
    if (!is_mnko_padding && d <= 32) {
      is_mnko_padding =
          ((seqlen_q % 128) == 0 && (seqlen_k % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 64) {
      if (is_dropout) {
        is_mnko_padding =
            ((seqlen_q % 128) == 0 && (seqlen_k % 128) == 0 ? false
                                                                     : true);
      } else {
        is_mnko_padding =
            ((seqlen_q % 128) == 0 && (seqlen_k % 256) == 0 ? false
                                                                     : true);
      }
    } else if (!is_mnko_padding && d <= 128) {
      is_mnko_padding =
          ((seqlen_q % 128) == 0 && (seqlen_k % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 256) {
      is_mnko_padding =
          ((seqlen_q % 128) == 0 && (seqlen_k % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 512) {
      is_mnko_padding =
          ((seqlen_q % 128) == 0 && (seqlen_k % 128) == 0 ? false
                                                                   : true);
    }

    // TODO: Change to tensor.shape()
    // Q layout [b, seqlen_q, h, d]
    q_lengths = std::vector<Index>{b, h, seqlen_q, d};
    q_strides =
        std::vector<Index>{q_batch_stride, q_head_stride, q_seq_stride, 1};

    // K layout [b, seqlen_k, h_k, d]
    k_lengths = std::vector<Index>{b, h_k, seqlen_k, d};
    k_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

    // V layout [b, seqlen_k, h_k, d]
    v_lengths = std::vector<Index>{b, h_k, d, seqlen_k};
    v_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

    // Y layout [b, seqlen_q, h, d]
    out_lengths = std::vector<Index>{b, h, seqlen_q, d};
    out_strides = std::vector<Index>{out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

    // LSE layout [b, h, seqlen_q]
    lse_lengths = std::vector<Index>{b, h, seqlen_q};
    // std::vector<Index> lse_strides{h*seqlen_q, seqlen_q, 1};
  }

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  void *__restrict__ z_ptr;
  void *__restrict__ out_ptr;
  void *__restrict__ softmax_lse_ptr;

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
      bool has_lse,
      bool has_dropout_randval,
      const mask_info &mask,
      const Index b,
      const Index seqlen_q,
      const Index seqlen_k,
      const Index h,
      const Index h_k,
      const Index d,
      const at::Tensor &q,
      const at::Tensor &k,
      const at::Tensor &v,
      c10::optional<at::Tensor> &alibi_slopes_,
      at::Tensor out,
      at::Tensor z,
      at::Tensor softmax_lse,
      at::Tensor dropout_randval,
      const float softmax_scale,
      const float p_dropout,
      std::pair<uint64_t*, uint64_t*> drop_seed_offset)
      : BatchedParams(has_lse, has_dropout_randval, mask, b, seqlen_q, seqlen_k, h, h_k, d, q, k, v, out,
                   softmax_lse, dropout_randval, softmax_scale, p_dropout, drop_seed_offset) {
    z_ptr = return_softmax ? z.data_ptr() : nullptr;

    // Z layout [b, h, seqlen_q, seqlen_k]
    z_lengths = std::vector<Index>{b, h, seqlen_q, seqlen_k};
    z_strides =
        std::vector<Index>{h * seqlen_q * seqlen_k,
                           seqlen_q * seqlen_k, seqlen_k, 1};
  }

  bool return_softmax;
};

#if !defined(__WMMA__)
// Backward Batched Arguments
struct FlashBwdBatchedParams : public BatchedParams {
  explicit FlashBwdBatchedParams(
      const Index b, const Index seqlen_q, const Index seqlen_k,
      const Index h, const Index h_k, const Index d, const at::Tensor &q,
      const at::Tensor &k, const at::Tensor &v,
      at::Tensor &out, // TODO: Fix constness
      const at::Tensor &dout, at::Tensor &dq, at::Tensor &dk,
      at::Tensor &dv, at::Tensor &dsoftmax,
      at::Tensor &softmax_lse, // TODO: Fix constness
      const float p_dropout, const float softmax_scale, const bool is_causal)
      : BatchedParams(b, seqlen_q, seqlen_k, h, h_k, d, q, k, v,
                      out, softmax_lse, p_dropout, softmax_scale, is_causal),
        dq_ptr(dq.data_ptr()), dk_ptr(dk.data_ptr()), dv_ptr(dv.data_ptr()),
        dout_ptr(dout.data_ptr()), dsoftmax_ptr(dsoftmax.data_ptr()) {
    z_ptr = nullptr;

    Index dkv_batch_stride = dk.stride(0);
    Index dkv_seq_stride = dk.stride(-3);
    Index dq_seq_stride = dq.stride(-3);
    Index dout_seq_stride = dout.stride(-3);
    Index dkv_head_stride = dk.stride(-2);

    TORCH_CHECK(dq_seq_stride == q_seq_stride);
    TORCH_CHECK(dout_seq_stride == out_seq_stride);

    // Z layout [b, h, seqlen_q, seqlen_k]
    z_lengths = std::vector<Index>{b, h, seqlen_q, seqlen_k};
    z_strides =
        std::vector<Index>{h * seqlen_q * seqlen_k,
                           seqlen_q * seqlen_k, seqlen_k, 1};

    // MQA / GQA readiness
    // KGrad layout [b, seqlen_k, h, d]
    dk_lengths = std::vector<Index>{b, h, seqlen_k, d};
    dk_strides = std::vector<Index>{dkv_batch_stride, dkv_head_stride,
                                    dkv_seq_stride, 1};

    // VGrad layout [b, seqlen_k, h, d]
    dv_lengths = std::vector<Index>{b, h, d, seqlen_k};
    dv_strides = std::vector<Index>{dkv_batch_stride, dkv_head_stride, 1,
                                    dkv_seq_stride};
  }

  void *__restrict__ dq_ptr;
  void *__restrict__ dk_ptr;
  void *__restrict__ dv_ptr;

  void *__restrict__ dout_ptr;
  void *__restrict__ dsoftmax_ptr;

  // KGrad layout [b, seqlen_k, h, d]
  std::vector<Index> dk_lengths;
  std::vector<Index> dk_strides;

  // VGrad layout [b, seqlen_k, h, d]
  std::vector<Index> dv_lengths;
  std::vector<Index> dv_strides;
};

// Common Grouped Arguments
struct GroupedParams : public BaseParams {
  explicit GroupedParams(const Index b, const Index seqlen_q,
                         const Index seqlen_k, const Index h,
                         const Index h_k, const Index d,
                         const at::Tensor &q, const at::Tensor &k,
                         const at::Tensor &v, at::Tensor &out,
                         const void *cu_seqlens_q_d,
                         const void *cu_seqlens_kv_d,
                         at::Tensor &softmax_lse, const float p_dropout,
                         const float softmax_scale, const bool is_causal)
      : BaseParams(b, seqlen_q, seqlen_k, h, h_k, d, q, k, v, out,
                   softmax_lse, p_dropout, softmax_scale, is_causal),
        seqlens_q(
            get_host_seqlens(static_cast<const int *>(cu_seqlens_q_d), b)),
        seqlens_kv(
            get_host_seqlens(static_cast<const int *>(cu_seqlens_kv_d), b)) {
    char *q_ptr = reinterpret_cast<char *>(q.data_ptr());
    char *k_ptr = reinterpret_cast<char *>(k.data_ptr());
    char *v_ptr = reinterpret_cast<char *>(v.data_ptr());
    char *out_ptr = reinterpret_cast<char *>(out.data_ptr());
    char *softmax_lse_ptr = reinterpret_cast<char *>(softmax_lse.data_ptr());

    // TODO: move to GPU
    for (int i = 0; i < b; ++i) {
      int curr_q_batch_stride = seqlens_q[i] * q_seq_stride;
      int curr_kv_batch_stride = seqlens_kv[i] * kv_seq_stride;
      int curr_out_batch_stride = seqlens_q[i] * out_seq_stride;

      if (!is_mnko_padding) {
        if (!is_dropout && d > 32 && d <= 64) {
          is_mnko_padding =
              !((seqlens_q[i] % 128) == 0 && (seqlens_kv[i] % 256) == 0);
        } else {
          is_mnko_padding =
              !((seqlens_q[i] % 128) == 0 && (seqlens_kv[i] % 128) == 0);
        }
      }

      q_ptrs.push_back(reinterpret_cast<void *>(q_ptr));
      q_ptr += get_size_in_bytes(curr_q_batch_stride, q.dtype());

      k_ptrs.push_back(reinterpret_cast<void *>(k_ptr));
      k_ptr += get_size_in_bytes(curr_kv_batch_stride, k.dtype());

      v_ptrs.push_back(reinterpret_cast<void *>(v_ptr));
      v_ptr += get_size_in_bytes(curr_kv_batch_stride, v.dtype());

      out_ptrs.push_back(reinterpret_cast<void *>(out_ptr));
      out_ptr += get_size_in_bytes(curr_out_batch_stride, out.dtype());

      softmax_lse_ptrs.push_back(reinterpret_cast<void *>(softmax_lse_ptr));
      softmax_lse_ptr +=
          get_size_in_bytes(softmax_lse_batch_stride, softmax_lse.dtype());

      // Q layout [b, seqlen_q, h, d]
      std::vector<Index> q_lengths{1, h, seqlens_q[i], d};
      std::vector<Index> q_strides{curr_q_batch_stride, q_head_stride,
                                   q_seq_stride, 1};

      // K layout [b, seqlen_k, h_k, d]
      std::vector<Index> k_lengths{1, h_k, seqlens_kv[i], d};
      std::vector<Index> k_strides{curr_kv_batch_stride, kv_head_stride,
                                   kv_seq_stride, 1};

      // V layout [b, seqlen_k, h_k, d]
      std::vector<Index> v_lengths{1, h_k, d, seqlens_kv[i]};
      std::vector<Index> v_strides{curr_kv_batch_stride, kv_head_stride, 1,
                                   kv_seq_stride};

      // Y layout [b, seqlen_q, h, d]
      std::vector<Index> out_lengths{1, h, seqlens_q[i], d};
      std::vector<Index> out_strides{curr_out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

      // LSE layout [b, h, seqlen_q]
      std::vector<Index> lse_lengths{1, h, seqlens_q[i]};
      std::vector<Index> lse_strides{h * seqlens_q[i], seqlens_q[i], 1};

      q_lengths_vec.push_back(q_lengths);
      q_strides_vec.push_back(q_strides);
      k_lengths_vec.push_back(k_lengths);
      k_strides_vec.push_back(k_strides);
      v_lengths_vec.push_back(v_lengths);
      v_strides_vec.push_back(v_strides);
      out_lengths_vec.push_back(out_lengths);
      out_strides_vec.push_back(out_strides);
      lse_lengths_vec.push_back(lse_lengths);
      lse_strides_vec.push_back(lse_strides);
    }
  }

  std::vector<int> seqlens_q;
  std::vector<int> seqlens_kv;

  std::vector<const void *> q_ptrs;
  std::vector<const void *> k_ptrs;
  std::vector<const void *> v_ptrs;
  std::vector<void *> z_ptrs;
  std::vector<void *> out_ptrs;
  std::vector<void *> softmax_lse_ptrs;

  std::vector<std::vector<Index>> q_lengths_vec;
  std::vector<std::vector<Index>> q_strides_vec;
  std::vector<std::vector<Index>> k_lengths_vec;
  std::vector<std::vector<Index>> k_strides_vec;
  std::vector<std::vector<Index>> v_lengths_vec;
  std::vector<std::vector<Index>> v_strides_vec;
  std::vector<std::vector<Index>> z_lengths_vec;
  std::vector<std::vector<Index>> z_strides_vec;
  std::vector<std::vector<Index>> out_lengths_vec;
  std::vector<std::vector<Index>> out_strides_vec;
  std::vector<std::vector<Index>> lse_lengths_vec;
  std::vector<std::vector<Index>> lse_strides_vec;
};

// Forward Grouped Arguments
struct FlashFwdGroupedParams : public GroupedParams {
  explicit FlashFwdGroupedParams(
      const Index b, const Index seqlen_q, const Index seqlen_k,
      const Index h, const Index h_k, const Index d, const at::Tensor &q,
      const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
      const void *cu_seqlens_q_d, const void *cu_seqlens_kv_d,
      std::vector<at::Tensor> &z_vec, at::Tensor &softmax_lse,
      const float p_dropout, const float softmax_scale, const bool is_causal,
      const bool return_softmax)
      : GroupedParams(b, seqlen_q, seqlen_k, h, h_k, d, q, k, v,
                      out, cu_seqlens_q_d, cu_seqlens_kv_d, softmax_lse,
                      p_dropout, softmax_scale, is_causal) {
    auto opts = q.options();
    for (int i = 0; i < b; ++i) {
      if (return_softmax) {
        z_vec.push_back(torch::empty({1, h, seqlens_q[i], seqlens_kv[i]},
                                     opts.dtype(torch::kUInt8)));
        z_ptrs.push_back(reinterpret_cast<void *>(z_vec[i].data_ptr()));
      } else {
        z_ptrs.push_back(nullptr);
      }

      // Z layout [b, h, seqlen_q, seqlen_k]
      std::vector<Index> z_lengths{1, h, seqlens_q[i], seqlens_kv[i]};
      std::vector<Index> z_strides{h * seqlens_q[i] * seqlens_kv[i],
                                   seqlens_q[i] * seqlens_kv[i], seqlens_kv[i],
                                   1};

      z_lengths_vec.push_back(z_lengths);
      z_strides_vec.push_back(z_strides);
    }
  }
};

// Backward Grouped Arguments
struct FlashBwdGroupedParams : public GroupedParams {
  explicit FlashBwdGroupedParams(
      const Index b, const Index seqlen_q, const Index seqlen_k,
      const Index h, const Index h_k, const Index d, const at::Tensor &q,
      const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
      const at::Tensor &dout, at::Tensor &dq, at::Tensor &dk,
      at::Tensor &dv, const void *cu_seqlens_q_d,
      const void *cu_seqlens_kv_d, std::vector<at::Tensor> &dsoftmax_vec,
      at::Tensor &softmax_lse, const float p_dropout,
      const float softmax_scale, const bool is_causal)
      : GroupedParams(b, seqlen_q, seqlen_k, h, h_k, d, q, k, v,
                      out, cu_seqlens_q_d, cu_seqlens_kv_d, softmax_lse,
                      p_dropout, softmax_scale, is_causal),
        bwd_out_ptrs(
            std::vector<const void *>(out_ptrs.begin(), out_ptrs.end())),
        bwd_softmax_lse_ptrs(std::vector<const void *>(
            softmax_lse_ptrs.begin(), softmax_lse_ptrs.end())) {
    // TODO: Use the actual type
    char *dq_ptr = reinterpret_cast<char *>(dq.data_ptr());
    char *dk_ptr = reinterpret_cast<char *>(dk.data_ptr());
    char *dv_ptr = reinterpret_cast<char *>(dv.data_ptr());
    char *dout_ptr = reinterpret_cast<char *>(dout.data_ptr());

    Index dq_seq_stride = dq.stride(-3);
    Index dkv_seq_stride = dk.stride(-3);
    Index dout_seq_stride = dout.stride(-3);
    Index dq_head_stride = dq.stride(-2);
    Index dkv_head_stride = dk.stride(-2);
    Index dout_head_stride = dout.stride(-2);

    TORCH_CHECK(dq_seq_stride == q_seq_stride);
    TORCH_CHECK(dout_seq_stride == out_seq_stride);

    auto opts = q.options();
    for (int i = 0; i < b; ++i) {
      // TODO: reuse it in the foward on GPU
      int curr_dq_batch_stride = seqlens_q[i] * dq_seq_stride;
      int curr_dkv_batch_stride = seqlens_kv[i] * dkv_seq_stride;
      int curr_dout_batch_stride = seqlens_q[i] * dout_seq_stride;

      z_ptrs.push_back(nullptr);

      dq_ptrs.push_back(reinterpret_cast<void *>(dq_ptr));
      dq_ptr += get_size_in_bytes(curr_dq_batch_stride, dq.dtype());

      dk_ptrs.push_back(reinterpret_cast<void *>(dk_ptr));
      dk_ptr += get_size_in_bytes(curr_dkv_batch_stride, dk.dtype());

      dv_ptrs.push_back(reinterpret_cast<void *>(dv_ptr));
      dv_ptr += get_size_in_bytes(curr_dkv_batch_stride, dv.dtype());

      dout_ptrs.push_back(reinterpret_cast<const void *>(dout_ptr));
      dout_ptr += get_size_in_bytes(curr_dout_batch_stride, dout.dtype());

      dsoftmax_vec.push_back(
          torch::empty({1, h, seqlens_q[i]}, opts.dtype(torch::kFloat32)));
      dsoftmax_ptrs.push_back(
          reinterpret_cast<void *>(dsoftmax_vec[i].data_ptr()));

      // Z layout [b, h, seqlen_q, seqlen_k]
      std::vector<Index> z_lengths =
          std::vector<Index>{b, h, seqlens_q[i], seqlens_kv[i]};
      std::vector<Index> z_strides =
          std::vector<Index>{h * seqlens_q[i] * seqlens_kv[i],
                             seqlens_q[i] * seqlens_kv[i], seqlens_kv[i], 1};

      // MQA / GQA readiness
      // KGrad layout [b, seqlen_k, h, d]
      std::vector<Index> dk_lengths{1, h, seqlens_kv[i], d};
      std::vector<Index> dk_strides{curr_dkv_batch_stride, dkv_head_stride,
                                    dkv_seq_stride, 1};

      // VGrad layout [b, seqlen_k, h, d]
      std::vector<Index> dv_lengths{1, h, d, seqlens_kv[i]};
      std::vector<Index> dv_strides{curr_dkv_batch_stride, dkv_head_stride, 1,
                                    dkv_seq_stride};

      z_lengths_vec.push_back(z_lengths);
      z_strides_vec.push_back(z_strides);
      dk_lengths_vec.push_back(dk_lengths);
      dk_strides_vec.push_back(dk_strides);
      dv_lengths_vec.push_back(dv_lengths);
      dv_strides_vec.push_back(dv_strides);
    }
  }

  std::vector<void *> dq_ptrs;
  std::vector<void *> dk_ptrs;
  std::vector<void *> dv_ptrs;

  std::vector<const void *> bwd_out_ptrs;
  std::vector<const void *> bwd_softmax_lse_ptrs;

  std::vector<const void *> dout_ptrs;
  std::vector<void *> dsoftmax_ptrs;

  // MQA / GQA readiness
  std::vector<std::vector<Index>> dk_lengths_vec;
  std::vector<std::vector<Index>> dk_strides_vec;
  std::vector<std::vector<Index>> dv_lengths_vec;
  std::vector<std::vector<Index>> dv_strides_vec;

  bool return_softmax;
};
#endif