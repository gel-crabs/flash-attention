// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define BF16_SWITCH(COND, ...)                                                 \
  [&] {                                                                        \
    if (COND) {                                                                \
      using T = device_gemm_trait::BFloat16;                                   \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using T = device_gemm_trait::Float16;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)                                           \
  [&] {                                                                        \
    if (HEADDIM <= 32) {                                                       \
      constexpr static int kHeadDim = 32;                                      \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM <= 64) {                                                \
      constexpr static int kHeadDim = 64;                                      \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM <= 128) {                                               \
      constexpr static int kHeadDim = 128;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define GROUP_SWITCH(GROUP, DTYPE, ...)                                        \
  [&] {                                                                        \
    if (GROUP == 10) {                                                         \
      constexpr static int kQueryGroupNumber = 10;                             \
    } else if (GROUP == 20) {                                                  \
      constexpr static int kQueryGroupNumber = 20;                             \
    }                                                                          \
    if (DTYPE == "fp16") {                                                     \
      using kDataType = device_gemm_trait::BFloat16;                           \
      return __VA_ARGS__();                                                    \
    } else if (DTYPE == "bf16") {                                              \
      using kDataType = device_gemm_trait::Float16;                            \
      return __VA_ARGS__();                                                    \
    }
  }()