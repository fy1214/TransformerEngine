/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "../util/math.h"
#include "../util/vectorized_pointwise.h"
#include "./activation_template.h"

void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_silu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, stream);
}

void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsilu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dsilu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, e, stream);
}

void nvte_swiglu_with_row_amax(const NVTETensor input, NVTETensor output, float* row_amax,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu_with_row_amax);
  using namespace transformer_engine;

  const Tensor* input_tensor = convertNVTETensorCheck(input);
  Tensor* output_tensor = convertNVTETensorCheck(output);

  CheckInputTensor(*input_tensor, "swiglu_with_row_amax_input");
  CheckOutputTensor(*output_tensor, "swiglu_with_row_amax_output", /*allow_empty=*/false);

  NVTE_CHECK(row_amax != nullptr, "row_amax must be non-null");
  NVTE_CHECK(input_tensor->flat_last_dim() % 2 == 0,
             "SwiGLU input last dim must be even, got ", input_tensor->flat_last_dim(), ".");

  const size_t rows = input_tensor->flat_first_dim();
  const size_t cols = input_tensor->flat_last_dim() / 2;

  NVTE_CHECK(output_tensor->flat_first_dim() == rows,
             "SwiGLU output rows mismatch: expected ", rows, ", got ",
             output_tensor->flat_first_dim(), ".");
  NVTE_CHECK(output_tensor->flat_last_dim() == cols,
             "SwiGLU output cols mismatch: expected ", cols, ", got ",
             output_tensor->flat_last_dim(), ".");
  NVTE_CHECK(!is_fp8_dtype(output_tensor->dtype()),
             "nvte_swiglu_with_row_amax expects a high-precision output tensor.");
  NVTE_CHECK(output_tensor->has_data(), "SwiGLU output data must be allocated.");

  if (rows > 0) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(row_amax, 0, rows * sizeof(float), stream));
  }

  Empty e = {};
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input_tensor->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output_tensor->dtype(), OType,

          constexpr int nvec = 32 / sizeof(IType);
          GatedActivationKernelLauncher<nvec, fp32, Empty, silu<fp32, fp32>>(
              reinterpret_cast<const IType*>(input_tensor->data.dptr),
              reinterpret_cast<OType*>(output_tensor->data.dptr),
              /*scale=*/nullptr, /*amax=*/nullptr, /*scale_inv=*/nullptr, rows, cols, e, stream,
              row_amax););  // NOLINT(*)
  );                        // NOLINT(*)
}

void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(grad, input, output, e, stream);
}

void nvte_clamped_swiglu(const NVTETensor input, NVTETensor output, float limit, float alpha,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_swiglu);
  using namespace transformer_engine;
  // Preserve original behavior: linear (gate) component offset is hard-coded to 1.0f.
  ClampedSwiGLUParam param = {limit, alpha, /*glu_linear_offset=*/1.0f};
  gated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_clamped_swiglu_v2(const NVTETensor input, NVTETensor output, float limit, float alpha,
                            float glu_linear_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_swiglu_v2);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  gated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_clamped_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                          float limit, float alpha, cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_dswiglu);
  using namespace transformer_engine;
  // Preserve original behavior: linear (gate) component offset is hard-coded to 1.0f.
  ClampedSwiGLUParam param = {limit, alpha, /*glu_linear_offset=*/1.0f};
  dgated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>, clamped_dsilu<fp32, fp32>>(
      grad, input, output, param, stream);
}

void nvte_clamped_dswiglu_v2(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                             float limit, float alpha, float glu_linear_offset,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_dswiglu_v2);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  dgated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>, clamped_dsilu<fp32, fp32>>(
      grad, input, output, param, stream);
}
