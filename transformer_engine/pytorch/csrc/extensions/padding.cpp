/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

void fused_multi_row_padding(at::Tensor input, at::Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list,
                             std::optional<at::Tensor> amax_input,
                             std::optional<at::Tensor> amax_output) {
  NVTE_CHECK(input_row_list.size() == padded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal  2.");

  const bool has_amax_in = amax_input.has_value() && amax_input->defined();
  const bool has_amax_out = amax_output.has_value() && amax_output->defined();
  NVTE_CHECK(has_amax_in == has_amax_out,
             "amax_input and amax_output must both be provided or both omitted.");

  // Optional companion amax: keep as a single contiguous FP32 buffer and pass
  // raw pointers into the kernel launcher (no per-expert NVTETensor list).
  const float* amax_in_ptr = nullptr;
  float* amax_out_ptr = nullptr;
  at::Tensor amax_in_storage, amax_out_storage;
  if (has_amax_in) {
    amax_in_storage = amax_input->contiguous().view(-1);
    amax_out_storage = amax_output->view(-1);
    NVTE_CHECK(amax_in_storage.scalar_type() == at::kFloat &&
                   amax_out_storage.scalar_type() == at::kFloat,
               "amax tensors must be FP32.");
    NVTE_CHECK(amax_in_storage.numel() == input.size(0),
               "amax_input length must match input rows.");
    NVTE_CHECK(amax_out_storage.numel() == output.size(0),
               "amax_output length must match padded output rows.");
    NVTE_CHECK(amax_out_storage.is_contiguous(), "amax_output must be contiguous.");
    amax_in_ptr = amax_in_storage.data_ptr<float>();
    amax_out_ptr = amax_out_storage.data_ptr<float>();
  }

  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<TensorWrapper> tensor_wrappers;
  std::vector<int> padded_num_rows_list;

  // Activation splits (input uses unpadded row counts; output uses padded counts).
  {
    void* d_input_ptr = reinterpret_cast<void*>(input.data_ptr());
    void* d_output_ptr = reinterpret_cast<void*>(output.data_ptr());
    const auto dtype = GetTransformerEngineDType(input.scalar_type());
    auto make_tensor = [&](void* dptr, const std::vector<size_t>& shape) -> NVTETensor {
      tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
      return tensor_wrappers.back().data();
    };
    for (size_t tensor_id = 0; tensor_id < input_row_list.size(); ++tensor_id) {
      const size_t in_rows = input_row_list[tensor_id];
      const size_t out_rows = padded_input_row_list[tensor_id];
      if (d_input_ptr == nullptr || in_rows == 0) {
        char* in_char = reinterpret_cast<char*>(d_input_ptr);
        in_char += in_rows * input.size(1) * input.element_size();
        d_input_ptr = reinterpret_cast<void*>(in_char);
        char* out_char = reinterpret_cast<char*>(d_output_ptr);
        out_char += out_rows * output.size(1) * output.element_size();
        d_output_ptr = reinterpret_cast<void*>(out_char);
        continue;
      }
      nvte_input_list.emplace_back(
          make_tensor(d_input_ptr, {in_rows, static_cast<size_t>(input.size(1))}));
      nvte_output_list.emplace_back(
          make_tensor(d_output_ptr, {out_rows, static_cast<size_t>(output.size(1))}));
      padded_num_rows_list.emplace_back(static_cast<int>(out_rows));

      char* in_char = reinterpret_cast<char*>(d_input_ptr);
      in_char += in_rows * input.size(1) * input.element_size();
      d_input_ptr = reinterpret_cast<void*>(in_char);
      char* out_char = reinterpret_cast<char*>(d_output_ptr);
      out_char += out_rows * output.size(1) * output.element_size();
      d_output_ptr = reinterpret_cast<void*>(out_char);
    }
  }

  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");
  NVTE_CHECK(padded_num_rows_list.size() == nvte_input_list.size(),
             "Number of input and padded row list must match");

  NVTE_SCOPED_GIL_RELEASE({
    if (has_amax_in) {
      nvte_multi_padding_with_amax(
          nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(), amax_in_ptr,
          amax_out_ptr, padded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
    } else {
      nvte_multi_padding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                         padded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
    }
  });
}

void fused_multi_row_unpadding(at::Tensor input, at::Tensor output,
                               std::vector<size_t> input_row_list,
                               std::vector<size_t> unpadded_input_row_list) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input_row_list.size() == unpadded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal  2.");

  const auto num_tensors = input_row_list.size();
  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<transformer_engine::DType> input_type_list;
  void* d_input_ptr = reinterpret_cast<void*>(input.data_ptr());
  void* d_output_ptr = reinterpret_cast<void*>(output.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    // Move the input pointer to the next split.
    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset =
        input_row_list[tensor_id] * input.size(1) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back({input_row_list[tensor_id], static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    // Move the output pointer to the next split.
    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset =
        unpadded_input_row_list[tensor_id] * output.size(1) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back(
        {unpadded_input_row_list[tensor_id], static_cast<size_t>(output.size(1))});
  }

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> unpadded_num_rows_list;
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_list.emplace_back(
        make_tensor(input_dptr_list[i], input_shape_list[i], input_type_list[i]));
    nvte_output_list.emplace_back(
        make_tensor(output_dptr_list[i], output_shape_list[i], input_type_list[i]));
    unpadded_num_rows_list.emplace_back(unpadded_input_row_list[i]);
  }

  // Check tensor lists
  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");
  NVTE_CHECK(unpadded_num_rows_list.size() == nvte_input_list.size() &&
             "Number of input and padded row list must match");

  // Launch TE kernel
  nvte_multi_unpadding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                       unpadded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
