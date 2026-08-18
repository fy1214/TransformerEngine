/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file padding.h
 *  \brief Functions handling padding.
 */

#ifndef TRANSFORMER_ENGINE_PADDING_H_
#define TRANSFORMER_ENGINE_PADDING_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Padding multiple tensors.
 *
 *  NOTE: Padding mode only support bottom.
 *
 *  For example, 3x3 matrix pad to 4x3 matrix.
 *
 *  source
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *
 *  destination
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *  | 0 | 0 | 0 |
 *
 *  \param[in]     num_tensors              Number of tensors.
 *  \param[in]     input_list               List of 2D input tensors.
 *  \param[in,out] output_list              List of padded tensors. Dimensions
 *                                          match tensors in input_list.
 *  \param[in]     padded_num_rows_list     List of padded num rows corresponding to input tensors.
 *  \param[in]     stream                   CUDA stream used for the operation.
 */
void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream);

/*! \brief Padding multiple tensors together with companion FP32 per-row amax.
 *
 *  Same bottom-row padding as nvte_multi_padding, and additionally copies or
 *  zero-pads a matching FP32 amax vector in the same kernel launch.
 *  ``amax_input`` / ``amax_output`` are contiguous flat buffers laid out in the
 *  same expert order as ``input_list`` (one float per row). Does not modify
 *  ``nvte_multi_padding``.
 *
 *  \param[in]     num_tensors              Number of tensors.
 *  \param[in]     input_list               List of 2D input tensors.
 *  \param[in,out] output_list              List of padded tensors.
 *  \param[in]     amax_input               Contiguous FP32 amax, length sum(input rows).
 *  \param[in,out] amax_output              Contiguous FP32 amax, length sum(padded rows).
 *  \param[in]     padded_num_rows_list     List of padded num rows corresponding to input tensors.
 *  \param[in]     stream                   CUDA stream used for the operation.
 */
void nvte_multi_padding_with_amax(size_t num_tensors, const NVTETensor* input_list,
                                  NVTETensor* output_list, const float* amax_input,
                                  float* amax_output, const int* padded_num_rows_list,
                                  cudaStream_t stream);

/*! \brief Unpadding multiple tensors (reverse operation of padding).
 *
 *  NOTE: Unpadding mode only removes bottom rows.
 *
 *  For example, 4x3 matrix unpad to 3x3 matrix.
 *
 *  source
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *  | 0 | 0 | 0 |
 *
 *  destination
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *
 *  \param[in]     num_tensors               Number of tensors.
 *  \param[in]     input_list                List of 2D padded input tensors.
 *  \param[in,out] output_list               List of unpadded tensors. Dimensions
 *                                           match original unpadded tensors.
 *  \param[in]     unpadded_num_rows_list    List of unpadded num rows corresponding to input tensors.
 *  \param[in]     stream                    CUDA stream used for the operation.
 */
void nvte_multi_unpadding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                          const int* unpadded_num_rows_list, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_PADDING_H_
