# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for SwiGLU fused per-token row amax."""

import pytest
import torch

import transformer_engine.pytorch.cpp_extensions as tex


def _ref_swiglu(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(7, 64), (128, 256), (3, 5, 128)])
def test_swiglu_with_row_amax(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(*shape, device="cuda", dtype=dtype)
    y, row_amax = tex.swiglu_with_row_amax(x)

    y_ref = _ref_swiglu(x)
    row_amax_ref = y_ref.detach().float().abs().amax(dim=-1).reshape(-1)

    torch.testing.assert_close(y, y_ref, atol=1e-2 if dtype != torch.float32 else 1e-5,
                               rtol=1e-2 if dtype != torch.float32 else 1e-5)
    torch.testing.assert_close(row_amax, row_amax_ref,
                               atol=1e-2 if dtype != torch.float32 else 1e-5,
                               rtol=1e-2 if dtype != torch.float32 else 1e-5)
