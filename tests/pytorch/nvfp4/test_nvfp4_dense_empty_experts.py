# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Empty-expert support for dense NVFP4 bulk quantize + dense grouped GEMM.

Two paths for bitwise compare harnesses:

* **legacy**: filter ``m_splits`` zeros, compact packed B/SF/alpha to active
  experts, call TE with all-positive splits (pre-change contract).
* **native**: pass ``m_splits`` with zeros through; TE skips empty groups.

Set ``NVTE_NVFP4_DENSE_REJECT_EMPTY=1`` to force the old assert (sanity only).
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import pytest
import torch

import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


_GATED = pytest.mark.skipif(
    (not _has_sm100())
    or (not hasattr(tex, "nvfp4_per_token_group_quantize_bulk_dense"))
    or (not hasattr(tex, "nvfp4_cutlass_grouped_per_token_gemm_dense")),
    reason="needs SM100 + dense NVFP4 bulk quantize/GEMM bindings",
)


def _positive_splits(m_splits: Sequence[int]) -> List[int]:
    return [int(m) for m in m_splits if int(m) > 0]


def _active_indices(m_splits: Sequence[int]) -> List[int]:
    return [i for i, m in enumerate(m_splits) if int(m) > 0]


def _compact_packed_rows(tensor: torch.Tensor, m_splits: Sequence[int], rows_per: int) -> torch.Tensor:
    active = _active_indices(m_splits)
    if len(active) == len(m_splits):
        return tensor
    return torch.cat([tensor[i * rows_per : (i + 1) * rows_per] for i in active], dim=0)


def dense_quantize(
    x: torch.Tensor,
    m_splits: Sequence[int],
    *,
    mode: str = "native",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rowwise dense bulk quantize.

    mode:
      - ``native``: pass ``m_splits`` as-is (may contain zeros)
      - ``legacy``: filter zeros before the TE call
    """
    assert mode in ("native", "legacy")
    splits = list(m_splits) if mode == "native" else _positive_splits(m_splits)
    if not splits:
        return (
            x.new_empty((0, x.size(1) // 2), dtype=torch.uint8),
            x.new_empty((0, x.size(1) // 16), dtype=torch.uint8),
            x.new_empty((0,), dtype=torch.float32),
        )
    q, sf, amax, _, _, _ = tex.nvfp4_per_token_group_quantize_bulk_dense(
        x.contiguous(),
        splits,
        True,
        False,
        with_swizzle=True,
        do_amax=True,
        row_amax=None,
    )
    return q, sf, amax


def dense_gemm(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    a_alpha: torch.Tensor,
    b_alpha: torch.Tensor,
    m_splits: Sequence[int],
    n_cols: int,
    *,
    mode: str = "native",
) -> torch.Tensor:
    """Dense grouped TN GEMM. ``mode=legacy`` compacts B to active experts."""
    assert mode in ("native", "legacy")
    out = torch.empty(a_q.size(0), n_cols, device=a_q.device, dtype=torch.bfloat16)
    if mode == "legacy":
        splits = _positive_splits(m_splits)
        if not splits:
            out.zero_()
            return out
        b_q = _compact_packed_rows(b_q, m_splits, n_cols)
        if b_sf.dim() > 1:
            b_sf = _compact_packed_rows(b_sf, m_splits, n_cols).reshape(-1)
        else:
            b_sf = _compact_packed_rows(b_sf, m_splits, b_sf.numel() // max(len(m_splits), 1))
        b_alpha = _compact_packed_rows(b_alpha.reshape(-1), m_splits, n_cols)
        m_use = splits
    else:
        m_use = list(m_splits)

    tex.nvfp4_cutlass_grouped_per_token_gemm_dense(
        a_q,
        b_q,
        a_sf.reshape(-1),
        b_sf.reshape(-1),
        a_alpha,
        b_alpha,
        out,
        None,
        None,
        None,
        None,
        m_use,
        False,
        "default",
    )
    return out


def compare_dense_empty_expert_paths(
    x: torch.Tensor,
    w: torch.Tensor,
    m_splits: Sequence[int],
    n_cols: int,
) -> dict:
    """Run legacy vs native quantize+GEMM and report bitwise equality.

    ``x`` is token-compact ``[sum(m_splits), K]``.
    ``w`` is full expert pack ``[G * n_cols, K]``.
    """
    aq_l, as_l, aa_l = dense_quantize(x, m_splits, mode="legacy")
    aq_n, as_n, aa_n = dense_quantize(x, m_splits, mode="native")
    wq, ws, wa = dense_quantize(w, [n_cols] * len(m_splits), mode="native")

    out_l = dense_gemm(aq_l, wq, as_l, ws, aa_l, wa, m_splits, n_cols, mode="legacy")
    out_n = dense_gemm(aq_n, wq, as_n, ws, aa_n, wa, m_splits, n_cols, mode="native")

    return {
        "quant_q_equal": torch.equal(aq_l, aq_n),
        "quant_sf_equal": torch.equal(as_l, as_n),
        "quant_amax_equal": torch.equal(aa_l, aa_n),
        "gemm_out_equal": torch.equal(out_l, out_n),
        "gemm_out_max_abs_diff": float((out_l.float() - out_n.float()).abs().max().item())
        if out_l.numel()
        else 0.0,
        "out_legacy": out_l,
        "out_native": out_n,
    }


@_GATED
@pytest.mark.parametrize(
    "m_splits",
    [
        [128, 128],
        [128, 0, 128],
        [0, 256],
        [256, 0],
        [128, 0, 0, 128],
    ],
)
def test_dense_empty_experts_quantize_and_gemm_bitwise(m_splits):
    prev = os.environ.pop("NVTE_NVFP4_DENSE_REJECT_EMPTY", None)
    try:
        torch.manual_seed(0)
        k, n = 128, 128
        sum_m = sum(m_splits)
        x = torch.randn(sum_m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(len(m_splits) * n, k, device="cuda", dtype=torch.bfloat16)
        report = compare_dense_empty_expert_paths(x, w, m_splits, n)
        assert report["quant_q_equal"], "legacy vs native quantize q mismatch"
        assert report["quant_sf_equal"], "legacy vs native quantize sf mismatch"
        assert report["quant_amax_equal"], "legacy vs native quantize amax mismatch"
        assert report["gemm_out_equal"], (
            f"legacy vs native GEMM mismatch max_abs={report['gemm_out_max_abs_diff']}"
        )
    finally:
        if prev is not None:
            os.environ["NVTE_NVFP4_DENSE_REJECT_EMPTY"] = prev


@_GATED
def test_dense_reject_empty_env_restores_assert():
    os.environ["NVTE_NVFP4_DENSE_REJECT_EMPTY"] = "1"
    try:
        x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="must be > 0"):
            tex.nvfp4_per_token_group_quantize_bulk_dense(
                x, [128, 0, 128], True, False, with_swizzle=True, do_amax=True, row_amax=None
            )
    finally:
        os.environ.pop("NVTE_NVFP4_DENSE_REJECT_EMPTY", None)
