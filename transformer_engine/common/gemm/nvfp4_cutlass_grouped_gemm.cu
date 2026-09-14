/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Grouped (MoE) NVFP4xNVFP4 -> BF16 GEMM with the per-token (per-row * per-col)
// fused EVT epilogue. Single CUTLASS ptr-array grouped launch replaces the
// per-expert Python loop in general_grouped_gemm.
//
// Design mirrors the dense per-token kernel in nvfp4_cutlass_gemm.cu:
//   * one physical layout (A row-major, B col-major, D = A @ B^T row-major);
//     TE's TN/NN/NT directions are realized by the caller choosing rowwise vs
//     columnwise operands, exactly like the dense per-token dispatcher.
//   * the same 3-level Sm90 EVT: D = bf16(1/2688^2 * alpha_a[i] * alpha_b[j] * acc).
// The only structural change vs. dense is ptr-array grouping:
//   * GroupProblemShape<Shape<int,int,int>> + KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
//   * the EVT row/col broadcast leaves take ElementInput_ = float* so CUTLASS
//     switches them to per-group array-of-pointers mode (ptr_col[l] / ptr_row[l]).
//
// Tile dispatch (BF16 overwrite, no bias). gemm_kind is caller-selected;
// K is not an fc1/fc2 heuristic (MoE intermediate sizes differ by model):
//   * DEFAULT: 1-CTA MmaTile=(128,128,256), MMA_N=128.
//   * FC1 + M%256 + N%256: 2-CTA (256,256,256), MMA_N=256.
//     Gated by NVTE_NVFP4_GROUPED_FC1_2SM_N256 (default on).
//   * FC2 + M%256: 2-CTA (256,128,256), MMA_N=128.
//   * FC1 opt-in 1-CTA (128,256,256) via NVTE_NVFP4_GROUPED_FC1_N256 (default off).

#include <transformer_engine/nvfp4_cutlass_gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "../util/system.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace transformer_engine {
namespace nvfp4_cutlass {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace cute_ = cute;
namespace fusion = cutlass::epilogue::fusion;

// ---- Type config (mirrors the dense per-token kernel) ---------------------

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementScale = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = cute_::Shape<cute_::_128, cute_::_128, cute_::_256>;
using ClusterShape = cute_::Shape<cute_::_1, cute_::_1, cute_::_1>;

// Ptr-array (grouped) schedules. NVFP4 = e2m1 data + ue4m3 SF, 1x16 vec.
using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// Per-group problem shape <M, N, K>.
using ProblemShape = cutlass::gemm::GroupProblemShape<cute_::Shape<int, int, int>>;

constexpr cutlass::FloatRoundStyle kRoundStyleFused = cutlass::FloatRoundStyle::round_to_nearest;
// NVFP4 spec constant: 1 / (fp4_max^2 * fp8_max^2) = 1/(6^2 * 448^2).
constexpr float kNvfp4DequantFactor = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);

// ---- Per-token fused EVT, lifted to grouped (array-of-pointers) ------------
// Sm90Col/RowBroadcast instantiate IsArrayOfPointers=true when ElementInput_
// is a pointer type (float*); then ptr_col/ptr_row become float const* const*
// and are indexed per group l. Everything else matches the dense EVT.

using AccFetchNode = fusion::Sm90AccFetch;

using RowScaleNode = fusion::Sm90ColBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput_=*/ElementScale*,  // pointer type -> per-group ptr array
    /*ElementCompute=*/ElementAccumulator>;

using ColScaleNode = fusion::Sm90RowBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput_=*/ElementScale*,  // pointer type -> per-group ptr array
    /*ElementCompute=*/ElementAccumulator>;

// Uniform NVFP4 dequant constant (same for all groups).
using ConstScaleNode = fusion::Sm90ScalarBroadcast<ElementScale>;

// L1: tmp1 = alpha_a[i] * acc.
using MulAccByRowEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                           ElementAccumulator, kRoundStyleFused>,
                                       RowScaleNode, AccFetchNode>;
// L2: tmp2 = alpha_b[j] * tmp1.
using MulByColEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                        ElementAccumulator, kRoundStyleFused>,
                                    ColScaleNode, MulAccByRowEVT>;
// L3: D = bf16(NVFP4_DEQUANT_K * tmp2).
using FusedEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementAccumulator, kRoundStyleFused>,
    ConstScaleNode, MulByColEVT>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutDTag*, AlignmentD, EpilogueSchedule,
    FusedEVT>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ElementADataT = typename ElementA::DataType;
using ElementBDataT = typename ElementB::DataType;
using ElementSFT = typename ElementA::ScaleFactorType;

// ---- fp32-output accumulate variant (grouped wgrad into fp32 main_grad) -----
// D_g = float(beta * C_g + NVFP4_DEQUANT_K * alpha_a_g[i] * alpha_b_g[j] * acc).
// Reuses the per-group ptr-array scale subtree; only the output element type and
// the beta*C add differ from the overwrite EVT. beta == 0 skips the C load.
using ElementCAcc = float;
using ElementDAcc = float;
constexpr int AlignmentCAcc = 128 / cutlass::sizeof_bits<ElementCAcc>::value;
constexpr int AlignmentDAcc = 128 / cutlass::sizeof_bits<ElementDAcc>::value;

// Z = NVFP4_DEQUANT_K * alpha_b[j] * (alpha_a[i] * acc), in fp32.
using ScaledAccEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                         ElementAccumulator, kRoundStyleFused>,
                                     ConstScaleNode, MulByColEVT>;
using BetaNode = fusion::Sm90ScalarBroadcast<ElementScale>;
using AccumEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementDAcc,
                                                     ElementAccumulator, kRoundStyleFused>,
                                 BetaNode, fusion::Sm90SrcFetch<ElementCAcc>, ScaledAccEVT>;

using CollectiveEpilogueAcc = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementCAcc, LayoutCTag*, AlignmentCAcc, ElementDAcc, LayoutDTag*, AlignmentDAcc,
    EpilogueSchedule, AccumEVT>::CollectiveOp;

using CollectiveMainloopAcc = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueAcc::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernelAcc = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopAcc,
                                                           CollectiveEpilogueAcc>;
using GemmAcc = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAcc>;

// ---- bias-fused overwrite variant (fprop): D = bf16(bias[n] + Z) -----------
// Bias is a per-output-channel (N) vector broadcast down rows -- the SAME
// broadcast family as alpha_b (ColScaleNode = Sm90RowBroadcast). Fed as FP32
// per-group pointer arrays to mirror the alpha arrays exactly: bias is tiny, so
// the host-side bf16->fp32 cast is negligible and this dodges any converting-
// load edge cases in the array-of-pointers broadcast. Reuses the fp32 Z subtree
// (ScaledAccEVT); only the top-level (bias + Z) add and the bf16 cast are new.
using BiasNode = fusion::Sm90RowBroadcast<
    /*Stages=*/0, MmaTileShape, /*ElementInput_=*/ElementScale*,
    /*ElementCompute=*/ElementAccumulator>;
// D = bf16(bias[n] + Z), Z (fp32) = NVFP4_DEQUANT_K * alpha_b[j] * alpha_a[i] * acc.
using FusedBiasEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::plus, ElementD, ElementAccumulator, kRoundStyleFused>, BiasNode,
    ScaledAccEVT>;

using CollectiveEpilogueBias = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutDTag*, AlignmentD, EpilogueSchedule,
    FusedBiasEVT>::CollectiveOp;
using CollectiveMainloopBias = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueBias::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;
using GemmKernelBias = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopBias,
                                                            CollectiveEpilogueBias>;
using GemmBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelBias>;

// BF16 overwrite kernel parameterized by tile / cluster / schedule. EVT graph
// is identical to FusedEVT above; only CtaTileShape and 1SM vs 2SM change.
template <class MmaTile, class Cluster, class MainSched, class EpiSched>
struct GroupedBf16Overwrite {
  using RowScale = fusion::Sm90ColBroadcast<
      /*Stages=*/0, MmaTile, /*ElementInput_=*/ElementScale*,
      /*ElementCompute=*/ElementAccumulator>;
  using ColScale = fusion::Sm90RowBroadcast<
      /*Stages=*/0, MmaTile, /*ElementInput_=*/ElementScale*,
      /*ElementCompute=*/ElementAccumulator>;
  using ConstScale = fusion::Sm90ScalarBroadcast<ElementScale>;
  using MulAccByRow = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                          ElementAccumulator, kRoundStyleFused>,
                                      RowScale, AccFetchNode>;
  using MulByCol = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                       ElementAccumulator, kRoundStyleFused>,
                                   ColScale, MulAccByRow>;
  using FusedEVT = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementAccumulator, kRoundStyleFused>,
      ConstScale, MulByCol>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag*, AlignmentC, ElementD,
      LayoutDTag*, AlignmentD, EpiSched, FusedEVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
      ElementAccumulator, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// fc2: 2SM cluster, MMA_N stays 128 (short K cannot hide OverlappingAccum).
using Kernel2SmN128 = GroupedBf16Overwrite<
    cute_::Shape<cute_::_256, cute_::_128, cute_::_256>,
    cute_::Shape<cute_::_2, cute_::_1, cute_::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;
// fc1 opt-in: 1SM MMA_N=256. Off by default (NVTE_NVFP4_GROUPED_FC1_N256).
using Kernel1SmN256 = GroupedBf16Overwrite<
    cute_::Shape<cute_::_128, cute_::_256, cute_::_256>,
    cute_::Shape<cute_::_1, cute_::_1, cute_::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;
// fc1: 2SM MMA_N=256 (CUTLASS example 75 tile). Needs long K for reverse-epi.
using Kernel2SmN256 = GroupedBf16Overwrite<
    cute_::Shape<cute_::_256, cute_::_256, cute_::_256>,
    cute_::Shape<cute_::_2, cute_::_1, cute_::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;

// Query the SM count exactly once (cudaGetDeviceProperties is very slow and was
// adding a ~ms fixed cost to every grouped launch).
static int cached_sm_count() {
  static int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return sm;
}

static inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }

// Process-persistent device buffers reused across launches, to avoid the
// per-call cudaMalloc/cudaFree churn that dominated runtime for small grouped
// GEMMs. which=0 -> metadata scratch, which=1 -> CUTLASS workspace.
// Assumes grouped GEMMs are issued serially on one stream (the TE norm); the
// stream-ordered free on regrow keeps it safe under that assumption.
static void* persistent_buffer(size_t bytes, cudaStream_t stream, int which) {
  static void* bufs[2] = {nullptr, nullptr};
  static size_t caps[2] = {0, 0};
  if (bytes > caps[which]) {
    if (bufs[which] != nullptr) {
      NVTE_CHECK_CUDA(cudaFreeAsync(bufs[which], stream));
    }
    const size_t newcap = bytes + bytes / 2;  // slack to avoid frequent regrows
    NVTE_CHECK_CUDA(cudaMallocAsync(&bufs[which], newcap, stream));
    caps[which] = newcap;
  }
  return bufs[which];
}

// Process-persistent PAGEABLE host staging buffer, mirror of the device metadata
// scratch. All per-group arrays are packed into it host-side, then shipped in a
// SINGLE H2D copy (vs. one cudaMemcpyAsync per array). Deliberately pageable (not
// pinned): cudaMemcpyAsync from pageable host memory copies the source into the
// driver staging buffer before returning, so the buffer is safe to overwrite on
// the next call even when the host runs ahead of the stream (training). A pinned
// + truly-async buffer would need event-guarded reuse to avoid that race.
static void* persistent_host_buffer(size_t bytes) {
  static std::vector<uint8_t> buf;
  if (buf.size() < bytes) {
    buf.resize(bytes + bytes / 2);  // slack to avoid frequent regrows
  }
  return buf.data();
}

// A/B switch: batched single H2D of all metadata (default) vs. the legacy
// one-cudaMemcpyAsync-per-array path. Set NVTE_NVFP4_GROUPED_BATCHED_H2D=0 to
// measure the per-array baseline against the batched copy in the same build.
static bool use_batched_h2d() {
  static bool v = transformer_engine::getenv<bool>("NVTE_NVFP4_GROUPED_BATCHED_H2D", true);
  return v;
}

static bool all_aligned_256(const std::vector<int>& xs) {
  if (xs.empty()) return false;
  bool saw_nonzero = false;
  for (int x : xs) {
    // Empty experts (M==0) are skipped from the CUTLASS launch; ignore them for
    // alternate-tile predicates so dense MoE packs with holes still select tiles
    // based on the active groups.
    if (x == 0) continue;
    saw_nonzero = true;
    if (x % 256 != 0) return false;
  }
  return saw_nonzero;
}

// Alternate-tile predicate: RequiredKind + which extents must be 256-aligned.
// enabled is the env gate (fc2 is always on; fc1 2SM default on, 1SM N=256 off).
template <NVTENvfp4GroupedGemmKind RequiredKind, bool CheckM, bool CheckN>
static bool should_use_alt_tile(NVTENvfp4GroupedGemmKind kind,
                                [[maybe_unused]] const std::vector<int>& Ms,
                                [[maybe_unused]] const std::vector<int>& Ns, bool enabled) {
  if (kind != RequiredKind || !enabled) return false;
  if constexpr (CheckM) {
    if (!all_aligned_256(Ms)) return false;
  }
  if constexpr (CheckN) {
    if (!all_aligned_256(Ns)) return false;
  }
  return true;
}

// Build the shared Z-subtree EVT arguments:
//   Z = NVFP4_DEQUANT_K * alpha_b[j] * (alpha_a[i] * acc).
// ArgsT is FusedEVT::Arguments (overwrite path) or ScaledAccEVT::Arguments
// (accumulate path). The two are structurally identical aggregates but distinct
// C++ types (the enclosing Sm90EVT differs only in its top-level output element),
// so the target type must be named explicitly per call site. aa_d/ab_d are the
// per-group device pointer arrays consumed by the array-of-pointers broadcasts.
template <class ArgsT, class P>
static ArgsT make_z_args(P aa_d, P ab_d) {
  // clang-format off
  return ArgsT{
      {/*scalars=*/{kNvfp4DequantFactor}, /*scalar_ptrs=*/{nullptr}, /*dScalar=*/{}},
      {
          {ab_d, /*null_default=*/ElementScale{0}, /*dRow=*/{}},
          {
              {aa_d, /*null_default=*/ElementScale{0}, /*dCol=*/{}},
              {},  // AccFetch
              {},  // multiplies
          },
          {},  // multiplies
      },
      {},  // multiplies
  };
  // clang-format on
}

// Core launcher. All *_ptrs are host vectors of device pointers (length G);
// Ms/Ns/Ks are host per-group extents. SFs must already be swizzled.
// Shared workspace-alloc + launch tail for both output paths.
template <class GemmT>
static void run_grouped_gemm(GemmT& gemm, typename GemmT::Arguments& args, int G,
                             cudaStream_t stream) {
  size_t workspace_size = GemmT::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace = persistent_buffer(workspace_size, stream, /*which=*/1);
  }

  cutlass::Status status = gemm.can_implement(args);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-token GEMM cannot implement: ",
             cutlassGetStatusString(status), " (num_groups=", G, ")");

  status = gemm.initialize(args, workspace, stream);
  NVTE_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS NVFP4 grouped per-token GEMM initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-token GEMM run failed: ", cutlassGetStatusString(status));

  // No per-call frees: scratch + workspace live in persistent_buffer and are
  // reused across launches (and grow on demand).
}

// Accumulate=false -> overwrite, ElementD=bf16. Accumulate=true -> fp32 output
// with D += beta * C (beta=1 accumulates into main_grad, beta=0 overwrites).
// OverwriteGemmT / OverwriteEVTT / ClusterM select an alternate tile for the
// bf16 overwrite path (fc1/fc2). Accumulate and fused-bias stay on the default
// 1SM (128,128,256) kernel.
template <bool Accumulate, class OverwriteGemmT = Gemm, class OverwriteEVTT = FusedEVT,
          int ClusterM = 1>
static void run_cutlass_grouped_per_token_gemm_impl(
    const std::vector<const void*>& a_data_ptrs, const std::vector<const void*>& b_data_ptrs,
    const std::vector<const void*>& a_sf_ptrs, const std::vector<const void*>& b_sf_ptrs,
    const std::vector<const float*>& alpha_a_ptrs, const std::vector<const float*>& alpha_b_ptrs,
    const std::vector<const float*>& bias_ptrs, const std::vector<void*>& d_ptrs,
    const std::vector<int>& Ms, const std::vector<int>& Ns, const std::vector<int>& Ks, float beta,
    cudaStream_t stream) {
  using GemmT = std::conditional_t<Accumulate, GemmAcc, OverwriteGemmT>;
  using StrideAT = typename GemmT::GemmKernel::InternalStrideA;
  using StrideBT = typename GemmT::GemmKernel::InternalStrideB;
  using StrideCT = typename GemmT::GemmKernel::InternalStrideC;
  using StrideDT = typename GemmT::GemmKernel::InternalStrideD;
  using LayoutSFAT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFBT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using BlkCfgT = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementDT = std::conditional_t<Accumulate, ElementDAcc, ElementD>;

  const int G = static_cast<int>(Ms.size());

  // Host-side per-group metadata.
  std::vector<typename ProblemShape::UnderlyingProblemShape> problems(G);
  std::vector<StrideAT> stride_A_h(G);
  std::vector<StrideBT> stride_B_h(G);
  std::vector<StrideCT> stride_C_h(G);
  std::vector<StrideDT> stride_D_h(G);
  std::vector<LayoutSFAT> layout_SFA_h(G);
  std::vector<LayoutSFBT> layout_SFB_h(G);

  std::vector<const ElementADataT*> a_ptr_h(G);
  std::vector<const ElementBDataT*> b_ptr_h(G);
  std::vector<const ElementSFT*> sfa_ptr_h(G);
  std::vector<const ElementSFT*> sfb_ptr_h(G);
  std::vector<ElementDT*> d_ptr_h(G);

  for (int g = 0; g < G; ++g) {
    const int M = Ms[g], N = Ns[g], K = Ks[g];
    problems[g] = {M, N, K};
    stride_A_h[g] = cutlass::make_cute_packed_stride(StrideAT{}, {M, K, 1});
    stride_B_h[g] = cutlass::make_cute_packed_stride(StrideBT{}, {N, K, 1});
    stride_C_h[g] = cutlass::make_cute_packed_stride(StrideCT{}, {M, N, 1});
    stride_D_h[g] = cutlass::make_cute_packed_stride(StrideDT{}, {M, N, 1});
    layout_SFA_h[g] = BlkCfgT::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
    layout_SFB_h[g] = BlkCfgT::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));

    a_ptr_h[g] = reinterpret_cast<const ElementADataT*>(a_data_ptrs[g]);
    b_ptr_h[g] = reinterpret_cast<const ElementBDataT*>(b_data_ptrs[g]);
    sfa_ptr_h[g] = reinterpret_cast<const ElementSFT*>(a_sf_ptrs[g]);
    sfb_ptr_h[g] = reinterpret_cast<const ElementSFT*>(b_sf_ptrs[g]);
    d_ptr_h[g] = reinterpret_cast<ElementDT*>(d_ptrs[g]);
  }

  // Mirror all per-group metadata to device through ONE persistent scratch
  // buffer (one H2D copy per array, zero per-call cudaMalloc/Free). All arrays
  // are O(G) and tiny; 256B sub-alignment is safe for every cute POD type here.
  const size_t need = align256(problems.size() * sizeof(problems[0])) +
                      align256(stride_A_h.size() * sizeof(StrideAT)) +
                      align256(stride_B_h.size() * sizeof(StrideBT)) +
                      align256(stride_C_h.size() * sizeof(StrideCT)) +
                      align256(stride_D_h.size() * sizeof(StrideDT)) +
                      align256(layout_SFA_h.size() * sizeof(LayoutSFAT)) +
                      align256(layout_SFB_h.size() * sizeof(LayoutSFBT)) +
                      align256(a_ptr_h.size() * sizeof(a_ptr_h[0])) +
                      align256(b_ptr_h.size() * sizeof(b_ptr_h[0])) +
                      align256(sfa_ptr_h.size() * sizeof(sfa_ptr_h[0])) +
                      align256(sfb_ptr_h.size() * sizeof(sfb_ptr_h[0])) +
                      align256(d_ptr_h.size() * sizeof(d_ptr_h[0])) +
                      align256(alpha_a_ptrs.size() * sizeof(alpha_a_ptrs[0])) +
                      align256(alpha_b_ptrs.size() * sizeof(alpha_b_ptrs[0])) +
                      (bias_ptrs.empty() ? 0 : align256(bias_ptrs.size() * sizeof(bias_ptrs[0])));
  const bool batched = use_batched_h2d();
  uint8_t* scr = static_cast<uint8_t*>(persistent_buffer(need, stream, /*which=*/0));
  uint8_t* hscr = batched ? static_cast<uint8_t*>(persistent_host_buffer(need)) : nullptr;
  size_t off = 0;
  // Batched path: pack each array into the pageable host mirror at its 256B-aligned
  // slot and defer to ONE H2D after all puts. Legacy path: copy each array to the
  // device scratch immediately. Either way return the DEVICE pointer (scr + off).
  auto put = [&](const auto& vec) {
    using T = typename std::decay_t<decltype(vec)>::value_type;
    const size_t bytes = vec.size() * sizeof(T);
    T* p = reinterpret_cast<T*>(scr + off);
    if (batched) {
      std::memcpy(hscr + off, vec.data(), bytes);
    } else {
      NVTE_CHECK_CUDA(cudaMemcpyAsync(p, vec.data(), bytes, cudaMemcpyHostToDevice, stream));
    }
    off += align256(bytes);
    return p;
  };
  auto* problems_d = put(problems);
  auto* stride_A_d = put(stride_A_h);
  auto* stride_B_d = put(stride_B_h);
  auto* stride_C_d = put(stride_C_h);
  auto* stride_D_d = put(stride_D_h);
  auto* layout_SFA_d = put(layout_SFA_h);
  auto* layout_SFB_d = put(layout_SFB_h);
  auto* a_ptr_d = put(a_ptr_h);
  auto* b_ptr_d = put(b_ptr_h);
  auto* sfa_ptr_d = put(sfa_ptr_h);
  auto* sfb_ptr_d = put(sfb_ptr_h);
  auto* d_ptr_d = put(d_ptr_h);
  // Per-token outer-scale ptr arrays (consumed by the array-of-pointers EVT).
  auto* alpha_a_d = put(alpha_a_ptrs);
  auto* alpha_b_d = put(alpha_b_ptrs);
  // Optional per-group bias ptr array (fprop bf16 overwrite path only).
  const float** bias_d = bias_ptrs.empty() ? nullptr : put(bias_ptrs);

  // Batched path only: one H2D for ALL per-group metadata (off == packed bytes).
  // Stream-ordered before the GEMM consumes the device pointers returned above.
  if (batched) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(scr, hscr, off, cudaMemcpyHostToDevice, stream));
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cached_sm_count();
  if constexpr (ClusterM == 2) {
    hw_info.cluster_shape = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  GemmT gemm;
  if constexpr (Accumulate) {
    // D = float(beta * C + Z). beta == 0 skips the C load (uninitialized D safe);
    // beta == 1 accumulates in place (ptr_C aliases ptr_D == main_grad).
    typename AccumEVT::Arguments fusion_args{
        {/*scalars=*/{beta}, /*scalar_ptrs=*/{nullptr}, /*dScalar=*/{}},      // beta
        {},                                                                   // C source fetch
        make_z_args<typename ScaledAccEVT::Arguments>(alpha_a_d, alpha_b_d),  // Z subtree
        {},                                                                   // multiply_add
    };
    // ptr_C aliases ptr_D (== main_grad). The epilogue wants ElementC const**;
    // d_ptr_d is ElementCAcc** (non-const), so round-trip through void* to add
    // the inner const (a direct reinterpret_cast would reject the qualifier change).
    auto* c_ptr_d = reinterpret_cast<const ElementCAcc**>(reinterpret_cast<void*>(d_ptr_d));
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d,
         layout_SFB_d},
        {fusion_args, /*ptr_C=*/c_ptr_d, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G, stream);
  } else if (bias_d == nullptr) {
    // Overwrite path: D = bf16(Z).
    typename OverwriteEVTT::Arguments fusion_args =
        make_z_args<typename OverwriteEVTT::Arguments>(alpha_a_d, alpha_b_d);
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d,
         layout_SFB_d},
        {fusion_args, /*ptr_C=*/nullptr, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G, stream);
  } else {
    // Bias-fused overwrite path: D = bf16(bias[n] + Z). Same metadata/strides
    // as the plain overwrite path (only the epilogue EVT differs), so the
    // device pointer arrays packed above are reused as-is.
    typename FusedBiasEVT::Arguments fusion_args{
        {bias_d, /*null_default=*/ElementScale{0}, /*dRow=*/{}},
        make_z_args<typename ScaledAccEVT::Arguments>(alpha_a_d, alpha_b_d),
        {},  // plus
    };
    GemmBias gemm_bias;
    typename GemmBias::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d,
         layout_SFB_d},
        {fusion_args, /*ptr_C=*/nullptr, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm_bias, args, G, stream);
  }
}

// ---- Dense / contiguous-offset path (device-side metadata fill) ------------
// Mirrors miniTE gemm_grouped_cutlass_v2_evt fill_grouped_gemm_args_kernel:
// operands stay concatenated; ptr/stride/layout/problem tables are written on
// device so the host list→H2D path is skipped. Still launches the same
// PtrArray TMA NVFP4 GemmUniversal instances (and TE dual-alpha EVT).

// Exclusive prefix tables for dense MoE packing (one thread per group):
//   a_row[g+1] = sum_{i<=g} M_i
//   b_row[g+1] = (g+1) * N
//   a_sf[g+1]  = a_row[g+1] * k_sf     (k_sf = K/16 SF elems per row)
//   b_sf[g+1]  = b_row[g+1] * k_sf
// Launch: <<<1, G, G*sizeof(int32_t), stream>>>
__global__ void fill_grouped_per_token_dense_prefix_offsets_kernel(
    const int32_t* __restrict__ m_splits, int32_t* __restrict__ a_row,
    int32_t* __restrict__ b_row, int64_t* __restrict__ a_sf, int64_t* __restrict__ b_sf, int G,
    int N, int k_sf) {
  extern __shared__ int32_t smem_m[];
  const int g = static_cast<int>(threadIdx.x);
  if (g < G) {
    smem_m[g] = m_splits[g];
  }
  __syncthreads();

  int32_t excl_m = 0;
  for (int i = 0; i < g && i < G; ++i) {
    excl_m += smem_m[i];
  }

  if (g == 0) {
    a_row[0] = 0;
    b_row[0] = 0;
    a_sf[0] = 0;
    b_sf[0] = 0;
  }
  if (g < G) {
    const int32_t m = smem_m[g];
    const int32_t a_end = excl_m + m;
    const int32_t b_end = (g + 1) * N;
    a_row[g + 1] = a_end;
    b_row[g + 1] = b_end;
    a_sf[g + 1] = static_cast<int64_t>(a_end) * k_sf;
    b_sf[g + 1] = static_cast<int64_t>(b_end) * k_sf;
  }
}

template <class StrideAT, class StrideBT, class StrideCT, class StrideDT, class LayoutSFAT,
          class LayoutSFBT, class BlkCfgT, class ElementDT>
__global__ void fill_grouped_per_token_dense_args_kernel(
    const ElementADataT** ptr_A, const ElementBDataT** ptr_B, const ElementSFT** ptr_SFA,
    const ElementSFT** ptr_SFB, ElementDT** ptr_D, const float** ptr_alpha_a,
    const float** ptr_alpha_b, StrideAT* stride_A, StrideBT* stride_B, StrideCT* stride_C,
    StrideDT* stride_D, LayoutSFAT* layout_SFA, LayoutSFBT* layout_SFB,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes, const void* a_base,
    const void* b_base, const void* sfa_base, const void* sfb_base, void* d_base,
    const float* alpha_a_base, const float* alpha_b_base, const int32_t* a_row_offsets,
    const int32_t* b_row_offsets, const int64_t* a_sf_offsets, const int64_t* b_sf_offsets,
    int K_packed, int K, int N, const int32_t* active_experts, int num_launch_groups) {
  const int slot = static_cast<int>(threadIdx.x);
  if (slot >= num_launch_groups) return;
  // active_experts == nullptr => identity mapping (no empty experts).
  const int g = (active_experts != nullptr) ? active_experts[slot] : slot;
  const int M_g = a_row_offsets[g + 1] - a_row_offsets[g];
  const int N_g = b_row_offsets[g + 1] - b_row_offsets[g];

  // A/B stored as packed FP4 uint8 rows of length K_packed = K/2.
  ptr_A[slot] = reinterpret_cast<const ElementADataT*>(
      static_cast<const char*>(a_base) + static_cast<int64_t>(a_row_offsets[g]) * K_packed);
  ptr_B[slot] = reinterpret_cast<const ElementBDataT*>(
      static_cast<const char*>(b_base) + static_cast<int64_t>(b_row_offsets[g]) * K_packed);
  ptr_SFA[slot] = static_cast<const ElementSFT*>(sfa_base) + a_sf_offsets[g];
  ptr_SFB[slot] = static_cast<const ElementSFT*>(sfb_base) + b_sf_offsets[g];
  ptr_D[slot] = reinterpret_cast<ElementDT*>(
      static_cast<char*>(d_base) +
      static_cast<int64_t>(a_row_offsets[g]) * N * static_cast<int>(sizeof(ElementDT)));
  ptr_alpha_a[slot] = alpha_a_base + a_row_offsets[g];
  ptr_alpha_b[slot] = alpha_b_base + b_row_offsets[g];

  stride_A[slot] = cutlass::make_cute_packed_stride(StrideAT{}, {M_g, K, 1});
  stride_B[slot] = cutlass::make_cute_packed_stride(StrideBT{}, {N_g, K, 1});
  stride_C[slot] = cutlass::make_cute_packed_stride(StrideCT{}, {M_g, N_g, 1});
  stride_D[slot] = cutlass::make_cute_packed_stride(StrideDT{}, {M_g, N_g, 1});

  auto shape_g = cute_::make_shape(M_g, N_g, K, 1);
  layout_SFA[slot] = BlkCfgT::tile_atom_to_shape_SFA(shape_g);
  layout_SFB[slot] = BlkCfgT::tile_atom_to_shape_SFB(shape_g);
  problem_sizes[slot] = cute_::make_shape(M_g, N_g, K);
}

// Dense launch: device fill of ptr-array metadata, then same Gemm as list path.
// Requires uniform N across groups (encoded in D.size(1) / Ns[0]). No bias.
// Prefix offset tables (a_row/b_row/a_sf/b_sf) are built on device from host Ms
// when the caller passes nullptr; otherwise the provided device tables are used.
// Empty experts (Ms[g] == 0) keep their weight slots in the G*N B pack but are
// dropped from the CUTLASS problem list (same semantics as the list GEMM path).
template <bool Accumulate, class OverwriteGemmT = Gemm, class OverwriteEVTT = FusedEVT,
          int ClusterM = 1>
static void run_cutlass_grouped_per_token_gemm_dense_impl(
    const void* a_base, const void* b_base, const void* a_sf_base, const void* b_sf_base,
    const float* alpha_a_base, const float* alpha_b_base, void* d_base,
    const int32_t* a_row_offsets, const int32_t* b_row_offsets, const int64_t* a_sf_offsets,
    const int64_t* b_sf_offsets, const std::vector<int>& Ms, const std::vector<int>& Ns, int K,
    float beta, cudaStream_t stream) {
  using GemmT = std::conditional_t<Accumulate, GemmAcc, OverwriteGemmT>;
  using StrideAT = typename GemmT::GemmKernel::InternalStrideA;
  using StrideBT = typename GemmT::GemmKernel::InternalStrideB;
  using StrideCT = typename GemmT::GemmKernel::InternalStrideC;
  using StrideDT = typename GemmT::GemmKernel::InternalStrideD;
  using LayoutSFAT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFBT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using BlkCfgT = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementDT = std::conditional_t<Accumulate, ElementDAcc, ElementD>;

  const int G = static_cast<int>(Ms.size());
  NVTE_CHECK(G > 0 && G <= 1024, "dense grouped GEMM requires 1 <= num_groups <= 1024, got ", G);
  NVTE_CHECK(K > 0 && (K % 2) == 0, "K must be positive and even (FP4 packed), got ", K);
  NVTE_CHECK((K % 16) == 0, "K must be a multiple of 16 for SF packing, got ", K);
  const int N = Ns[0];
  std::vector<int32_t> active_experts;
  active_experts.reserve(static_cast<size_t>(G));
  for (int g = 0; g < G; ++g) {
    NVTE_CHECK(Ns[g] == N, "dense grouped GEMM requires uniform N across groups; Ns[0]=", N,
               " Ns[", g, "]=", Ns[g]);
    NVTE_CHECK(Ms[g] >= 0, "m_splits[", g, "] must be >= 0");
    if (Ms[g] == 0) continue;
    NVTE_CHECK(Ms[g] % 128 == 0 && Ns[g] % 128 == 0 && K % 128 == 0,
               "group ", g, ": non-empty M/N/K must be multiples of 128");
    active_experts.push_back(static_cast<int32_t>(g));
  }
  const int G_launch = static_cast<int>(active_experts.size());
  if (G_launch == 0) {
    return;
  }
  const bool has_empty = (G_launch != G);
  const int K_packed = K / 2;
  const int k_sf = K / 16;

  const bool need_prefix = (a_row_offsets == nullptr || b_row_offsets == nullptr ||
                            a_sf_offsets == nullptr || b_sf_offsets == nullptr);
  NVTE_CHECK(!need_prefix || (a_row_offsets == nullptr && b_row_offsets == nullptr &&
                              a_sf_offsets == nullptr && b_sf_offsets == nullptr),
             "dense grouped GEMM offsets must be all provided or all nullptr");

  const size_t need =
      align256(static_cast<size_t>(G_launch) * sizeof(typename ProblemShape::UnderlyingProblemShape)) +
      align256(static_cast<size_t>(G_launch) * sizeof(StrideAT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(StrideBT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(StrideCT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(StrideDT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(LayoutSFAT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(LayoutSFBT)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const ElementADataT*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const ElementBDataT*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const ElementSFT*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const ElementSFT*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(ElementDT*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const float*)) +
      align256(static_cast<size_t>(G_launch) * sizeof(const float*)) +
      (has_empty ? align256(static_cast<size_t>(G_launch) * sizeof(int32_t)) : 0) +
      (need_prefix
           ? (align256(static_cast<size_t>(G) * sizeof(int32_t)) +
              align256(static_cast<size_t>(G + 1) * sizeof(int32_t)) +
              align256(static_cast<size_t>(G + 1) * sizeof(int32_t)) +
              align256(static_cast<size_t>(G + 1) * sizeof(int64_t)) +
              align256(static_cast<size_t>(G + 1) * sizeof(int64_t)))
           : 0);
  uint8_t* scr = static_cast<uint8_t*>(persistent_buffer(need, stream, /*which=*/0));
  size_t off = 0;
  auto take = [&](size_t bytes) {
    void* p = scr + off;
    off += align256(bytes);
    return p;
  };
  auto* problems_d =
      static_cast<typename ProblemShape::UnderlyingProblemShape*>(take(
          static_cast<size_t>(G_launch) * sizeof(typename ProblemShape::UnderlyingProblemShape)));
  auto* stride_A_d = static_cast<StrideAT*>(take(static_cast<size_t>(G_launch) * sizeof(StrideAT)));
  auto* stride_B_d = static_cast<StrideBT*>(take(static_cast<size_t>(G_launch) * sizeof(StrideBT)));
  auto* stride_C_d = static_cast<StrideCT*>(take(static_cast<size_t>(G_launch) * sizeof(StrideCT)));
  auto* stride_D_d = static_cast<StrideDT*>(take(static_cast<size_t>(G_launch) * sizeof(StrideDT)));
  auto* layout_SFA_d =
      static_cast<LayoutSFAT*>(take(static_cast<size_t>(G_launch) * sizeof(LayoutSFAT)));
  auto* layout_SFB_d =
      static_cast<LayoutSFBT*>(take(static_cast<size_t>(G_launch) * sizeof(LayoutSFBT)));
  auto* a_ptr_d = static_cast<const ElementADataT**>(
      take(static_cast<size_t>(G_launch) * sizeof(const ElementADataT*)));
  auto* b_ptr_d = static_cast<const ElementBDataT**>(
      take(static_cast<size_t>(G_launch) * sizeof(const ElementBDataT*)));
  auto* sfa_ptr_d = static_cast<const ElementSFT**>(
      take(static_cast<size_t>(G_launch) * sizeof(const ElementSFT*)));
  auto* sfb_ptr_d = static_cast<const ElementSFT**>(
      take(static_cast<size_t>(G_launch) * sizeof(const ElementSFT*)));
  auto* d_ptr_d =
      static_cast<ElementDT**>(take(static_cast<size_t>(G_launch) * sizeof(ElementDT*)));
  auto* alpha_a_d =
      static_cast<const float**>(take(static_cast<size_t>(G_launch) * sizeof(const float*)));
  auto* alpha_b_d =
      static_cast<const float**>(take(static_cast<size_t>(G_launch) * sizeof(const float*)));

  const int32_t* active_experts_d = nullptr;
  if (has_empty) {
    auto* active_scratch =
        static_cast<int32_t*>(take(static_cast<size_t>(G_launch) * sizeof(int32_t)));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(active_scratch, active_experts.data(),
                                    static_cast<size_t>(G_launch) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream));
    active_experts_d = active_scratch;
  }

  const int32_t* a_row_d = a_row_offsets;
  const int32_t* b_row_d = b_row_offsets;
  const int64_t* a_sf_d = a_sf_offsets;
  const int64_t* b_sf_d = b_sf_offsets;
  if (need_prefix) {
    auto* m_splits_d = static_cast<int32_t*>(take(static_cast<size_t>(G) * sizeof(int32_t)));
    auto* a_row_scratch =
        static_cast<int32_t*>(take(static_cast<size_t>(G + 1) * sizeof(int32_t)));
    auto* b_row_scratch =
        static_cast<int32_t*>(take(static_cast<size_t>(G + 1) * sizeof(int32_t)));
    auto* a_sf_scratch =
        static_cast<int64_t*>(take(static_cast<size_t>(G + 1) * sizeof(int64_t)));
    auto* b_sf_scratch =
        static_cast<int64_t*>(take(static_cast<size_t>(G + 1) * sizeof(int64_t)));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(m_splits_d, Ms.data(), static_cast<size_t>(G) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream));
    fill_grouped_per_token_dense_prefix_offsets_kernel<<<1, G, static_cast<size_t>(G) * sizeof(int32_t),
                                                         stream>>>(
        m_splits_d, a_row_scratch, b_row_scratch, a_sf_scratch, b_sf_scratch, G, N, k_sf);
    NVTE_CHECK_CUDA(cudaGetLastError());
    a_row_d = a_row_scratch;
    b_row_d = b_row_scratch;
    a_sf_d = a_sf_scratch;
    b_sf_d = b_sf_scratch;
  }

  fill_grouped_per_token_dense_args_kernel<StrideAT, StrideBT, StrideCT, StrideDT, LayoutSFAT,
                                           LayoutSFBT, BlkCfgT, ElementDT>
      <<<1, G_launch, 0, stream>>>(
          a_ptr_d, b_ptr_d, sfa_ptr_d, sfb_ptr_d, d_ptr_d, alpha_a_d, alpha_b_d, stride_A_d,
          stride_B_d, stride_C_d, stride_D_d, layout_SFA_d, layout_SFB_d, problems_d, a_base,
          b_base, a_sf_base, b_sf_base, d_base, alpha_a_base, alpha_b_base, a_row_d, b_row_d,
          a_sf_d, b_sf_d, K_packed, K, N, active_experts_d, G_launch);
  NVTE_CHECK_CUDA(cudaGetLastError());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cached_sm_count();
  if constexpr (ClusterM == 2) {
    hw_info.cluster_shape = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  GemmT gemm;
  if constexpr (Accumulate) {
    typename AccumEVT::Arguments fusion_args{
        {/*scalars=*/{beta}, /*scalar_ptrs=*/{nullptr}, /*dScalar=*/{}},
        {},
        make_z_args<typename ScaledAccEVT::Arguments>(alpha_a_d, alpha_b_d),
        {},
    };
    auto* c_ptr_d = reinterpret_cast<const ElementCAcc**>(reinterpret_cast<void*>(d_ptr_d));
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G_launch, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d,
         layout_SFB_d},
        {fusion_args, /*ptr_C=*/c_ptr_d, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G_launch, stream);
  } else {
    typename OverwriteEVTT::Arguments fusion_args =
        make_z_args<typename OverwriteEVTT::Arguments>(alpha_a_d, alpha_b_d);
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G_launch, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d,
         layout_SFB_d},
        {fusion_args, /*ptr_C=*/nullptr, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G_launch, stream);
  }
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine

// ---- C API ----------------------------------------------------------------

void nvte_nvfp4_cutlass_grouped_per_token_gemm(int num_groups, const NVTETensor* a_data,
                                               const NVTETensor* b_data, const NVTETensor* a_sf,
                                               const NVTETensor* b_sf, const NVTETensor* alpha_a,
                                               const NVTETensor* alpha_b, NVTETensor* d,
                                               const NVTETensor* bias, bool accumulate,
                                               enum NVTENvfp4GroupedGemmKind gemm_kind,
                                               cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(num_groups > 0, "num_groups must be positive, got ", num_groups);
  NVTE_CHECK(gemm_kind == NVTE_NVFP4_GROUPED_GEMM_DEFAULT ||
                 gemm_kind == NVTE_NVFP4_GROUPED_GEMM_FC1 ||
                 gemm_kind == NVTE_NVFP4_GROUPED_GEMM_FC2,
             "gemm_kind must be DEFAULT, FC1, or FC2, got ", static_cast<int>(gemm_kind));

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::vector<const void*> a_data_ptrs(num_groups), b_data_ptrs(num_groups), a_sf_ptrs(num_groups),
      b_sf_ptrs(num_groups);
  std::vector<const float*> alpha_a_ptrs(num_groups), alpha_b_ptrs(num_groups);
  std::vector<void*> d_ptrs(num_groups);
  std::vector<int> Ms(num_groups), Ns(num_groups), Ks(num_groups);

  // Output dtype must be uniform across groups (one kernel instance per launch).
  // BF16 -> overwrite. FP32 -> accumulate-capable (wgrad into fp32 main_grad).
  const bool d_is_fp32 = convertNVTETensorCheck(d[0])->data.dtype == DType::kFloat32;
  NVTE_CHECK(!accumulate || d_is_fp32,
             "NVFP4 grouped per-token GEMM accumulate=true requires FP32 outputs (main_grad)");

  // Optional fused bias (fprop only): per-group FP32 (N,) vector added in the
  // epilogue. Only valid on the BF16 overwrite path (mutually exclusive with the
  // FP32 accumulate/wgrad path). Empty -> no bias.
  const bool has_bias = bias != nullptr;
  NVTE_CHECK(!has_bias || !d_is_fp32,
             "NVFP4 grouped per-token GEMM bias requires BF16 outputs (fprop overwrite path)");
  std::vector<const float*> bias_ptrs(has_bias ? num_groups : 0);

  for (int g = 0; g < num_groups; ++g) {
    auto* a_t = convertNVTETensorCheck(a_data[g]);
    auto* b_t = convertNVTETensorCheck(b_data[g]);
    auto* sa_t = convertNVTETensorCheck(a_sf[g]);
    auto* sb_t = convertNVTETensorCheck(b_sf[g]);
    auto* aa_t = convertNVTETensorCheck(alpha_a[g]);
    auto* ab_t = convertNVTETensorCheck(alpha_b[g]);
    auto* d_t = convertNVTETensorCheck(d[g]);

    const auto a_shape = a_t->data.shape;
    const auto b_shape = b_t->data.shape;
    const auto d_shape = d_t->data.shape;
    NVTE_CHECK(a_shape.size() == 2, "A[", g, "] must be 2D (M, K)");
    NVTE_CHECK(b_shape.size() == 2, "B[", g, "] must be 2D (N, K)");
    NVTE_CHECK(d_shape.size() == 2, "D[", g, "] must be 2D (M, N)");

    const int M = static_cast<int>(a_shape[0]);
    const int K = static_cast<int>(a_shape[1]);
    const int N = static_cast<int>(b_shape[0]);

    NVTE_CHECK(static_cast<int>(b_shape[1]) == K, "group ", g, ": A.K/B.K mismatch");
    NVTE_CHECK(static_cast<int>(d_shape[0]) == M && static_cast<int>(d_shape[1]) == N, "group ", g,
               ": D shape mismatch");
    NVTE_CHECK(a_t->data.dtype == DType::kFloat4E2M1 && b_t->data.dtype == DType::kFloat4E2M1,
               "group ", g, ": A/B must be FP4 e2m1");
    NVTE_CHECK((d_t->data.dtype == DType::kFloat32) == d_is_fp32, "group ", g,
               ": D dtype must be uniform across groups");
    NVTE_CHECK(d_t->data.dtype == DType::kBFloat16 || d_t->data.dtype == DType::kFloat32, "group ",
               g, ": D must be BF16 or FP32");
    NVTE_CHECK(aa_t->data.dtype == DType::kFloat32 && ab_t->data.dtype == DType::kFloat32, "group ",
               g, ": alpha_a/alpha_b must be FP32");
    NVTE_CHECK(aa_t->data.numel() == static_cast<size_t>(M), "group ", g, ": alpha_a must be (M,)");
    NVTE_CHECK(ab_t->data.numel() == static_cast<size_t>(N), "group ", g, ": alpha_b must be (N,)");
    NVTE_CHECK(M > 0 && N > 0 && K > 0, "group ", g, ": M, N, K must be positive (filter empties)");
    NVTE_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "group ", g,
               ": M, N, K must be multiples of 128 (1-CTA MmaTile = (128,128,256)), got M=", M,
               " N=", N, " K=", K);

    a_data_ptrs[g] = a_t->data.dptr;
    b_data_ptrs[g] = b_t->data.dptr;
    a_sf_ptrs[g] = sa_t->data.dptr;
    b_sf_ptrs[g] = sb_t->data.dptr;
    alpha_a_ptrs[g] = reinterpret_cast<const float*>(aa_t->data.dptr);
    alpha_b_ptrs[g] = reinterpret_cast<const float*>(ab_t->data.dptr);
    d_ptrs[g] = d_t->data.dptr;
    Ms[g] = M;
    Ns[g] = N;
    Ks[g] = K;

    if (has_bias) {
      auto* bias_t = convertNVTETensorCheck(bias[g]);
      NVTE_CHECK(bias_t->data.dtype == DType::kFloat32, "group ", g, ": bias must be FP32");
      NVTE_CHECK(bias_t->data.numel() == static_cast<size_t>(N), "group ", g,
                 ": bias must be (N,)");
      bias_ptrs[g] = reinterpret_cast<const float*>(bias_t->data.dptr);
    }
  }

  static const bool fc1_2sm_n256 =
      transformer_engine::getenv<bool>("NVTE_NVFP4_GROUPED_FC1_2SM_N256", true);
  static const bool fc1_1sm_n256 =
      transformer_engine::getenv<bool>("NVTE_NVFP4_GROUPED_FC1_N256", false);

  if (d_is_fp32) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl</*Accumulate=*/true>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs,
        /*bias_ptrs=*/{}, d_ptrs, Ms, Ns, Ks, /*beta=*/accumulate ? 1.0f : 0.0f, stream);
  } else if (!has_bias &&
             nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC1, /*CheckM=*/true,
                                               /*CheckN=*/true>(gemm_kind, Ms, Ns, fc1_2sm_n256)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel2SmN256::Gemm,
        nvfp4_cutlass::Kernel2SmN256::FusedEVT, /*ClusterM=*/2>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs,
        /*bias_ptrs=*/{}, d_ptrs, Ms, Ns, Ks, /*beta=*/0.0f, stream);
  } else if (!has_bias &&
             nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC2, /*CheckM=*/true,
                                               /*CheckN=*/false>(gemm_kind, Ms, Ns, true)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel2SmN128::Gemm,
        nvfp4_cutlass::Kernel2SmN128::FusedEVT, /*ClusterM=*/2>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs,
        /*bias_ptrs=*/{}, d_ptrs, Ms, Ns, Ks, /*beta=*/0.0f, stream);
  } else if (!has_bias &&
             nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC1, /*CheckM=*/false,
                                               /*CheckN=*/true>(gemm_kind, Ms, Ns, fc1_1sm_n256)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel1SmN256::Gemm,
        nvfp4_cutlass::Kernel1SmN256::FusedEVT, /*ClusterM=*/1>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs,
        /*bias_ptrs=*/{}, d_ptrs, Ms, Ns, Ks, /*beta=*/0.0f, stream);
  } else {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl</*Accumulate=*/false>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs, bias_ptrs,
        d_ptrs, Ms, Ns, Ks, /*beta=*/0.0f, stream);
  }
#else
  NVTE_ERROR(
      "CUTLASS NVFP4 grouped per-token GEMM requires SM100 (Blackwell). Build with "
      "sm_100a/sm_100f.");
#endif
}

void nvte_nvfp4_cutlass_grouped_per_token_gemm_dense(
    int num_groups, const NVTETensor a_data, const NVTETensor b_data, const NVTETensor a_sf,
    const NVTETensor b_sf, const NVTETensor alpha_a, const NVTETensor alpha_b, NVTETensor d,
    const int32_t* a_row_offsets, const int32_t* b_row_offsets, const int64_t* a_sf_offsets,
    const int64_t* b_sf_offsets, const int32_t* m_splits, bool accumulate,
    enum NVTENvfp4GroupedGemmKind gemm_kind, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(num_groups > 0, "num_groups must be positive, got ", num_groups);
  NVTE_CHECK(gemm_kind == NVTE_NVFP4_GROUPED_GEMM_DEFAULT ||
                 gemm_kind == NVTE_NVFP4_GROUPED_GEMM_FC1 ||
                 gemm_kind == NVTE_NVFP4_GROUPED_GEMM_FC2,
             "gemm_kind must be DEFAULT, FC1, or FC2, got ", static_cast<int>(gemm_kind));
  NVTE_CHECK(m_splits != nullptr, "dense grouped GEMM requires host m_splits");
  const bool have_offsets = a_row_offsets != nullptr || b_row_offsets != nullptr ||
                            a_sf_offsets != nullptr || b_sf_offsets != nullptr;
  if (have_offsets) {
    NVTE_CHECK(a_row_offsets != nullptr && b_row_offsets != nullptr && a_sf_offsets != nullptr &&
                   b_sf_offsets != nullptr,
               "dense grouped GEMM offsets must be all provided or all nullptr");
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  auto* a_t = convertNVTETensorCheck(a_data);
  auto* b_t = convertNVTETensorCheck(b_data);
  auto* sa_t = convertNVTETensorCheck(a_sf);
  auto* sb_t = convertNVTETensorCheck(b_sf);
  auto* aa_t = convertNVTETensorCheck(alpha_a);
  auto* ab_t = convertNVTETensorCheck(alpha_b);
  auto* d_t = convertNVTETensorCheck(d);

  NVTE_CHECK(a_t->data.shape.size() == 2 && b_t->data.shape.size() == 2 &&
                 d_t->data.shape.size() == 2,
             "dense grouped GEMM A/B/D must be 2D");
  NVTE_CHECK(a_t->data.dtype == DType::kFloat4E2M1 && b_t->data.dtype == DType::kFloat4E2M1,
             "dense grouped GEMM A/B must be FP4 e2m1");
  NVTE_CHECK(aa_t->data.dtype == DType::kFloat32 && ab_t->data.dtype == DType::kFloat32,
             "dense grouped GEMM alpha_a/alpha_b must be FP32");
  NVTE_CHECK(d_t->data.dtype == DType::kBFloat16 || d_t->data.dtype == DType::kFloat32,
             "dense grouped GEMM D must be BF16 or FP32");

  const bool d_is_fp32 = d_t->data.dtype == DType::kFloat32;
  NVTE_CHECK(!accumulate || d_is_fp32,
             "dense grouped GEMM accumulate=true requires FP32 outputs");

  const int sum_M = static_cast<int>(a_t->data.shape[0]);
  const int K = static_cast<int>(a_t->data.shape[1]);
  const int sum_N = static_cast<int>(b_t->data.shape[0]);
  const int N = static_cast<int>(d_t->data.shape[1]);
  NVTE_CHECK(static_cast<int>(b_t->data.shape[1]) == K, "dense grouped GEMM A.K/B.K mismatch");
  NVTE_CHECK(static_cast<int>(d_t->data.shape[0]) == sum_M, "dense grouped GEMM D.M != A.M");
  NVTE_CHECK(aa_t->data.numel() == static_cast<size_t>(sum_M), "alpha_a must be (sum_M,)");
  NVTE_CHECK(ab_t->data.numel() == static_cast<size_t>(sum_N), "alpha_b must be (sum_N,)");

  std::vector<int> Ms(num_groups), Ns(num_groups, N), Ks(num_groups, K);
  int64_t acc_M = 0;
  // Empty experts allowed unless NVTE_NVFP4_DENSE_REJECT_EMPTY=1 (legacy compare).
  const bool reject_empty =
      transformer_engine::getenv<bool>("NVTE_NVFP4_DENSE_REJECT_EMPTY", false);
  for (int g = 0; g < num_groups; ++g) {
    Ms[g] = static_cast<int>(m_splits[g]);
    NVTE_CHECK(Ms[g] >= 0, "m_splits[", g, "] must be >= 0");
    if (reject_empty) {
      NVTE_CHECK(Ms[g] > 0, "m_splits[", g, "] must be > 0");
    }
    acc_M += Ms[g];
  }
  NVTE_CHECK(acc_M == sum_M, "sum(m_splits)=", acc_M, " must equal A.size(0)=", sum_M);
  NVTE_CHECK(sum_N == N * num_groups,
             "dense grouped GEMM expects B packed as (G*N, K) with uniform N; got sum_N=", sum_N,
             " G*N=", static_cast<int64_t>(N) * num_groups);

  static const bool fc1_2sm_n256 =
      transformer_engine::getenv<bool>("NVTE_NVFP4_GROUPED_FC1_2SM_N256", true);
  static const bool fc1_1sm_n256 =
      transformer_engine::getenv<bool>("NVTE_NVFP4_GROUPED_FC1_N256", false);

  const void* a_base = a_t->data.dptr;
  const void* b_base = b_t->data.dptr;
  const void* a_sf_base = sa_t->data.dptr;
  const void* b_sf_base = sb_t->data.dptr;
  const float* alpha_a_base = reinterpret_cast<const float*>(aa_t->data.dptr);
  const float* alpha_b_base = reinterpret_cast<const float*>(ab_t->data.dptr);
  void* d_base = d_t->data.dptr;

  if (d_is_fp32) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_dense_impl</*Accumulate=*/true>(
        a_base, b_base, a_sf_base, b_sf_base, alpha_a_base, alpha_b_base, d_base, a_row_offsets,
        b_row_offsets, a_sf_offsets, b_sf_offsets, Ms, Ns, K,
        /*beta=*/accumulate ? 1.0f : 0.0f, stream);
  } else if (nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC1, /*CheckM=*/true,
                                               /*CheckN=*/true>(gemm_kind, Ms, Ns, fc1_2sm_n256)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_dense_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel2SmN256::Gemm,
        nvfp4_cutlass::Kernel2SmN256::FusedEVT, /*ClusterM=*/2>(
        a_base, b_base, a_sf_base, b_sf_base, alpha_a_base, alpha_b_base, d_base, a_row_offsets,
        b_row_offsets, a_sf_offsets, b_sf_offsets, Ms, Ns, K, /*beta=*/0.0f, stream);
  } else if (nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC2, /*CheckM=*/true,
                                               /*CheckN=*/false>(gemm_kind, Ms, Ns, true)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_dense_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel2SmN128::Gemm,
        nvfp4_cutlass::Kernel2SmN128::FusedEVT, /*ClusterM=*/2>(
        a_base, b_base, a_sf_base, b_sf_base, alpha_a_base, alpha_b_base, d_base, a_row_offsets,
        b_row_offsets, a_sf_offsets, b_sf_offsets, Ms, Ns, K, /*beta=*/0.0f, stream);
  } else if (nvfp4_cutlass::should_use_alt_tile<NVTE_NVFP4_GROUPED_GEMM_FC1, /*CheckM=*/false,
                                               /*CheckN=*/true>(gemm_kind, Ms, Ns, fc1_1sm_n256)) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_dense_impl<
        /*Accumulate=*/false, nvfp4_cutlass::Kernel1SmN256::Gemm,
        nvfp4_cutlass::Kernel1SmN256::FusedEVT, /*ClusterM=*/1>(
        a_base, b_base, a_sf_base, b_sf_base, alpha_a_base, alpha_b_base, d_base, a_row_offsets,
        b_row_offsets, a_sf_offsets, b_sf_offsets, Ms, Ns, K, /*beta=*/0.0f, stream);
  } else {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_dense_impl</*Accumulate=*/false>(
        a_base, b_base, a_sf_base, b_sf_base, alpha_a_base, alpha_b_base, d_base, a_row_offsets,
        b_row_offsets, a_sf_offsets, b_sf_offsets, Ms, Ns, K, /*beta=*/0.0f, stream);
  }
#else
  NVTE_ERROR(
      "CUTLASS NVFP4 grouped per-token GEMM requires SM100 (Blackwell). Build with "
      "sm_100a/sm_100f.");
#endif
}
