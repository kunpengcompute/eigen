// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_KGEMM_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_KGEMM_H

#if EIGEN_ARCH_ARM64 && defined(EIGEN_VECTORIZE_NEON) && \
    defined(EIGEN_NEON_USE_KGEMM) && EIGEN_NEON_USE_KGEMM

#include "../../../../../Eigen/src/Core/arch/NEON/KGemmNeonKernel.h"

#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION
#include <atomic>
#endif

#ifndef EIGEN_NEON_KGEMM_REUSE_PACKING
#define EIGEN_NEON_KGEMM_REUSE_PACKING 1
#endif

#ifndef EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN
#define EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN 768
#endif

#ifndef EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K
#define EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K 512
#endif

namespace Eigen {
namespace internal {

#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION
struct KGemmTestInvocationCounters {
  KGemmTestInvocationCounters() : raw(0), packed(0) {}

  std::atomic<unsigned long long> raw;
  std::atomic<unsigned long long> packed;
};

EIGEN_STRONG_INLINE KGemmTestInvocationCounters&
kgemmTestInvocationCounters() {
  static KGemmTestInvocationCounters counters;
  return counters;
}

EIGEN_STRONG_INLINE void kgemmTestResetInvocationCounters() {
  kgemmTestInvocationCounters().raw.store(0, std::memory_order_relaxed);
  kgemmTestInvocationCounters().packed.store(0, std::memory_order_relaxed);
}

EIGEN_STRONG_INLINE unsigned long long kgemmTestRawInvocationCount() {
  return kgemmTestInvocationCounters().raw.load(std::memory_order_relaxed);
}

EIGEN_STRONG_INLINE unsigned long long kgemmTestPackedInvocationCount() {
  return kgemmTestInvocationCounters().packed.load(std::memory_order_relaxed);
}
#endif

// TensorContraction.h includes this file before defining the primary template.
// The declaration lets us register the KGemm partial specialization without
// making the implementation part of TensorContraction.h itself.
template <typename ResScalar, typename LhsScalar, typename RhsScalar,
          typename StorageIndex, typename OutputMapper, typename LhsMapper,
          typename RhsMapper, typename EnableKgemm>
struct TensorContractionKernel;

EIGEN_DONT_INLINE inline void kgemm_call(
    const float* lhs, const float* rhs, float* output, long int m,
    long int n, long int k, long int lhs_stride, long int rhs_stride,
    long int output_stride, float alpha, float beta) {
  // TensorContraction normalizes the operands and output to column-major.
  // KGemm consumes row-major NN operands, so compute the transposed view:
  //   C^T(n x m) = B^T(n x k) * A^T(k x m).
  eigen_assert(alpha == 1.0f);
  eigen_assert(beta == 0.0f || beta == 1.0f);
  kgemm_neon_fp32_nn(
      const_cast<float*>(rhs), const_cast<float*>(lhs), output,
      rhs_stride, lhs_stride, output_stride, n, m, k,
      beta != 0.0f ? 1 : 0);
}

template <typename Mapper>
struct KGemmRawContiguousLhsMapper : std::false_type {};

template <typename Mapper>
struct KGemmRawContiguousRhsMapper : std::false_type {};

template <typename Index, typename Tensor, typename nocontract_t,
          typename contract_t, int packet_size, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment,
          template <class> class MakePointer_>
struct KGemmRawContiguousLhsMapper<
    TensorContractionInputMapper<float, Index, Lhs, Tensor, nocontract_t,
                                 contract_t, packet_size,
                                 inner_dim_contiguous, inner_dim_reordered,
                                 Alignment, MakePointer_> >
    : std::integral_constant<
          bool, Tensor::RawAccess && inner_dim_contiguous &&
                    !inner_dim_reordered &&
                    array_size<nocontract_t>::value == 1 &&
                    array_size<contract_t>::value == 1> {
  typedef TensorContractionInputMapper<
      float, Index, Lhs, Tensor, nocontract_t, contract_t, packet_size,
      inner_dim_contiguous, inner_dim_reordered, Alignment, MakePointer_>
      Mapper;

  EIGEN_STRONG_INLINE static const float* data(
      const typename Mapper::SubMapper& mapper) {
    // For an eligible Lhs mapper, SubMapper folds its logical block origin
    // into the copied coefficient loader.
    return mapper.base_mapper().tensor().data();
  }
};

template <typename Index, typename Tensor, typename nocontract_t,
          typename contract_t, int packet_size, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment,
          template <class> class MakePointer_>
struct KGemmRawContiguousRhsMapper<
    TensorContractionInputMapper<float, Index, Rhs, Tensor, nocontract_t,
                                 contract_t, packet_size,
                                 inner_dim_contiguous, inner_dim_reordered,
                                 Alignment, MakePointer_> >
    : std::integral_constant<
          bool, Tensor::RawAccess && inner_dim_contiguous &&
                    !inner_dim_reordered &&
                    array_size<nocontract_t>::value == 1 &&
                    array_size<contract_t>::value == 1> {
  typedef TensorContractionInputMapper<
      float, Index, Rhs, Tensor, nocontract_t, contract_t, packet_size,
      inner_dim_contiguous, inner_dim_reordered, Alignment, MakePointer_>
      Mapper;

  EIGEN_STRONG_INLINE static const float* data(
      const typename Mapper::SubMapper& mapper) {
    // Rhs SubMapper does not fold its block origin into CoeffLoader. Convert
    // the logical (k,n) origin through the mapper to preserve its strides.
    const typename Mapper::Base& base = mapper.base_mapper();
    return base.tensor().data() +
           base.computeIndex(mapper.vert_offset(), mapper.horiz_offset());
  }
};

template <typename LhsMapper, typename RhsMapper>
struct CanUseKgemmContraction
    : std::integral_constant<
          bool, KGemmRawContiguousLhsMapper<LhsMapper>::value &&
                    KGemmRawContiguousRhsMapper<RhsMapper>::value> {};

template <typename StorageIndex>
struct KGemmTensorBlockMemAllocator {
  typedef void* BlockMemHandle;

  template <typename Device>
  EIGEN_DEVICE_FUNC static BlockMemHandle allocate(
      Device& device, const StorageIndex bm, const StorageIndex bk,
      const StorageIndex bn, float** lhs_block, float** rhs_block) {
    const BlockSizes sizes = computeBlockSizes(bm, bk, bn);
    char* memory = static_cast<char*>(
        device.allocate(sizes.lhs_bytes + sizes.rhs_bytes));
    eigen_assert(memory != NULL);
    *lhs_block = reinterpret_cast<float*>(memory);
    *rhs_block = reinterpret_cast<float*>(memory + sizes.lhs_bytes);
    return memory;
  }

  template <typename Device>
  EIGEN_DEVICE_FUNC static BlockMemHandle allocateSlices(
      Device& device, const StorageIndex bm, const StorageIndex bk,
      const StorageIndex bn, const StorageIndex num_lhs,
      const StorageIndex num_rhs, const StorageIndex num_slices,
      std::vector<float*>* lhs_blocks, std::vector<float*>* rhs_blocks) {
    eigen_assert(num_slices > 0);
    eigen_assert(num_lhs >= 0 && num_rhs >= 0);
    eigen_assert(num_lhs == 0 || lhs_blocks != NULL);
    eigen_assert(num_rhs == 0 || rhs_blocks != NULL);
    const BlockSizes sizes = computeBlockSizes(bm, bk, bn);
    char* memory = static_cast<char*>(device.allocate(
        (num_lhs * sizes.lhs_bytes + num_rhs * sizes.rhs_bytes) *
        num_slices));
    eigen_assert(memory != NULL);
    char* next = memory;
    for (StorageIndex slice = 0; slice < num_slices; ++slice) {
      if (num_lhs > 0) lhs_blocks[slice].resize(num_lhs);
      for (StorageIndex block = 0; block < num_lhs; ++block) {
        lhs_blocks[slice][block] = reinterpret_cast<float*>(next);
        next += sizes.lhs_bytes;
      }
      if (num_rhs > 0) rhs_blocks[slice].resize(num_rhs);
      for (StorageIndex block = 0; block < num_rhs; ++block) {
        rhs_blocks[slice][block] = reinterpret_cast<float*>(next);
        next += sizes.rhs_bytes;
      }
    }
    return memory;
  }

  template <typename Device>
  EIGEN_DEVICE_FUNC static void deallocate(Device& device,
                                           BlockMemHandle handle) {
    device.deallocate(handle);
  }

 private:
  struct BlockSizes {
    StorageIndex lhs_bytes;
    StorageIndex rhs_bytes;
  };

  EIGEN_DEVICE_FUNC static BlockSizes computeBlockSizes(
      const StorageIndex bm, const StorageIndex bk, const StorageIndex bn) {
    const StorageIndex alignment = numext::maxi(EIGEN_MAX_ALIGN_BYTES, 1);
    const StorageIndex lhs_rows = numext::div_ceil<StorageIndex>(bm, 16) * 16;
    const StorageIndex rhs_cols = numext::div_ceil<StorageIndex>(bn, 4) * 4;
    BlockSizes sizes;
    sizes.lhs_bytes = numext::div_ceil<StorageIndex>(
                          lhs_rows * bk * sizeof(float), alignment) *
                      alignment;
    sizes.rhs_bytes = numext::div_ceil<StorageIndex>(
                          rhs_cols * bk * sizeof(float), alignment) *
                      alignment;
    return sizes;
  }
};

template <typename StorageIndex, typename OutputMapper, typename LhsMapper,
          typename RhsMapper>
struct TensorContractionKernel<
    float, float, float, StorageIndex, OutputMapper, LhsMapper, RhsMapper,
    std::enable_if_t<CanUseKgemmContraction<LhsMapper, RhsMapper>::value> > {
  enum { HasBeta = true, IsKGemm = true };

#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
  typedef float* LhsBlock;
  typedef float* RhsBlock;
  typedef KGemmTensorBlockMemAllocator<StorageIndex> BlockMemAllocator;
  typedef typename BlockMemAllocator::BlockMemHandle BlockMemHandle;
#else
  typedef const float* LhsBlock;
  typedef const float* RhsBlock;
  typedef void* BlockMemHandle;
#endif

  EIGEN_DEVICE_FUNC TensorContractionKernel(
      StorageIndex m, StorageIndex k, StorageIndex n, StorageIndex bm,
      StorageIndex bk, StorageIndex bn)
      : lhs_stride_(m),
        rhs_stride_(k),
        bm_(bm),
        bk_(bk),
        bn_(bn),
        reuse_packing_(shouldReusePacking(m, k, n)) {}

  template <typename Device>
  EIGEN_DEVICE_FUNC BlockMemHandle allocate(Device& device,
                                             LhsBlock* lhs_block,
                                             RhsBlock* rhs_block) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    if (reuse_packing_) {
      return BlockMemAllocator::allocate(device, bm_, bk_, bn_, lhs_block,
                                         rhs_block);
    }
    *lhs_block = NULL;
    *rhs_block = NULL;
    return NULL;
#else
    *lhs_block = NULL;
    *rhs_block = NULL;
    return NULL;
#endif
  }

  template <typename Device>
  EIGEN_DEVICE_FUNC BlockMemHandle allocateSlices(
      Device& device, const StorageIndex num_lhs, const StorageIndex num_rhs,
      const StorageIndex num_slices, std::vector<LhsBlock>* lhs_blocks,
      std::vector<RhsBlock>* rhs_blocks) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    if (reuse_packing_) {
      return BlockMemAllocator::allocateSlices(
          device, bm_, bk_, bn_, num_lhs, num_rhs, num_slices, lhs_blocks,
          rhs_blocks);
    }
    eigen_assert(num_slices > 0);
    for (StorageIndex slice = 0; slice < num_slices; ++slice) {
      if (lhs_blocks != NULL) lhs_blocks[slice].resize(num_lhs, NULL);
      if (rhs_blocks != NULL) rhs_blocks[slice].resize(num_rhs, NULL);
    }
    return NULL;
#else
    eigen_assert(num_slices > 0);
    eigen_assert(num_lhs >= 0 && num_rhs >= 0);
    for (StorageIndex slice = 0; slice < num_slices; ++slice) {
      if (lhs_blocks != NULL) lhs_blocks[slice].resize(num_lhs, NULL);
      if (rhs_blocks != NULL) rhs_blocks[slice].resize(num_rhs, NULL);
    }
    return NULL;
#endif
  }

  template <typename Device>
  EIGEN_DEVICE_FUNC static void deallocate(Device& device,
                                           BlockMemHandle handle) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    if (handle != NULL) BlockMemAllocator::deallocate(device, handle);
#else
    EIGEN_UNUSED_VARIABLE(device);
    EIGEN_UNUSED_VARIABLE(handle);
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packLhs(
      LhsBlock* lhs_block, const typename LhsMapper::SubMapper& data_mapper,
      const StorageIndex depth, const StorageIndex rows) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    const float* data =
        KGemmRawContiguousLhsMapper<LhsMapper>::data(data_mapper);
    if (reuse_packing_) {
      kgemm_pack_b_kmajor16(*lhs_block, data,
                            static_cast<long int>(lhs_stride_),
                            static_cast<long int>(rows),
                            static_cast<long int>(depth));
    } else {
      *lhs_block = const_cast<float*>(data);
    }
#else
    EIGEN_UNUSED_VARIABLE(depth);
    EIGEN_UNUSED_VARIABLE(rows);
    *lhs_block = KGemmRawContiguousLhsMapper<LhsMapper>::data(data_mapper);
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void packRhs(
      RhsBlock* rhs_block, const typename RhsMapper::SubMapper& data_mapper,
      const StorageIndex depth, const StorageIndex cols) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    const float* data =
        KGemmRawContiguousRhsMapper<RhsMapper>::data(data_mapper);
    if (reuse_packing_) {
      kgemm_pack_a_kmajor4(*rhs_block, data,
                           static_cast<long int>(rhs_stride_),
                           static_cast<long int>(cols),
                           static_cast<long int>(depth));
    } else {
      *rhs_block = const_cast<float*>(data);
    }
#else
    EIGEN_UNUSED_VARIABLE(depth);
    EIGEN_UNUSED_VARIABLE(cols);
    *rhs_block = KGemmRawContiguousRhsMapper<RhsMapper>::data(data_mapper);
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void invoke(
      const OutputMapper& output_mapper, const LhsBlock& lhs_block,
      const RhsBlock& rhs_block, const StorageIndex rows,
      const StorageIndex depth, const StorageIndex cols, const float alpha,
      const float beta) {
    eigen_assert(lhs_block != NULL && rhs_block != NULL &&
                 output_mapper.data() != NULL);
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    if (reuse_packing_) {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION
      kgemmTestInvocationCounters().packed.fetch_add(1,
                                                     std::memory_order_relaxed);
#endif
      eigen_assert(alpha == 1.0f);
      eigen_assert(beta == 0.0f || beta == 1.0f);
      kgemm_neon_fp32_nn_packed(
          rhs_block, lhs_block, const_cast<float*>(output_mapper.data()),
          static_cast<long int>(output_mapper.stride()),
          static_cast<long int>(cols), static_cast<long int>(rows),
          static_cast<long int>(depth), beta != 0.0f ? 1 : 0);
    } else {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION
      kgemmTestInvocationCounters().raw.fetch_add(1,
                                                  std::memory_order_relaxed);
#endif
      kgemm_call(
          lhs_block, rhs_block, const_cast<float*>(output_mapper.data()),
          static_cast<long int>(rows), static_cast<long int>(cols),
          static_cast<long int>(depth), static_cast<long int>(lhs_stride_),
          static_cast<long int>(rhs_stride_),
          static_cast<long int>(output_mapper.stride()), alpha, beta);
    }
#else
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION
    kgemmTestInvocationCounters().raw.fetch_add(1,
                                                std::memory_order_relaxed);
#endif
    kgemm_call(
        lhs_block, rhs_block, const_cast<float*>(output_mapper.data()),
        static_cast<long int>(rows), static_cast<long int>(cols),
        static_cast<long int>(depth), static_cast<long int>(lhs_stride_),
        static_cast<long int>(rhs_stride_),
        static_cast<long int>(output_mapper.stride()), alpha, beta);
#endif
  }

 private:
  EIGEN_DEVICE_FUNC static bool shouldReusePacking(
      const StorageIndex m, const StorageIndex k, const StorageIndex n) {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
    return numext::mini(m, n) >=
               static_cast<StorageIndex>(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN) &&
           k >= static_cast<StorageIndex>(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K);
#else
    EIGEN_UNUSED_VARIABLE(m);
    EIGEN_UNUSED_VARIABLE(k);
    EIGEN_UNUSED_VARIABLE(n);
    return false;
#endif
  }

  const StorageIndex lhs_stride_;
  const StorageIndex rhs_stride_;
  const StorageIndex bm_;
  const StorageIndex bk_;
  const StorageIndex bn_;
  const bool reuse_packing_;
};

}  // namespace internal
}  // namespace Eigen

#endif  // ARM64 NEON KGEMM

#endif  // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_KGEMM_H
