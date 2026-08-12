# API Reference

## Interfaces

The optimization integrates with Eigen's public TensorContraction expression through compile-time macros. It does not add a mandatory public Eigen API. The main configuration and internal interfaces are listed in [**Table 1** KGemm Interfaces](#KGemmInterfaces).

**Table 1** KGemm Interfaces<a id="KGemmInterfaces"></a>

|Name|Description|
|--|--|
|`EIGEN_NEON_USE_KGEMM`|Enables the AArch64 NEON KGemm TensorContraction specialization.|
|`EIGEN_NEON_KGEMM_REUSE_PACKING`|Controls scheduler packing reuse for large contractions. The default is 1.|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN`|Effective M/N threshold for packing reuse. The default is 768.|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K`|K threshold for packing reuse. The default is 512.|
|`Tensor::contract`|User-facing TensorContraction entry point, unchanged from upstream Eigen.|
|`kgemm_neon_fp32_nn`|Internal contiguous FP32 NN GEMM driver.|
|`kgemm_neon_fp32_nn_packed`|Internal packed FP32 GEMM driver.|

## Compile-Time Macros

### EIGEN_NEON_USE_KGEMM

**Function**

Registers the KGemm kernel specialization for eligible AArch64 NEON FP32 TensorContraction expressions.

**Definition**

```text
-DEIGEN_NEON_USE_KGEMM=1
```

Use the same macro value for every translation unit that includes Eigen Tensor headers and is linked into the same program. If it is undefined or set to 0, Eigen uses its native implementation.

### EIGEN_NEON_KGEMM_REUSE_PACKING

**Function**

Controls whether large KGemm contractions use TensorContraction shared block buffers to pack once and reuse the result.

**Definition**

```text
-DEIGEN_NEON_KGEMM_REUSE_PACKING=0
```

The default is 1. Set it to 0 to build a no-reuse performance baseline.

### Packing-Reuse Thresholds

```text
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN=768
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_K=512
```

The packed path is selected only when both normalized effective M/N dimensions meet the M/N threshold and K meets the K threshold.

## TensorContraction API

### Tensor::contract

**Function**

Contracts two tensors along the specified dimensions. The patch preserves Eigen's public interface and only changes the internal kernel for eligible expressions.

**Invocation**

```cpp
Eigen::array<Eigen::IndexPair<int>, 1> dims = {
    Eigen::IndexPair<int>(1, 0)};
output.device(device) = lhs.contract(rhs, dims);
```

**Selection Policy**

|Condition|Requirement|
|--|--|
|Architecture|AArch64 with compiler NEON support|
|Scalar type|Inputs and output are `float`|
|Expression|Two-dimensional, one contraction dimension, canonical contiguous storage|
|Mapper|RawAccess for both inputs, contiguous inner dimension, RHS not reordered|
|Device|`DefaultDevice` or `ThreadPoolDevice`|

If any condition is not met, Eigen automatically falls back to its native packing and GEBP path.

**Example**

```cpp
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<float, 2, Eigen::RowMajor> a(64, 128);
Eigen::Tensor<float, 2, Eigen::RowMajor> b(128, 96);
Eigen::Tensor<float, 2, Eigen::RowMajor> c(64, 96);
Eigen::array<Eigen::IndexPair<int>, 1> dims = {
    Eigen::IndexPair<int>(1, 0)};
c = a.contract(b, dims);
```

## Internal KGemm Interfaces

### kgemm_neon_fp32_nn

```cpp
static inline void kgemm_neon_fp32_nn(
    float* a, float* b, float* c,
    long int lda, long int ldb, long int ldc,
    long int m, long int n, long int k, long int addc);
```

This interface consumes contiguous row-major NN matrix views. `addc=0` overwrites the output; a nonzero value accumulates into it. This is an Eigen internal interface without ABI stability guarantees.

### kgemm_neon_fp32_nn_packed

```cpp
static inline void kgemm_neon_fp32_nn_packed(
    float* pa, float* pb, float* c,
    long int ldc, long int m, long int n,
    long int k, long int addc);
```

`pa` uses four-row K-major panels and `pb` uses 16-column K-major panels. The TensorContraction scheduler invokes this interface; application code should not call it directly.
