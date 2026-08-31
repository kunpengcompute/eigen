# API Reference

<!-- md-trans-meta sourceCommit=3c5af46324366cb6f7c09e9b8ffe0014626ff606 translatedAt=2026-08-28T08:54:25.366Z pushedAt=2026-08-29T07:08:13.923Z -->

## Interface Description

The KGemm optimization integrates with Eigen's public TensorContraction expression through compile-time macros, without introducing any new public Eigen API that users must call. The main configuration and internal interfaces are shown in [**Table 1** KGemm Interface List](#kgemm-interface-list).

**Table 1** KGemm interface list<a id="kgemm-interface-list"></a>

|Interface Name|Description|
|--|--|
|`EIGEN_NEON_USE_KGEMM`|Enables the AArch64 NEON KGemm TensorContraction specialization.|
|`EIGEN_NEON_KGEMM_REUSE_PACKING`|Controls whether large TensorContraction operations reuse the scheduler packing. Default value: 1|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN`|Minimum M/N value for packing reuse to take effect. Default value: 768|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K`|Minimum K value for packing reuse. Default value: 512|
|`Tensor::contract`|User-side TensorContraction entry point, with an interface identical to native Eigen|
|`kgemm_neon_fp32_nn`|Internal driver for contiguous FP32 NN GEMM|
|`kgemm_neon_fp32_nn_packed`|Internal driver for pre-packed FP32 GEMM|

## Compilation Macro Definitions

### EIGEN_NEON_USE_KGEMM

**Macro Function**

Registers the KGemm kernel specialization for AArch64 NEON FP32 TensorContraction when the conditions are met.

**Definition Method**

```text
-DEIGEN_NEON_USE_KGEMM=1
```

All translation units that include Eigen tensor headers and participate in linking the same program should use the same macro configuration. When it is undefined or set to 0, the native Eigen implementation is used.

### EIGEN_NEON_KGEMM_REUSE_PACKING

**Macro Function**

Controls whether large KGemm operations use the TensorContraction shared block buffer to perform packing once and reuse the result.

**Definition Method**

```text
-DEIGEN_NEON_KGEMM_REUSE_PACKING=0
```

The default value is 1. Setting it to 0 builds a no-reuse performance comparison version.

### Packing Reuse Threshold

```text
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN=768
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_K=512
```

The packed path is used only when the normalized effective M/N both reach the M/N threshold and K reaches the K threshold.

## TensorContraction Interface

### Tensor::contract

**Function Description**

Contracts two tensors along the specified dimensions. The patch keeps the public Eigen interfaces unchanged and only changes the internal kernel used when the selection conditions are met.

**Call Form**

```cpp
Eigen::array<Eigen::IndexPair<int>, 1> dims = {
    Eigen::IndexPair<int>(1, 0)};
output.device(device) = lhs.contract(rhs, dims);
```

**Selection Conditions**

|Condition|Requirement|
|--|--|
|Architecture|AArch64 with NEON enabled by the compiler|
|Scalar type|Both input and output are `float`|
|Expression|Two-dimensional, one contraction dimension, canonical contiguous storage|
|Mapper|Both left and right inputs support `RawAccess`, inner dimension contiguous, RHS not pre-reordered |
|Device|`DefaultDevice` or `ThreadPoolDevice`|

When the conditions are not met, it automatically falls back to Eigen's native packing and GEBP.

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

This interface accepts contiguous RowMajor NN matrix views. `addc=0` indicates overwriting the output, while a non-zero value indicates accumulation. It is an internal Eigen interface and provides no ABI stability guarantees.

### kgemm_neon_fp32_nn_packed

```cpp
static inline void kgemm_neon_fp32_nn_packed(
    float* pa, float* pb, float* c,
    long int ldc, long int m, long int n,
    long int k, long int addc);
```

`pa` uses a 4-row K-major panel layout, and `pb` uses a 16-column K-major panel layout. This interface is invoked by the TensorContraction scheduler and is not recommended for direct invocation by service code.

## Change History

| Date | Description |
| --- | --- |
| 2026-09-30 | This is the first official release. Added the AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0. |
