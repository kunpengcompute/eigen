# Quick Start

<!-- md-trans-meta sourceCommit=3c5af46324366cb6f7c09e9b8ffe0014626ff606 translatedAt=2026-08-28T08:54:20.951Z pushedAt=2026-08-29T06:18:58.541Z -->

This document describes how to quickly enable and verify the Eigen TensorContraction KGemm optimization.

## Environment Requirements

- An AArch64 Linux system with an Armv8-A NEON processor.

- GCC or Clang with C++11 support.

- Eigen 3.4.0, 5.0.0, or the confirmed compatible 5.0.1.

## Enabling AArch64 Optimization

### Obtaining and Applying the Patch to Eigen Source Code

The steps to obtain and apply the patch are as follows.

1. Clone this repository to obtain the optimization patch.

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
```

2. Clone the Eigen upstream source code and check out the target version.

```bash
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0
```

3. Use `git apply --check` to verify that the patch can be applied.

```bash
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

4. Apply the patch using `git apply`.

```bash
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

For Eigen 3.4.0, use `3.4.0/eigen-3.4.0-kgemm.patch`. Define
`EIGEN_NEON_USE_KGEMM=1` when compiling service code.

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM=1 \
  -Ieigen-source your_program.cpp -o your_program
```

When KGemm does not meet the dispatch conditions, it automatically falls back to the Eigen native kernel, requiring no branch handling on the app side.

## Usage Examples (Including Header File Enablement)

- The app side continues to use the standard Tensor contraction interface.

```cpp
#define EIGEN_USE_THREADS
#define EIGEN_NEON_USE_KGEMM 1
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<float, 2, Eigen::RowMajor> lhs(m, k);
Eigen::Tensor<float, 2, Eigen::RowMajor> rhs(k, n);
Eigen::Tensor<float, 2, Eigen::RowMajor> out(m, n);
Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
out = lhs.contract(rhs, dims);
```

- Use `Eigen::ThreadPoolDevice` in multi-threaded scenarios.

```cpp
Eigen::ThreadPool pool(thread_count);
Eigen::ThreadPoolDevice device(&pool, thread_count);
out.device(device) = lhs.contract(rhs, dims);
```

## Usage Example (Performance Comparison)

Use the same test program to compile the native NEON and KGemm versions separately.

```bash
# Native NEON.
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_neon

# KGemm NEON
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM=1 -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_kgemm
```

## Verifying Optimization Results

Run both with the same input, thread count, number of repetitions, and CPU pinning conditions. First confirm that the computation results are identical, then compare GFLOPS or elapsed time.

## FAQs

For installation, macro definitions, and fallback conditions, see [Installation Guide](installation_guide.md) and [API Reference](api_reference.md).

## Change History

| Date | Description |
| --- | --- |
| 2026-09-30 | This is the first official release. Added AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0. |
