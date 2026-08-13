# Quick Start

This chapter shows how to enable and verify the Eigen TensorContraction KGemm optimization.

## Environment Requirements

- AArch64 Linux and an ARMv8-A NEON processor.
- GCC or Clang with C++11 support.
- Eigen 3.4.0, 5.0.0, or the verified-compatible 5.0.1 release.

## Enabling AArch64 Optimization

### Obtain and Apply a Patch to Upstream Eigen

To obtain and apply the patch, follow these steps.

1. Clone this repository to get the optimization patches.

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
```

2. Clone the upstream Eigen source and check out the target release.

```bash
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0
```

3. Verify that the patch applies with `git apply --check`.

```bash
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

4. Apply the patch with `git apply`.

```bash
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

For Eigen 3.4.0, use `3.4.0/eigen-3.4.0-kgemm.patch`. Define
`EIGEN_NEON_USE_KGEMM=1` when compiling the application.

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM=1 \
  -Ieigen-source your_program.cpp -o your_program
```

When KGemm dispatch conditions are not met, execution automatically falls back to Eigen's native kernel.

## Usage Example (Including Header File Enablement)

- Application code continues to use the standard Tensor contraction API.

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

- Use `Eigen::ThreadPoolDevice` for multithreaded execution.

```cpp
Eigen::ThreadPool pool(thread_count);
Eigen::ThreadPoolDevice device(&pool, thread_count);
out.device(device) = lhs.contract(rhs, dims);
```

## Usage Example (Performance Comparison)

Build the same application test program in native NEON and KGemm configurations.

```bash
# Native NEON
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_neon

# KGemm NEON
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM=1 -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_kgemm
```

## Verifying Optimization Effects

Run both executables with identical inputs, thread counts, repetitions, and CPU affinity. Confirm matching results before comparing GFLOPS or elapsed time.

## FAQs

For installation, macros, and fallback rules, see the [Installation Guide](installation_guide.md) and [API Reference](api_reference.md).

## Revision History

| Release Date | Revision Record |
| --- | --- |
| 2026-09-30 | First official release. Adds the AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0. |
