# Quick Start

<!-- md-trans-meta sourceCommit=1da8a7b972c205a5fed78be3f7f99a013fefddb8 translatedAt=2026-10-16T10:34:06.997Z pushedAt=2026-10-23T07:27:10.633Z -->

This document describes how to quickly enable and verify the Eigen TensorContraction KGemm optimization.

## Environment Requirements

- An AArch64 Linux system with an Armv8-A NEON processor.
- GCC or Clang with C++11 support.
- Eigen 3.4.0, 5.0.0, or the confirmed compatible 5.0.1 version.

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

3. Verify patch integrity, then use `git apply --check` to check applicability.

   ```bash
   (cd eigen/5.0.0 && sha256sum -c SHA256SUMS) || exit 1
   git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
   ```

4. Apply the patch using `git apply`.

   ```bash
   git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
   ```

For Eigen 3.4.0, use `3.4.0/eigen-3.4.0-kgemm.patch`. Define `EIGEN_NEON_USE_KGEMM=1` when compiling service code.

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM=1 \
  -Ieigen-source your_program.cpp -o your_program
```

When KGemm does not meet the dispatch conditions, it automatically falls back to the Eigen open-source kernel, requiring no branch handling on the app side.

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

## Verifying Optimization Results

Run the NEON and KGemm builds with the same input, thread count, number of repetitions, and CPU pinning conditions. First confirm that the computation results are identical, then compare GFLOPS or elapsed time.

## FAQs

For installation, macro definitions, and fallback conditions, see [Installation Guide](installation_guide.md) and [API Reference](api_reference.md).

## Change History

| Version | Date | Description |
| --- | --- | --- |
| 01 | 2026-09-30 | This is the first official release. |
