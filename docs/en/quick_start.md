# Quick Start

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

3. Verify patch integrity, then use `git apply --check` to check applicability.

   ```bash
   (cd eigen/5.0.0 && sha256sum -c SHA256SUMS) || exit 1
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

When KGemm does not meet the dispatch conditions, it automatically falls back to the Eigen open-source kernel, requiring no branch handling on the app side.

### Verifying Patch Integrity

Obtain the patch and `SHA256SUMS` from the same trusted release. From the release repository root,
run the command for your version:

```bash
(cd 5.0.0 && sha256sum -c SHA256SUMS) || exit 1
# For 3.4.0:
(cd 3.4.0 && sha256sum -c SHA256SUMS) || exit 1
```

Continue only when the patch is reported as `OK` and the command exits with code 0.
On `FAILED`, a missing file, or any nonzero exit status, stop and obtain both files again from the
trusted release, then recheck. `git apply --check` checks applicability, not integrity.
A checksum list obtained alongside the patch is not a publisher's digital signature.

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

### Measured Performance Reference

CPU: Kunpeng 950; OS: openEuler 24.03 LTS-SP3.

| Threads | M×K×N | NEON ms | KGEMM ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | 128×128×128 | 0.0899 | 0.0735 | 1.223× |
| 1 | 512×512×512 | 5.5757 | 4.5462 | 1.226× |
| 1 | 768×512×768 | 12.4079 | 10.1028 | 1.228× |
| 1 | 781×513×783 | 13.3818 | 10.6886 | 1.252× |
| 4 | 128×128×128 | 0.0512 | 0.0340 | 1.505× |
| 4 | 512×512×512 | 1.8540 | 1.3577 | 1.370× |
| 4 | 768×512×768 | 3.5875 | 2.9963 | 1.197× |
| 4 | 781×513×783 | 3.7806 | 3.0478 | 1.241× |

## FAQs

For installation, macro definitions, and fallback conditions, see [Installation Guide](installation_guide.md) and [API Reference](api_reference.md).

## Change History

| Date | Description |
| --- | --- |
| 2026-09-30 | This is the first official release. |
