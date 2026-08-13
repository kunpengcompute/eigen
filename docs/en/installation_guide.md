# Installation Guide

This guide describes how to apply the TensorContraction KGemm optimization to Eigen 3.4.0, 5.0.0, and compatible releases.

## Environment Requirements

| Category | Requirement |
| --- | --- |
| Operating system | Linux AArch64; openEuler 24.03 LTS is recommended |
| Processor | AArch64 processor with ARMv8-A NEON support |
| Compiler | GCC 12 or Clang 17; other AArch64 NEON-capable versions may also work |
| Build tools | Git, CMake, and a C++11-capable build environment |

## Compilation and Installation

### Obtaining Code

#### Obtain and Apply a Release Patch

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0          # or 3.4.0
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

For Eigen 3.4.0, use `3.4.0/eigen-3.4.0-kgemm.patch` instead.

### Installation Procedure

Eigen is header-only. Add the source root to the include path directly, or install it with CMake.

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/eigen-kgemm
cmake --build build
cmake --install build
```

## Compilation Option Description

| Option | Description | Default |
| --- | --- | --- |
| `EIGEN_NEON_USE_KGEMM` | Enables KGemm for AArch64 NEON TensorContraction | Undefined (disabled) |
| `EIGEN_USE_THREADS` | Enables the Eigen ThreadPool device | Undefined (disabled) |
| `EIGEN_NEON_KGEMM_REUSE_PACKING` | Reuses right-hand packing in multithreaded execution | `1` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN` | Minimum M/N dimensions for packing reuse | `768` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K` | Minimum K dimension for packing reuse | `512` |

Recommended compiler flags.

```bash
-O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM
```

For multithreaded tests, also add `-DEIGEN_USE_THREADS` and `-pthread`.

## Running Test

Release patches do not include test programs. Use an application TensorContraction workload or upstream Eigen tests to build native NEON and KGemm configurations and compare correctness and performance. Example KGemm flags.

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o tensor_contraction_test
./tensor_contraction_test
```

## FAQs

### What if the patch cannot be applied?

Confirm that the source matches the Eigen release named by the patch, then run `git apply --check` to locate conflicts. The Eigen 5.0.0 patch has also been verified to apply cleanly to the official 5.0.1 tag.

### Why is KGemm not selected after defining the macro?

KGemm is limited to AArch64 NEON FP32 TensorContraction with the supported non-transposed layout and minimum dimension thresholds. Other combinations fall back to Eigen's native path.

### How is multithreading enabled?

Define `EIGEN_USE_THREADS`, execute the contraction with `Eigen::ThreadPoolDevice`, and link with `-pthread`.

## Revision History

| Release Date | Revision Record |
| --- | --- |
| 2026-09-30 | First official release. Adds the AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0. |
