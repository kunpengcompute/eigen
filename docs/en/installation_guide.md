# Installation Guide

<!-- md-trans-meta sourceCommit=3c5af46324366cb6f7c09e9b8ffe0014626ff606 translatedAt=2026-08-28T08:54:24.086Z pushedAt=2026-08-29T06:39:40.031Z -->

This document describes how to apply the TensorContraction KGemm optimization patches to Eigen 3.4.0, 5.0.0, and compatible versions.

## Environment Requirements

| Category | Requirement |
| --- | --- |
| OS | Linux AArch64, openEuler 24.03 LTS recommended |
| Processor | AArch64 processor supporting Armv8-A NEON |
| Compiler | GCC 12 or Clang 17; other versions supporting AArch64 NEON are also acceptable |
| Build tool | Git, CMake, and a build environment supporting C++11 |

## Compilation and Installation

### Obtaining Code

#### Obtaining and Applying the Patch to Eigen Source Code

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0          # or 3.4.0
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

For Eigen 3.4.0, replace the above path with `3.4.0/eigen-3.4.0-kgemm.patch`.

### Installation Steps

Eigen is a header-only library. You can either add the source root directory directly to the include path, or install it using CMake.

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/eigen-kgemm
cmake --build build
cmake --install build
```

## Compilation Options

| Option | Description | Default Value |
| --- | --- | --- |
| `EIGEN_NEON_USE_KGEMM` | Enables the KGemm path in AArch64 NEON TensorContraction | Undefined (Off) |
| `EIGEN_USE_THREADS` | Enables the Eigen ThreadPool device | Undefined (Off) |
| `EIGEN_NEON_KGEMM_REUSE_PACKING` | Enables right-hand side matrix packing reuse in multithreading | `1` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN` | Minimum M/N value for packing reuse | `768` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K` | Minimum K value for packing reuse | `512` |

Recommended compilation parameters:

```bash
-O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM
```

For multi-threaded testing, also add `-DEIGEN_USE_THREADS` and `-pthread`.

## Running Tests

The release patches do not include test programs. You can use service-side TensorContraction test cases or Eigen upstream tests, building native NEON and KGemm configurations separately to compare correctness and performance. An example of KGemm build parameters is shown below.

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o tensor_contraction_test
./tensor_contraction_test
```

## FAQs

### What should I do if the patch cannot be applied?

First confirm that the current source code matches the Eigen version indicated by the patch, and use `git apply --check` to locate conflicts. The Eigen 5.0.0 patch has been verified to apply directly to the official 5.0.1 tag.

### Why does KGemm not take effect after the macro is defined?

KGemm applies only to TensorContraction with AArch64 NEON, FP32, and non-transposed layout, and is subject to a minimum dimension threshold. Other combinations automatically fall back to the native Eigen path.

### How do I enable multithreading?

Define `EIGEN_USE_THREADS`, use `Eigen::ThreadPoolDevice` to execute the contraction, and add `-pthread` when linking.

## Change History

| Date | Description |
| --- | --- |
| 2026-09-30 | This is the first official release. Added the AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0. |
