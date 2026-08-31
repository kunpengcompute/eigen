# Release Notes

<!-- md-trans-meta sourceCommit=3c5af46324366cb6f7c09e9b8ffe0014626ff606 translatedAt=2026-08-28T08:54:23.758Z pushedAt=2026-08-29T03:47:56.596Z -->

## Version Compatibility Notes

| Patch Directory | Baseline Version | Compatibility Notes |
| --- | --- | --- |
| `3.4.0` | Eigen 3.4.0 | Generated and verified against the 3.4.0 source code |
| `5.0.0` | Eigen 5.0.0 | Generated against 5.0.0; verified to apply to the official 5.0.1 tag |

### Product Version Information

| Item | Content |
| --- | --- |
| Name | Eigen TensorContraction KGemm Optimization Patch |
| Current Version | V1.0.0 |
| Release Date | 2026-08-12 |
| Open Source License | Mozilla Public License 2.0 |

### OS, Compiler, and CPU

| Category | Verified Environment |
| --- | --- |
| OS | openEuler 24.03 LTS (AArch64) |
| Compiler | GCC 12, Clang 17 |
| CPU instruction set | Armv8-A NEON |
| Thread configuration | 1, 2, 4, 8 threads |

## Version Update Notes

### V1.0.0

1. Integrated `kgemm_neon_fp32_nn` into the TensorContraction dispatch layer while retaining the Eigen native kernel fallback path.

2. Added `TensorContractionKGemm.h` to isolate KGemm adaptation, threshold, and multithreading logic.

3. Supports ThreadPool parallel execution and right-hand side matrix packing reuse to reduce redundant packing.

4. Provided single-file patches and SHA-256 checksums for Eigen 3.4.0 and 5.0.0.

## Known Limitations

1. KGemm is enabled only on AArch64 NEON FP32 and the supported NN data layout.

2. Contractions of small sizes or those that do not meet the threshold automatically fall back to the native Eigen implementation.

3. The patch directory only lists Eigen versions that have been verified. For other versions, run `git apply --check` first and complete correctness testing before applying.

4. Actual gains depend on matrix shape, thread count, CPU frequency, CPU pinning, and memory bandwidth.

## Documentation

### V1.0.0 Documents

| Document | Description |
| --- | --- |
| [Installation Guide](installation_guide.md) | Environment, patch application, compilation, and testing methods |
| [Quick Start](quick_start.md) | Shortest enablement process and usage examples |
| [API Reference](api_reference.md) | Macros, interfaces, and dispatch constraints |

## Obtaining the Documentation

You can browse and obtain the relevant documentation by visiting the [open-source repository](https://gitcode.com/boostkit/eigen).
