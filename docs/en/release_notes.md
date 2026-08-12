# Release Notes

## Version Mapping

| Patch directory | Baseline | Compatibility |
| --- | --- | --- |
| `3.4.0` | Eigen 3.4.0 | Generated and verified against the 3.4.0 source |
| `5.0.0` | Eigen 5.0.0 | Generated for 5.0.0 and verified to apply cleanly to the official 5.0.1 tag |

### Product Version Information

| Item | Value |
| --- | --- |
| Name | Eigen TensorContraction KGemm Optimization Patch |
| Current version | V1.0.0 |
| Release date | 2026.08.12 |
| License | Mozilla Public License 2.0 |

### OS, Compiler, and CPU

| Category | Verified environment |
| --- | --- |
| Operating system | openEuler 24.03 LTS (AArch64) |
| Compiler | GCC 12 and Clang 17 |
| CPU instruction set | ARMv8-A NEON |
| Thread counts | 1, 2, 4, and 8 |

## Change Description

### V1.0.0

1. Integrates `kgemm_neon_fp32_nn` at the TensorContraction dispatch layer while retaining Eigen's native kernel fallback.
2. Adds `TensorContractionKGemm.h` to isolate KGemm adaptation, thresholds, and multithreaded logic.
3. Supports ThreadPool execution and right-hand packing reuse to reduce repeated packing.
4. Provides standalone patches and SHA-256 checksums for Eigen 3.4.0 and 5.0.0.

## Known Limitations

1. KGemm is enabled only for AArch64 NEON FP32 with the supported NN data layout.
2. Small contractions or shapes below the dispatch thresholds use Eigen's native implementation.
3. Only the listed Eigen releases are verified. Run `git apply --check` and correctness tests before using another release.
4. Performance depends on shape, thread count, CPU frequency, affinity, and memory bandwidth.

## Documentation

### V1.0.0 Documentation

| Document | Description |
| --- | --- |
| [Installation Guide](installation_guide.md) | Environment, patch application, compilation, and testing |
| [Quick Start](quick_start.md) | Short enablement path and usage examples |
| [API Reference](api_reference.md) | Macros, interfaces, and dispatch constraints |

### Obtaining Documentation

Chinese documentation is under `docs/zh`, and English documentation is under `docs/en`. Use the root `README.md` and `README_EN.md` as documentation entry points.
