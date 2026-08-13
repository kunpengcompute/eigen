# Introduction to Eigen TensorContraction KGemm

## Latest Updates

- [2026.09.30]: First official release with the AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0.

## Project Introduction

[Eigen](https://gitlab.com/libeigen/eigen) is a C++ template library for linear algebra, matrices, vectors, and tensor computations.

This project optimizes FP32 TensorContraction matrix multiplication on AArch64. It integrates the NEON KGemm implementation as a `TensorContractionKernel` specialization and supports both `DefaultDevice` and `ThreadPoolDevice`. Expressions that do not satisfy the KGemm selection policy continue to use Eigen's native implementation.

## Directory Structure

The full project directory structure is as follows.

```text
eigen/
├── docs/                           # Documentation directory
│   ├── LICENSE                     # Documentation license (MPL-2.0)
│   ├── en/                         # English documentation
│   │   ├── api_reference.md        # API reference
│   │   ├── installation_guide.md   # Installation guide
│   │   ├── quick_start.md          # Quick start guide
│   │   └── release_notes.md        # Release notes
│   └── zh/                         # Chinese documentation
│       ├── api_reference.md        # API reference
│       ├── installation_guide.md   # Installation guide
│       ├── quick_start.md          # Quick start guide
│       └── release_notes.md        # Release notes
├── 3.4.0/
│   ├── eigen-3.4.0-kgemm.patch     # Eigen 3.4.0 source patch (tests excluded)
│   └── SHA256SUMS                  # Patch checksum
├── 5.0.0/
│   ├── eigen-5.0.0-kgemm.patch     # Eigen 5.0.0 source patch, also applicable to 5.0.1
│   └── SHA256SUMS                  # Patch checksum
├── LICENSE                         # Project license (MPL-2.0)
├── README.md                       # Chinese project introduction
└── README_EN.md                    # English project introduction
```

## Release Notes

For details about feature changes in each released version, see [Release Notes](docs/en/release_notes.md).

## Quick Start

For instructions on obtaining Eigen, applying the appropriate patch, and enabling KGemm, see [Quick Start](docs/en/quick_start.md).

## Documentation

| Document Name | Description |
| --------- | --------- |
| [Installation Guide](./docs/en/installation_guide.md) | Describes how to obtain, apply, compile, and verify the Eigen AArch64 KGemm patches. |
| [Quick Start](./docs/en/quick_start.md) | Provides patch selection, KGemm enablement, and a TensorContraction example. |
| [Release Notes](./docs/en/release_notes.md) | Provides patch versions, baselines, features, and known limitations. |
| [API Reference](./docs/en/api_reference.md) | Describes compile-time macros, the Tensor API, and internal KGemm interfaces. |

## Disclaimer

This repository only optimizes selected FP32 TensorContraction matrix multiplication paths for AArch64 processors. It retains the design and fallback mechanisms of the original open-source software. Vulnerability and security issues should be handled by the corresponding upstream community according to its response process. Please monitor Eigen upstream notices and updates. The ARM64 computing community assumes no responsibility for software vulnerabilities or security issues.

## License

This project uses the same primary license as Eigen (MPL-2.0). For details, see [LICENSE](LICENSE). Individual upstream files in the patch may remain under other compatible licenses listed by the corresponding Eigen release.

## Contributing Guide

If you have questions or want to report feature requirements or bugs, submit an issue.

## Suggestions and Feedback

Contributions are welcome. If you have questions or suggestions, submit an issue and we will respond as soon as possible.

## Acknowledgments

Thanks to the Eigen community and everyone who contributed to KGemm development, porting, and validation.
