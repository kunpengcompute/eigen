# Eigen TensorContraction KGemm

<!-- md-trans-meta sourceCommit=3c5af46324366cb6f7c09e9b8ffe0014626ff606 translatedAt=2026-08-28T08:54:23.296Z pushedAt=2026-08-29T03:10:37.356Z -->

## Latest News

- [2026-09-30]: First official release. Added AArch64 KGemm TensorContraction optimization patches for Eigen 3.4.0 and 5.0.0.

## Project Introduction

[Eigen](https://gitlab.com/libeigen/eigen) is a C++ template library for linear algebra, matrix, and vector operations, providing rich functionality including the Tensor module.

This project optimizes the FP32 matrix multiplication performance of Eigen TensorContraction on the AArch64 architecture by integrating NEON KGemm into Eigen as a `TensorContractionKernel` specialization, supporting both `DefaultDevice` and `ThreadPoolDevice`. Expressions that do not meet the KGemm selection criteria will fall back to the native Eigen implementation.

## Directory Structure

The complete directory structure of the project is as follows.

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
│   ├── eigen-3.4.0-kgemm.patch     # Eigen 3.4.0 source optimization patch (excluding test files)
│   └── SHA256SUMS                  # Patch checksums
├── 5.0.0/
│   ├── eigen-5.0.0-kgemm.patch     # Eigen 5.0.0 source optimization patch, applicable to 5.0.1
│   └── SHA256SUMS                  # Patch checksums
├── LICENSE                         # Project license (MPL-2.0)
├── README.md                       # Project introduction (Chinese)
└── README_EN.md                    # Project introduction (English)
```

## Release Notes

For detailed information on feature changes in each release, see [Release Notes](docs/en/release_notes.md).

## Quick Start

For detailed steps on obtaining the Eigen source code, applying the corresponding patch, and enabling KGemm, see [Quick Start](docs/en/quick_start.md).

## Documentation

| Resource Name | Description |
| --------- | --------- |
| [Installation Guide](./docs/en/installation_guide.md) | Provides the steps for obtaining, applying, compiling, and verifying the Eigen AArch64 KGemm optimization patch. |
| [Quick Start](./docs/en/quick_start.md) | Provides patch selection, KGemm enabling, and TensorContraction examples. |
| [Release Notes](./docs/en/release_notes.md) | Provides the patch version, baseline, features, and known limitations. |
| [API Reference](./docs/en/api_reference.md) | Provides descriptions of compilation macros, Tensor APIs, and internal KGemm interfaces. |

## Disclaimer

This repository only optimizes the performance of a specific FP32 matrix multiplication path of Eigen TensorContraction on AArch64 processors, inheriting the design and fallback mechanisms of the original open-source software. Any vulnerabilities and security issues should be resolved by the corresponding upstream community according to its response mechanisms. Please closely follow the notices and version updates released by the Eigen upstream community. The AArch64 computing community assumes no responsibility for software vulnerabilities and security issues.

## License

This project is licensed under the same license (MPL-2.0) as the main Eigen code. For details, see [LICENSE](LICENSE). Individual upstream files in the patches may still be subject to the compatible licenses listed in the Eigen release package.

## Contributing Guidelines

If you encounter any issues during use, or need to report feature requests and bug reports, you can submit issues to contact us.

## Suggestions and Communication

All contributions to the community are welcome. If you have any questions or suggestions, please submit an issue, and we will respond as soon as possible. Thank you for your support.

## Acknowledgments

We thank the Eigen community and all contributors who participated in the development, porting, and verification of KGemm. We welcome and appreciate your contributions!
