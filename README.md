# Eigen TensorContraction KGemm介绍

## 最新消息

- [2026.09.30]：第一次正式发布，新增基于Eigen 3.4.0和5.0.0的ARM64 KGemm TensorContraction优化补丁。

## 项目介绍

[Eigen](https://gitlab.com/libeigen/eigen)是一款用于线性代数、矩阵和向量运算的C++模板库，提供包括Tensor模块在内的丰富功能。

本项目针对Eigen TensorContraction在ARM64架构上的FP32矩阵乘性能进行优化，将NEON KGemm以`TensorContractionKernel`特化方式接入Eigen，并支持`DefaultDevice`和`ThreadPoolDevice`。不满足KGemm选择条件的表达式仍使用Eigen原生实现。

## 目录结构

项目全量目录层级介绍如下。

```text
eigen/
├── docs/                           # 文档目录
│   ├── LICENSE                     # 文档许可证（MPL-2.0）
│   ├── en/                         # 英文文档
│   │   ├── api_reference.md        # API参考
│   │   ├── installation_guide.md   # 安装指南
│   │   ├── quick_start.md          # 快速入门文档
│   │   └── release_notes.md        # 版本说明书
│   └── zh/                         # 中文文档
│       ├── api_reference.md        # API参考
│       ├── installation_guide.md   # 安装指南
│       ├── quick_start.md          # 快速入门文档
│       └── release_notes.md        # 版本说明书
├── 3.4.0/
│   ├── eigen-3.4.0-kgemm.patch     # Eigen 3.4.0源码优化补丁（不含测试文件）
│   └── SHA256SUMS                  # 补丁校验值
├── 5.0.0/
│   ├── eigen-5.0.0-kgemm.patch     # Eigen 5.0.0源码优化补丁，可应用于5.0.1
│   └── SHA256SUMS                  # 补丁校验值
├── LICENSE                         # 项目许可证（MPL-2.0）
├── README.md                       # 中文项目介绍
└── README_EN.md                    # 英文项目介绍
```

## 版本说明

每个发布版本特性变更详细信息，请参见《[版本说明书](docs/zh/release_notes.md)》。

## 快速入门

获取Eigen源码、应用对应版本补丁及使能KGemm的详细步骤，请参见《[快速入门](docs/zh/quick_start.md)》。

## 文档

| 资源名称 | 资源简介 |
| --------- | --------- |
| [安装指南](./docs/zh/installation_guide.md) | 提供Eigen ARM64 KGemm优化补丁的获取、应用、编译和验证步骤。 |
| [快速入门](./docs/zh/quick_start.md) | 提供补丁选择、KGemm使能和TensorContraction示例。 |
| [版本说明书](./docs/zh/release_notes.md) | 提供补丁版本、基线、特性和已知限制。 |
| [API参考](./docs/zh/api_reference.md) | 提供编译宏、Tensor API和内部KGemm接口说明。 |

## 免责声明

此代码仓仅对Eigen TensorContraction在ARM64处理器上的特定FP32矩阵乘路径进行性能优化，继承原生开源软件的设计与回退机制。任何漏洞与安全问题应由相应上游社区根据其响应机制解决。请密切关注Eigen上游社区发布的通知和版本更新。ARM64计算社区对软件漏洞及安全问题不承担任何责任。

## License

本项目遵循与Eigen主要代码相同的许可证（MPL-2.0），具体请参见[LICENSE](LICENSE)文件。补丁中个别上游文件仍可能适用Eigen发布包所列的兼容许可证。

## 贡献指南

如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交Issues联系我们。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交Issues，我们会尽快回复。感谢您的支持。

## 致谢

感谢Eigen社区及所有参与KGemm开发、移植和验证的贡献者，欢迎贡献！
