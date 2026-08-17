# 版本说明书

## 版本配套说明

| 补丁目录 | 基线版本 | 兼容性说明 |
| --- | --- | --- |
| `3.4.0` | Eigen 3.4.0 | 面向 3.4.0 源码生成并验证 |
| `5.0.0` | Eigen 5.0.0 | 面向 5.0.0 生成；已验证可直接应用于官方 5.0.1 标签 |

### 产品版本信息

| 项目 | 内容 |
| --- | --- |
| 名称 | Eigen TensorContraction KGemm 优化补丁 |
| 当前版本 | V1.0.0 |
| 发布日期 | 2026.08.12 |
| 开源协议 | Mozilla Public License 2.0 |

### OS、编译器及CPU

| 类别 | 已验证环境 |
| --- | --- |
| 操作系统 | openEuler 24.03 LTS（AArch64） |
| 编译器 | GCC 12、Clang 17 |
| CPU 指令集 | ARMv8-A NEON |
| 线程配置 | 1、2、4、8 线程 |

## 版本更新说明

### V1.0.0

1. 将 `kgemm_neon_fp32_nn` 接入 TensorContraction 调度层，并保留 Eigen 原生 kernel 回退路径。
2. 新增 `TensorContractionKGemm.h`，隔离 KGemm 适配、阈值和多线程逻辑。
3. 支持 ThreadPool 并行执行及右矩阵 packing 复用，减少重复 packing。
4. 分别提供 Eigen 3.4.0 和 5.0.0 的单文件补丁及 SHA-256 校验值。

## 已知限制

1. KGemm 仅在 AArch64 NEON FP32 和支持的 NN 数据布局上启用。
2. 小尺寸或不满足阈值的 contraction 自动使用 Eigen 原生实现。
3. 补丁目录中仅声明已验证的 Eigen 版本；其他版本应先运行 `git apply --check` 并完成正确性测试。
4. 实际收益与矩阵形状、线程数、CPU 频率、绑核和内存带宽有关。

## 版本配套文档

### V1.0.0版本配套文档

| 文档 | 说明 |
| --- | --- |
| [安装指南](installation_guide.md) | 环境、补丁应用、编译和测试方法 |
| [快速入门](quick_start.md) | 最短启用流程和使用示例 |
| [接口参考](api_reference.md) | 宏、接口和调度约束 |

## 获取文档的方法

您可以通过访问开源仓（https://gitcode.com/boostkit/eigen）浏览和获取相关文档。
