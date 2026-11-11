# 快速入门

本文档介绍如何快速启用并验证Eigen TensorContraction KGemm优化，涵盖环境要求、补丁获取与应用、示例代码使能及性能对比验证，适用于在AArch64平台使用Eigen进行矩阵运算开发的读者。

## 环境要求

| 环境 | 要求 |
| --- | --- |
| 操作系统 | Linux AArch64 |
| 处理器 | 支持ARMv8-A NEON的AArch64处理器 |
| 编译器 | GCC 12或Clang 17，支持C++11；其他支持AArch64 NEON的版本也可使用 |
| 构建工具 | Git、CMake以及支持C++11的构建环境 |
| Eigen版本 | Eigen 3.4.0、5.0.0，或已确认兼容的5.0.1版本 |

## 使能ARM64优化

**获取补丁并应用到Eigen原版源码**

获取并应用补丁的步骤如下。

1. 克隆本仓库，获取优化补丁。

   ```bash
   git clone -b main https://gitcode.com/boostkit/eigen eigen
   ```

2. 克隆Eigen上游源码并检出目标版本。

   ```bash
   git clone https://gitlab.com/libeigen/eigen eigen-source
   git -C eigen-source checkout 5.0.0
   ```

3. 先校验补丁完整性，再使用`git apply --check`确认补丁可以应用。

   ```bash
   (cd eigen/5.0.0 && sha256sum -c SHA256SUMS) || exit 1
   git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
   ```

4. 使用`git apply`应用补丁。

   ```bash
   git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
   ```

Eigen 3.4.0请使用`3.4.0/eigen-3.4.0-kgemm.patch`。编译业务代码时定义`EIGEN_NEON_USE_KGEMM=1`。

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM=1 \
  -Ieigen-source your_program.cpp -o your_program
```

KGemm不满足调度条件时会自动回退到Eigen开源kernel，无需应用侧分支处理。

## 使用示例（包含头文件使能方法示例）

- 应用侧继续使用标准Tensor contraction接口。

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

- 多线程场景使用`Eigen::ThreadPoolDevice`。

   ```cpp
   Eigen::ThreadPool pool(thread_count);
   Eigen::ThreadPoolDevice device(&pool, thread_count);
   out.device(device) = lhs.contract(rhs, dims);
   ```

## 验证优化效果

使用相同输入、线程数、重复次数和CPU绑定条件运行NEON和KGemm版本，先确认计算结果一致，再比较GFLOPS或耗时。

## 更多功能

安装、宏定义和回退条件请参见《[安装指南](installation_guide.md)》及《[API参考](api_reference.md)》。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
