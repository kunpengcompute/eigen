# 快速入门

本章介绍如何快速启用并验证 Eigen TensorContraction KGemm 优化。

## 环境要求

- AArch64 Linux 系统及 ARMv8-A NEON 处理器。
- GCC 或 Clang，支持 C++11。
- Eigen 3.4.0、5.0.0，或已确认兼容的 5.0.1。

## 使能ARM64优化

### 获取补丁并应用到Eigen原版源码

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

3. 使用 `git apply --check` 校验补丁可以应用。

```bash
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

4. 使用 `git apply` 应用补丁。

```bash
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

Eigen 3.4.0 请使用 `3.4.0/eigen-3.4.0-kgemm.patch`。编译业务代码时定义
`EIGEN_NEON_USE_KGEMM=1`。

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM=1 \
  -Ieigen-source your_program.cpp -o your_program
```

KGemm 不满足调度条件时会自动回退到 Eigen 原生 kernel，无需应用侧分支处理。

## 使用示例（包含头文件使能方法示例）

- 应用侧继续使用标准 Tensor contraction 接口。

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

- 多线程场景使用 `Eigen::ThreadPoolDevice`。

```cpp
Eigen::ThreadPool pool(thread_count);
Eigen::ThreadPoolDevice device(&pool, thread_count);
out.device(device) = lhs.contract(rhs, dims);
```

## 使用示例（性能对比）

使用同一业务测试程序分别编译原生 NEON 和 KGemm 版本。

```bash
# 原生 NEON
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_neon

# KGemm NEON
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM=1 -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_kgemm
```

## 验证优化效果

使用相同输入、线程数、重复次数和 CPU 绑定条件运行两者，先确认计算结果一致，
再比较 GFLOPS 或耗时。

## 常见问题

安装、宏定义和回退条件请参见[安装指南](installation_guide.md)及[接口参考](api_reference.md)。

## 修订记录

| 发布日期 | 修订记录 |
| --- | --- |
| 2026-09-30 | 第一次正式发布。新增基于Eigen 3.4.0和5.0.0的ARM64 KGemm TensorContraction优化补丁。 |
