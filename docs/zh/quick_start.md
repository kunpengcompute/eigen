# 快速入门

本章介绍如何快速启用并验证Eigen TensorContraction KGemm优化。

## 环境要求

- AArch64 Linux系统及ARMv8-A NEON处理器。
- GCC或Clang，支持C++11。
- Eigen 3.4.0、5.0.0，或已确认兼容的5.0.1版本。

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

3. 使用`git apply --check`校验补丁可以应用。

   ```bash
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

- 多线程场景使用`Eigen::ThreadPoolDevice`。

```cpp
Eigen::ThreadPool pool(thread_count);
Eigen::ThreadPoolDevice device(&pool, thread_count);
out.device(device) = lhs.contract(rhs, dims);
```

## 使用示例（性能对比）

使用同一业务测试程序分别编译开源NEON和KGemm版本。

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_neon

g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM=1 -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o test_kgemm
```

## 验证优化效果

使用相同输入、线程数、重复次数和CPU绑定条件运行两者，先确认计算结果一致，再比较GFLOPS或耗时。

## 常见问题

安装、宏定义和回退条件请参见[安装指南](installation_guide.md)及[接口参考](api_reference.md)。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
