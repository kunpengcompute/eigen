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

### 实测性能参考

CPU：Kunpeng 950；操作系统：openEuler 24.03 LTS-SP3。

| 版本 | 线程 | M×K×N | NEON 毫秒 | KGEMM 毫秒 | 加速比 |
| --- | ---: | --- | ---: | ---: | ---: |
| 3.4.0 | 1 | 128×128×128 | 0.0846 | 0.0733 | 1.154× |
| 3.4.0 | 1 | 512×512×512 | 5.2108 | 4.5442 | 1.147× |
| 3.4.0 | 1 | 768×512×768 | 11.5808 | 10.0959 | 1.147× |
| 3.4.0 | 1 | 781×513×783 | 12.5720 | 10.7063 | 1.174× |
| 3.4.0 | 4 | 128×128×128 | 0.0397 | 0.0344 | 1.153× |
| 3.4.0 | 4 | 512×512×512 | 1.8079 | 1.5148 | 1.193× |
| 3.4.0 | 4 | 768×512×768 | 4.2860 | 2.9688 | 1.444× |
| 3.4.0 | 4 | 781×513×783 | 4.3294 | 3.1083 | 1.393× |
| 5.0.0 | 1 | 128×128×128 | 0.0899 | 0.0735 | 1.223× |
| 5.0.0 | 1 | 512×512×512 | 5.5757 | 4.5462 | 1.226× |
| 5.0.0 | 1 | 768×512×768 | 12.4079 | 10.1028 | 1.228× |
| 5.0.0 | 1 | 781×513×783 | 13.3818 | 10.6886 | 1.252× |
| 5.0.0 | 4 | 128×128×128 | 0.0512 | 0.0340 | 1.505× |
| 5.0.0 | 4 | 512×512×512 | 1.8540 | 1.3577 | 1.370× |
| 5.0.0 | 4 | 768×512×768 | 3.5875 | 2.9963 | 1.197× |
| 5.0.0 | 4 | 781×513×783 | 3.7806 | 3.0478 | 1.241× |

## 常见问题

安装、宏定义和回退条件请参见[安装指南](installation_guide.md)及[接口参考](api_reference.md)。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
