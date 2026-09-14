# 安装指南

本文介绍如何在Eigen 3.4.0、5.0.0及兼容版本中应用TensorContraction KGemm优化补丁。

## 环境要求

| 类别 | 要求 |
| --- | --- |
| 操作系统 | Linux AArch64，推荐openEuler 24.03 LTS |
| 处理器 | 支持ARMv8-A NEON的AArch64处理器 |
| 编译器 | GCC 12或Clang 17；其他支持AArch64 NEON的版本也可使用 |
| 构建工具 | Git、CMake以及支持C++11的构建环境 |

## 编译安装

### 获取代码

#### 获取发布补丁并应用到Eigen源码

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0          
(cd eigen/5.0.0 && sha256sum -c SHA256SUMS) || exit 1
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

Eigen3.4.0请将上述路径替换为`3.4.0/eigen-3.4.0-kgemm.patch`。

### 补丁完整性校验

请从可信的同一发布版本获取补丁及 `SHA256SUMS`。从发布仓库根目录执行对应版本的命令：

```bash
(cd 5.0.0 && sha256sum -c SHA256SUMS) || exit 1
# 使用 3.4.0 时：
(cd 3.4.0 && sha256sum -c SHA256SUMS) || exit 1
```

仅在输出补丁文件名及 `OK` 且命令退出码为 0 后继续应用。若出现 `FAILED`、文件缺失或非零退出码，
停止应用，重新从可信来源获取补丁与校验文件并再次验证。`git apply --check` 只检查补丁适用性，
不能替代 SHA-256 完整性检查；同源 SHA-256 清单本身也不等同于发布者的数字签名。

### 安装步骤

Eigen是头文件库。可以直接把源码根目录加入include路径，也可以使用CMake安装。

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/eigen-kgemm
cmake --build build
cmake --install build
```

## 编译选项说明

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `EIGEN_NEON_USE_KGEMM` | 在AArch64 NEON TensorContraction中启用KGemm路径 | 未定义（关闭） |
| `EIGEN_USE_THREADS` | 启用Eigen ThreadPool设备 | 未定义（关闭） |
| `EIGEN_NEON_KGEMM_REUSE_PACKING` | 启用多线程右矩阵packing复用 | `1` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN` | packing复用的M/N最小值 | `768` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K` | packing复用的K最小值 | `512` |

推荐编译参数。

```bash
-O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM
```

多线程测试还需增加`-DEIGEN_USE_THREADS`和`-pthread`。

## 运行测试

发布补丁不附带测试程序。可使用业务侧TensorContraction用例或Eigen上游测试，
分别构建开源NEON与KGemm配置，比较正确性和性能。KGemm构建参数示例如下。

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o tensor_contraction_test
./tensor_contraction_test
```

## 常见问题

### 补丁无法应用怎么办？

先确认当前源码对应补丁标注的Eigen版本，并使用`git apply --check`查看冲突位置。Eigen 5.0.0补丁已验证可直接应用于官方5.0.1标签。

### 定义宏后为什么没有进入KGemm？

KGemm仅适用于AArch64 NEON、FP32、非转置布局对应的TensorContraction，并受最小维度阈值约束。其他组合自动回退到Eigen开源路径。

### 如何启用多线程？

同时定义`EIGEN_USE_THREADS`，使用`Eigen::ThreadPoolDevice`执行contraction，并在链接时加入`-pthread`。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
