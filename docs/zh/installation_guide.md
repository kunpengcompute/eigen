# 安装指南

本文介绍如何在 Eigen 3.4.0、5.0.0 及兼容版本中应用 TensorContraction KGemm 优化补丁。

## 环境要求

| 类别 | 要求 |
| --- | --- |
| 操作系统 | Linux AArch64，推荐 openEuler 24.03 LTS |
| 处理器 | 支持 ARMv8-A NEON 的 AArch64 处理器 |
| 编译器 | GCC 12 或 Clang 17；其他支持 AArch64 NEON 的版本也可使用 |
| 构建工具 | Git、CMake 以及支持 C++11 的构建环境 |

## 编译安装

### 获取代码

#### 获取发布补丁并应用到Eigen源码

```bash
git clone -b main https://gitcode.com/boostkit/eigen eigen
git clone https://gitlab.com/libeigen/eigen eigen-source
git -C eigen-source checkout 5.0.0          # 或 3.4.0
git -C eigen-source apply --check ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
git -C eigen-source apply ../eigen/5.0.0/eigen-5.0.0-kgemm.patch
```

Eigen 3.4.0 请将上述路径替换为 `3.4.0/eigen-3.4.0-kgemm.patch`。

### 安装步骤

Eigen 是头文件库。可以直接把源码根目录加入 include 路径，也可以使用 CMake 安装：

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/eigen-kgemm
cmake --build build
cmake --install build
```

## 编译选项说明

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `EIGEN_NEON_USE_KGEMM` | 在 AArch64 NEON TensorContraction 中启用 KGemm 路径 | 未定义（关闭） |
| `EIGEN_USE_THREADS` | 启用 Eigen ThreadPool 设备 | 未定义（关闭） |
| `EIGEN_NEON_KGEMM_REUSE_PACKING` | 启用多线程右矩阵 packing 复用 | `1` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN` | packing 复用的 M/N 最小值 | `768` |
| `EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K` | packing 复用的 K 最小值 | `512` |

推荐编译参数：

```bash
-O3 -DNDEBUG -march=armv8-a -DEIGEN_NEON_USE_KGEMM
```

多线程测试还需增加 `-DEIGEN_USE_THREADS` 和 `-pthread`。

## 运行测试

发布补丁不附带测试程序。可使用业务侧 TensorContraction 用例或 Eigen 上游测试，
分别构建原生 NEON 与 KGemm 配置，比较正确性和性能。KGemm 构建参数示例：

```bash
g++ -O3 -DNDEBUG -march=armv8-a -DEIGEN_USE_THREADS \
  -DEIGEN_NEON_USE_KGEMM -Ieigen-source \
  tensor_contraction_test.cpp -pthread -o tensor_contraction_test
./tensor_contraction_test
```

## 常见问题

### 补丁无法应用怎么办？

先确认当前源码对应补丁标注的 Eigen 版本，并使用 `git apply --check` 查看冲突位置。Eigen 5.0.0 补丁已验证可直接应用于官方 5.0.1 标签。

### 定义宏后为什么没有进入 KGemm？

KGemm 仅适用于 AArch64 NEON、FP32、非转置布局对应的 TensorContraction，并受最小维度阈值约束。其他组合自动回退到 Eigen 原生路径。

### 如何启用多线程？

同时定义 `EIGEN_USE_THREADS`，使用 `Eigen::ThreadPoolDevice` 执行 contraction，并在链接时加入 `-pthread`。
