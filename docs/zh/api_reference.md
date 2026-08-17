# API参考

## 接口说明

KGemm优化通过编译宏接入Eigen公开的TensorContraction表达式，不新增用户必须调用的公开Eigen API。主要配置和内部接口如[**表 1** KGemm接口列表](#KGemm接口列表)所示。

**表 1** KGemm接口列表<a id="KGemm接口列表"></a>

|接口名称|接口说明|
|--|--|
|`EIGEN_NEON_USE_KGEMM`|启用ARM64 NEON KGemm TensorContraction特化。|
|`EIGEN_NEON_KGEMM_REUSE_PACKING`|控制大尺寸TensorContraction是否复用调度器packing，默认值为1。|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN`|packing复用的有效M/N最小值，默认值为768。|
|`EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K`|packing复用的K最小值，默认值为512。|
|`Tensor::contract`|用户侧TensorContraction入口，接口与原生Eigen一致。|
|`kgemm_neon_fp32_nn`|内部连续FP32 NN GEMM驱动。|
|`kgemm_neon_fp32_nn_packed`|内部已packing FP32 GEMM驱动。|

## 编译宏定义

### EIGEN_NEON_USE_KGEMM

**宏功能**

为满足条件的AArch64 NEON FP32 TensorContraction注册KGemm kernel特化。

**定义方法**

```text
-DEIGEN_NEON_USE_KGEMM=1
```

所有包含Eigen Tensor头文件并参与同一程序链接的翻译单元应使用相同宏配置。未定义或设置为0时，使用Eigen原生实现。

### EIGEN_NEON_KGEMM_REUSE_PACKING

**宏功能**

控制大尺寸KGemm是否使用TensorContraction共享block buffer完成一次packing并复用结果。

**定义方法**

```text
-DEIGEN_NEON_KGEMM_REUSE_PACKING=0
```

默认值为1。设置为0可构建no-reuse性能对照版本。

### Packing复用阈值

```text
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN=768
-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_K=512
```

只有规范化后的有效M/N均达到M/N阈值且K达到K阈值时，才使用packed路径。

## TensorContraction接口

### Tensor::contract

**函数功能**

沿指定维度对两个Tensor进行收缩。补丁保持Eigen公开接口不变，只改变满足选择条件时使用的内部kernel。

**调用形式**

```cpp
Eigen::array<Eigen::IndexPair<int>, 1> dims = {
    Eigen::IndexPair<int>(1, 0)};
output.device(device) = lhs.contract(rhs, dims);
```

**选择条件**

|条件|要求|
|--|--|
|架构|AArch64且编译器启用NEON|
|标量类型|输入和输出均为`float`|
|表达式|二维、一个收缩维度、规范连续存储|
|mapper|左右输入支持RawAccess，内维连续，RHS未重排|
|设备|`DefaultDevice`或`ThreadPoolDevice`|

条件不满足时自动回退到Eigen原生packing和GEBP。

**示例**

```cpp
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<float, 2, Eigen::RowMajor> a(64, 128);
Eigen::Tensor<float, 2, Eigen::RowMajor> b(128, 96);
Eigen::Tensor<float, 2, Eigen::RowMajor> c(64, 96);
Eigen::array<Eigen::IndexPair<int>, 1> dims = {
    Eigen::IndexPair<int>(1, 0)};
c = a.contract(b, dims);
```

## 内部KGemm接口

### kgemm_neon_fp32_nn

```cpp
static inline void kgemm_neon_fp32_nn(
    float* a, float* b, float* c,
    long int lda, long int ldb, long int ldc,
    long int m, long int n, long int k, long int addc);
```

该接口接收连续的RowMajor NN矩阵视图。`addc=0`表示覆盖输出，非0表示累加。它是Eigen内部接口，不提供ABI稳定性保证。

### kgemm_neon_fp32_nn_packed

```cpp
static inline void kgemm_neon_fp32_nn_packed(
    float* pa, float* pb, float* c,
    long int ldc, long int m, long int n,
    long int k, long int addc);
```

`pa`使用4行K-major panel布局，`pb`使用16列K-major panel布局。该接口由TensorContraction调度器调用，不建议业务代码直接调用。

## 修订记录

| 发布日期 | 修订记录 |
| --- | --- |
| 2026-09-30 | 第一次正式发布。新增基于Eigen 3.4.0和5.0.0的ARM64 KGemm TensorContraction优化补丁。 |
