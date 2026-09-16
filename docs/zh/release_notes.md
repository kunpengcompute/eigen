# 版本说明书

## 版本配套说明

| 补丁目录 | 基线版本 | 兼容性说明 |
| --- | --- | --- |
| 3.4.0 | Eigen 3.4.0 | 面向3.4.0源码生成并验证 |
| 5.0.0 | Eigen 5.0.0 | 面向5.0.0生成；已验证可直接应用于官方5.0.1标签 |

### 产品版本信息

<a name="table62675726"></a>

<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p1597721693713"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p11923034"><a name="p11923034"></a><a name="p11923034"></a><span id="text14311218114"><a name="text14311218114"></a><a name="text14311218114"></a>26.2.RC1</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>Eigen TensorContraction KGemm</span></p>
</td>
</tr>
<tr id="row19308111718"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>V1.0.0</span></p>
</td>
</tr>
</tbody>
</table>

### 与操作系统、编译器和CPU配套说明

| 类别 | 已验证环境 |
| --- | --- |
| 操作系统 | openEuler 24.03 LTS（AArch64） |
| 编译器 | GCC12、Clang17 |
| CPU指令集 | ARMv8-A NEON |
| 线程配置 | 1、2、4、8线程 |

## 版本更新说明

### V1.0.0

1. 将`kgemm_neon_fp32_nn`接入TensorContraction调度层，并保留Eigen开源kernel回退路径。
2. 新增`TensorContractionKGemm.h`，隔离KGemm适配、阈值和多线程逻辑。
3. 支持ThreadPool并行执行及右矩阵packing复用，减少重复packing。
4. 分别提供Eigen 3.4.0和5.0.0的单文件补丁及SHA-256校验值。

## 已知限制

1. KGemm仅在AArch64 NEON FP32和支持的NN数据布局上启用。
2. 小尺寸或不满足阈值的contraction自动使用Eigen开源实现。
3. 补丁目录中仅声明已验证的Eigen版本；其他版本应先运行`git apply --check`并完成正确性测试。
4. 实际收益与矩阵形状、线程数、CPU频率、绑核和内存带宽有关。

## 版本配套文档

### V1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
| --- | --- |---|
| [《Eigen 安装指南》](installation_guide.md) | 环境、补丁应用、编译和测试方法 | 开源仓 |
| [《Eigen 快速入门》](quick_start.md) | 最短启用流程和使用示例 |开源仓 |
| [《Eigen API参考》](api_reference.md) | 宏、接口和调度约束 |开源仓 |

## 获取文档的方法

您可以通过访问[Eigen 开源仓](https://gitcode.com/boostkit/eigen)浏览和获取相关文档。
