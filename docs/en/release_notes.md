# Release Notes

## Version Compatibility Notes

| Patch Directory | Baseline Version | Compatibility Notes |
| --- | --- | --- |
| `3.4.0` | Eigen 3.4.0 | Generated and verified against the 3.4.0 source code |
| `5.0.0` | Eigen 5.0.0 | Generated against 5.0.0; verified to apply to the official 5.0.1 tag |

### Product Version Information

<a name="table62675726"></a>

<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p1597721693713"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p11923034"><a name="p11923034"></a><a name="p11923034"></a><span id="text14311218114"><a name="text14311218114"></a><a name="text14311218114"></a>26.2.RC1</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>Software Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>Eigen TensorContraction KGemm</span></p>
</td>
</tr>
<tr id="row19308111718"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>Software Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a><span id="text17191017111119"><a name="text17191017111119"></a><a name="text17191017111119"></a>V1.0.0</span></p>
</td>
</tr>
</tbody>
</table>

### OS, Compiler, and CPU

| Category | Verified Environment |
| --- | --- |
| OS | openEuler 24.03 LTS (AArch64) |
| Compiler | GCC 12, Clang 17 |
| CPU instruction set | Armv8-A NEON |
| Thread configuration | 1, 2, 4, 8 threads |

## Version Update Notes

### V1.0.0

1. Integrated `kgemm_neon_fp32_nn` into the TensorContraction dispatch layer while retaining the Eigen open-source kernel fallback path.

2. Added `TensorContractionKGemm.h` to isolate KGemm adaptation, threshold, and multithreading logic.

3. Supports ThreadPool parallel execution and right-hand side matrix packing reuse to reduce redundant packing.

4. Provided single-file patches and SHA-256 checksums for Eigen 3.4.0 and 5.0.0.

## Known Limitations

1. KGemm is enabled only on AArch64 NEON FP32 and the supported NN data layout.

2. Contractions of small sizes or those that do not meet the threshold automatically fall back to the open-source Eigen implementation.

3. The patch directory only lists Eigen versions that have been verified. For other versions, run `git apply --check` first and complete correctness testing before applying.

4. Actual gains depend on matrix shape, thread count, CPU frequency, CPU pinning, and memory bandwidth.

## Documentation

### V1.0.0 Documents

| Document | Description |
| --- | --- |
| [Eigen Installation Guide](installation_guide.md) | Environment, patch application, compilation, and testing methods |
| [Eigen Quick Start](quick_start.md) | Shortest enablement process and usage examples |
| [Eigen API Reference](api_reference.md) | Macros, interfaces, and dispatch constraints |

## Obtaining the Documentation

You can browse and obtain the relevant documentation by visiting the [Eigen open-source repository](https://gitcode.com/boostkit/eigen).
