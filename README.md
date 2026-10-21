# DeepInfer-Ascend

> 让 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 支持**昇腾 950 以外的昇腾加速卡**。

上游 DeepInfer 的昇腾后端是围绕**昇腾 950**（`NpuArch=3510`）构建的：
从 kernel DSL、GEMM 库、通信库到数值格式，每一层都假定 950 的能力集，
README 写明的硬件支持是 "NVIDIA SM100 GPUs / Huawei Ascend 950 NPUs"，
依赖是 CANN 9.2 + `torch-npu 2.13`。

而昇腾在用的卡远不止 950。本仓库的目标是把「昇腾」这一层从「950」里拆出来，
让同一套引擎能落到 **A2（910B 系列）/ A3（910C、`Ascend910_93`）** 以及后续型号上。

> **状态：bring-up 与可行性验证阶段，尚不能用于推理。**
> 当前第一目标是 **910C（A3 / `NpuArch=2201`）**。

## 设计原则

1. **不新增加速器类型。** 910B/910C 就是 `ascend`，不该变成第四种 `AcceleratorKind`。
   要加的是 **ascend 内部的 soc 维度** —— 上游 `PlatformEnum` 目前只有 `GPU`/`NPU`，
   没有芯片粒度，这是扩展点。
2. **型号参数从 CANN 读，不写死。** 核数、UB 容量、峰值算力都在
   `$ASCEND_HOME/aarch64-linux/data/platform_config/<SoC>.ini` 里，
   运行时可查。上游把 950 的数值硬编码进了代码（见下）。
3. **能力按特性判断，不按型号判断。** 「有没有原生 FP8」比「是不是 910C」更稳定，
   新卡出来不用再改一遍。
4. **通用修正回上游，型号专有实现留本仓库。**

## 昇腾各代的关键差异

| | A2 / A3（910B、910C） | 950PR |
|---|---|---|
| `NpuArch` | **2201** | 3510 |
| `--npu-arch`（bisheng） | `dav-2201` | `dav-3510` |
| `CCEC_*_version`（另一套命名） | `dav-c220-cube` / `dav-c220-vec` | `dav-c310-cube` / `dav-c310-vec` |
| 原生 FP8 / MXFP8 / MXFP4 | ❌ | ✅ |
| 低精度加速路径 | INT8 / INT4 / INT2 | FP8 / FP4 |
| Cube 核 × 频率（910C 实测） | 24 × 1800 MHz | 32 × 1650 MHz |
| Unified Buffer | 192 KiB | 248 KiB |
| L2 | 192 MiB | 128 MiB |

⚠️ **两套 arch 命名不能混用。** `platform_config` 里写的是 CCEC 版本（`dav-c220-cube`），
而编译器 `--npu-arch` 要的是数字命名（`dav-2201`）。传错会得到
`bisheng: error: Unsupported NPU architecture or soc`。

## FP8 缺失：来自 CANN 自身的三重证据

这是 950 与其他昇腾代系最根本的分界，也是移植 DeepInfer 的核心障碍。

1. **头文件门控** —— `asc/include/simt_api/asc_fp8.h`：
   ```c
   #if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
   ```
   不含 2201。
2. **mmad 的 dtype 表** —— `Ascend910_9382.ini` 的 `[AICoreintrinsicDtypeMap]`：
   ```
   25=Intrinsic_mmad|u32u8u8,s32s8s8,s32u8s8,f16f16f16,f32f16f16,f16f16u2,
                     u8,s8,f162f16,f162f32,f16u2,u8s8,h322f32,f32f32f32,f322f32
   ```
   无任何 fp8 / e4m3 / e2m1 项。
3. **加速形状表** —— 同文件 `[DtypeMKN]` 只为 `DT_INT8`(16,32,16)、`DT_INT4`(16,64,16)、
   `DT_INT2`、`DT_UINT1` 提供加速形状，**没有 FP8/FP4 条目**。

而 DeepInfer 目前唯一支持的模型 `DeepSeek-V4.1-Flash` 的权重即
`{"quant_method":"fp8","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}`，
KV cache 的全部五种格式都是 `e4m3`/`e2m1`（框架内没有 BF16 KV cache 这个选项），
indexer 无条件调用 `deep_gemm.fp8_fp4_*`。

**→ 非 950 的昇腾卡只能走 INT8 W8A8 / W4A8 重量化**，与 vLLM-Ascend 在 A2/A3 上的既定做法一致。

## kernel DSL：tilelang 路线在 910C 上不通（已在 CANN 9.2 上定论）

tilelang 的 arch 是环境变量可配的，950 只是默认值：

```python
# tilelang/contrib/bisheng.py
def get_npu_arch(npu_arch: str | None = None) -> str:
    return npu_arch or os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
```

用官方算例 `examples/ascend/example_vecadd.py`，在 **CANN 9.2.0 + tilelang 0.1.15** 下，
每个组合独立进程（避开 tilelang 的进程内存缓存），并拦截 `bisheng.compile_ascend`
验证实际传入的 arch：

| 组合 | 实际 arch | 结果 |
|---|---|---|
| SimtVF `dav-3510` | `dav-3510` | ✅ 编译通过 |
| SimdVF `dav-3510` | `dav-3510` | ✅ 编译通过 |
| **SimtVF `dav-2201`** | `dav-2201` | ❌ 失败 |
| **SimdVF `dav-2201`** | `dav-2201` | ❌ 失败 |

编译器诊断（华为自己的头文件与 bisheng）：

```
asc/include/simt_api/math_functions.h:22: error:
    attribute __simd_callee__/__simt_callee__/__simd_vf__/__simt_vf__ is not [supported]
asc/include/simt_api/device_functions.h:132: error: unknown type name 'long2'
bisheng: error: __simt_vf__ is not supported in function: asc_syncthreads
    on current architecture
```

**→ VF（vector function）编程模型在 arch 2201 上不支持，是代际差异而非 CANN 版本问题。**
CANN 9.2 已是最新版：950 两种模式都编得过，910C 两种模式都编不过。

DeepInfer 的 `tilelang_ops/asc/` 共 30 个文件，其中 **24 个**使用
`T.SimdVF`（66 处）/ `T.SimtVF`（43 处）。
**kernel 层对 910C 需要用 Ascend C（非 VF）重写，而不是适配。**

### 可行路线

| 路线 | 状态 |
|---|---|
| tilelang + 现有 DeepInfer kernel | ❌ VF 模型 2201 不支持 |
| Ascend C 手写（如 DeepSelect 的实现方式，非 VF） | 可行，但等于重写 30 个 kernel |
| DeepInfer 引擎层（调度器 / sampler / 异步调度 / 结构化输出） | 平台无关，可直接复用 |
| kernel 与通信层 | 需整体替换（HCCL `MoeDistributeDispatch*` 取代 DeepEP-Ascend 的 UB 路径） |

## 已定位、可直接修正的点

| 位置 | 上游现状 | 应改为 |
|---|---|---|
| `deep_cluster/hardware/device.py` `get_bf16_tflops_limit()` | 硬编码 `432` | 按 soc 查 platform_config。432 = 32 核 × 16×16×16 MAC × 2 FLOP × 1.65 GHz（950）；910C 同式得 **354** |
| `deep_cluster/hardware/ascend.py` 设备发现 | 首选 `/sys/class/ubcore` + `ub_totalues`（UnifiedBus，950 特性） | 无 ubcore 的卡落 `/dev/davinci*` 回退路径 |
| 同上 `device_numa_node()` | 回退路径拿到 numa=-1 后直接 `raise RuntimeError` | 按一卡多 die 拓扑重做映射 |
| `deepinfer/platform.py` `_resolve_acl_host_register_api()` | 符号缺失直接 `RuntimeError` | 基类 `pin_host_memory` 本就是 no-op，应降级而非硬失败 |
| `deepinfer/platform.py` `TORCH_NPU_DEVICE_CAPABILITY="9.0"` | 注释自称 "hack it temporarily" | 按 soc 给值 |
| `requirements-ascend.txt` | `torch-npu==2.13.0rc1` + CANN 9.2 | A2/A3 的官方配套上限是 CANN 9.1.X + `torch-npu 2.12.0.post2` |

### host pin 在 910C 上是静默空操作

按 `deepinfer/shared_memory.py` 的真实路径（POSIX shm + `torch.from_numpy` + `pin_host_memory`）
在 910C 实测，2 GiB buffer、写满真实页、每项 5 轮，H2D 中位带宽：

| 模式 | GB/s |
|---|---|
| `torch pin_memory=True`（参照） | **59.3** |
| `aclrtMallocHost`（参照） | **59.3** |
| shm 未 pin | 15.8 |
| shm + `ACL_HOST_REG_PINNED`（上游现状） | 17.1 |
| shm + `ACL_HOST_REG_MAPPED` | 24.0 |

`aclrtHostRegisterV2` 返回 0、`is_pinned()` 翻 `True`，但带宽只有真锁页内存的四成 ——
且因为 `is_pinned()` 为真，`non_blocking=True` 的拷贝会被当作真异步发出。
换 flag 解决不了（两者都远低于 59.3）；方向是 shm 能否改用 `aclrtMallocHost`，
涉及跨进程共享语义，非一行改动。

## 关系与分工

- **上游**：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- **通用修正**（设备发现回退、硬编码常量、硬失败降级）走 fork
  [lucifer-holly/DeepInfer](https://github.com/lucifer-holly/DeepInfer) 的分支，向上游提 PR
- **昇腾型号专有的实现**留在本仓库
- 上游已有一份方向相同的 draft RFC（deepseek-ai/DeepInfer#46，纯文档、无实现）

## License

随上游，Apache License 2.0。
