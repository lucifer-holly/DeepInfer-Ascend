# DeepInfer-Ascend

面向**昇腾 910C（Atlas A3 / `Ascend910_93` 系列）** 的 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 适配。

上游 DeepInfer 的昇腾后端面向**昇腾 950**（`dav-3510` / `NpuArch=3510`）构建，
要求 CANN 9.2、`torch-npu 2.13`。本仓库记录并实现在 910C（`dav-c220` / `NpuArch=2201`）
上运行所需的改动。

> **状态：早期调研与 bring-up。** 尚不可用于推理。

## 为什么不是「改几个 if」

910C 与 950 是不同代的架构，差异集中在数值格式而非代码分支：

| | 昇腾 950 (`NpuArch=3510`) | 昇腾 910C (`NpuArch=2201`) |
|---|---|---|
| 原生 FP8 / MXFP8 / MXFP4 | ✅ | ❌ |
| 低精度加速路径 | FP8 / FP4 | INT8 / INT4 |
| Cube 核数 × 频率 | 32 × 1650 MHz | 24 × 1800 MHz |
| Unified Buffer | 248 KiB | 192 KiB |
| L2 | 128 MiB | 192 MiB |

FP8 缺失有三重来自 CANN 自身的证据：

1. `asc/include/simt_api/asc_fp8.h` 的门控是 `#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)` —— 不含 2201
2. 平台配置 `Ascend910_9382.ini` 的 `Intrinsic_mmad` dtype 表中无任何 fp8 / e4m3 / e2m1 项
3. 同文件 `[DtypeMKN]` 只为 `DT_INT8` / `DT_INT4` / `DT_INT2` / `DT_UINT1` 提供加速形状，无 FP8/FP4 条目

而 DeepInfer 当前唯一支持的模型 `DeepSeek-V4.1-Flash` 的权重即为
`{"quant_method": "fp8", "expert_dtype": "fp4"}`，KV cache 的全部五种格式也都是 `e4m3`/`e2m1`，
indexer 无条件调用 `deep_gemm.fp8_fp4_*`。因此 910C 路线只能走 **INT8 W8A8 / W4A8** 重量化。

## 已确认可直接修正的点

| 位置 | 现状 | 910C |
|---|---|---|
| `deep_cluster/hardware/device.py` `get_bf16_tflops_limit()` | 硬编码 `432` | `354`（24 cube核 × 16×16×16 MAC × 2 FLOP × 1.8 GHz） |
| `deep_cluster/hardware/ascend.py` 设备发现 | 首选 `/sys/class/ubcore`（UnifiedBus，950 特性） | 无 ubcore，走 `/dev/davinci*` 回退 |
| 同上 `device_numa_node()` | 回退路径返回 numa=-1 后直接 `raise` | 需按一卡双 die 拓扑重做 |
| `deepinfer/platform.py` `_resolve_acl_host_register_api()` | 符号缺失直接 `RuntimeError` | 基类 `pin_host_memory` 本就是 no-op，应降级 |
| `deepinfer/platform.py` `pin_host_memory()` | `aclrtHostRegisterV2(PINNED)` | 实测在 910C 上拿不到 DMA 快路径（详见下） |

### host pin 的实测

按 `deepinfer/shared_memory.py` 的真实路径（POSIX shm + `torch.from_numpy`）在 910C 上实测，
2 GiB buffer、写满真实页、每项 5 轮，H2D 中位带宽：

| 模式 | GB/s |
|---|---|
| `torch pin_memory=True`（参照） | **59.3** |
| `aclrtMallocHost`（参照） | **59.3** |
| shm 未 pin | 15.8 |
| shm + `ACL_HOST_REG_PINNED`（上游现状） | 17.1 |
| shm + `ACL_HOST_REG_MAPPED` | 24.0 |

注册 POSIX shm 到不了真锁页内存的 DMA 路径。`aclrtHostRegisterV2` 返回 0、
`is_pinned()` 翻 `True`，带宽只有四成 —— 是个**看起来成功的空操作**。
换 flag 解决不了（两者都远低于 59.3）；方向是 shm 能否改用 `aclrtMallocHost`，
但涉及跨进程共享语义。

## 关系

- 上游：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- 通用性的修正走 fork [lucifer-holly/DeepInfer](https://github.com/lucifer-holly/DeepInfer) 向上游提 PR；
  910C 专有的实现留在本仓库
- 上游已有一份方向相同的 draft RFC：deepseek-ai/DeepInfer#46（纯文档，无实现）

## License

随上游，Apache License 2.0。
