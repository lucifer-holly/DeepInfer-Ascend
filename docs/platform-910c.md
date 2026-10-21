# 昇腾 A2/A3 平台事实

实测平台：`Ascend910_9382`（910C，A3 系列），CANN 9.0.1 与 9.2.0 下结论一致。
数据取自 CANN 自带的 `data/platform_config/<SoC_version>.ini` —— 这是 CANN
对本机的权威描述，可在任意 CANN 安装中复核。
完整原文见 [`reference/Ascend910_9382.ini`](../reference/Ascend910_9382.ini)。

## ⚠️ 两套 arch 命名不可混用

这是移植时最容易先栽进去的坑。

| 用途 | 910C | 950PR | 来源 |
|---|---|---|---|
| 编译器 `--npu-arch=` | **`dav-2201`** | **`dav-3510`** | bisheng 实际接受的值 |
| `CCEC_*_version` | `dav-c220-cube` / `dav-c220-vec` | `dav-c310-cube` / `dav-c310-vec` | `platform_config` 的 INI |
| `NpuArch` | `2201` | `3510` | 同上 |

把 INI 里的 `dav-c220` 当成 `--npu-arch` 传进去会得到：

```
bisheng: error: Unsupported NPU architecture or soc: NPU arch is not supported!!!
LLVM ERROR: PlugIn Error: no aicore arch specified.
```

实测只有**数字命名**能过 arch 检查；`dav-c220` / `dav-c220-cube` / `dav-c220-vec` /
`dav-c310` / `dav-c100` / `dav-m200` 一律被拒。
复核用 [`tools/probe_bisheng_arch.sh`](../tools/probe_bisheng_arch.sh)。

## 规格对照

| | 910C (`Ascend910_9382`) | 950PR (`Ascend950PR_9589`) |
|---|---|---|
| `cube_core_cnt` / `vector_core_cnt` | 24 / 48 | 32 / 64 |
| `cube_freq` | 1800 MHz | 1650 MHz |
| `cube_m/n/k_size` | 16 / 16 / 16 | 16 / 16 / 16 |
| `ub_size` | **196608（192 KiB）** | 253952（248 KiB） |
| `l1_size` / `l0_c_size` | 524288 / 131072 | — |
| `l2_size` | 192 MiB | 128 MiB |
| `memory_size` | 64 GiB / die | — |
| `support_bf16` / `support_fixpipe` | 1 / 1 | 1 / 1 |

### BF16 cube 峰值可以算出来

```
cube_core_cnt × (cube_m × cube_n × cube_k MAC × 2 FLOP) × cube_freq

910C   : 24 × 8192 × 1.80 GHz = 353.9 TFLOP/s
950PR  : 32 × 8192 × 1.65 GHz = 432.5 TFLOP/s
```

**上游 `get_bf16_tflops_limit()` 硬编码的 432 正是 950 的 32 核 SKU 这一支** ——
但同代系内 28 核是 378、36 核是 530，所以那个常量在 950 上就已经不准。
已向上游提 PR：[deepseek-ai/DeepInfer#170](https://github.com/deepseek-ai/DeepInfer/pull/170)。

⚠️ 这是**单 die** 的值（DeepInfer 每 rank 一个 die）。对外报告若按整卡口径，
不要和这个数混用。

### `ub_size` 影响已有代码

`DeepSelect` 的 `TOPK_UB_LIMIT_BYTES = 248 * 1024` = 253952，恰是 **950 的 `ub_size` 原值**。
910C 是 196608，不重做 tiling 会越界。

## NUMA 亲和性在这台机器上不存在

不是解析失败，是数据本身没有：

- 所有 davinci PCI 设备 `numa_node = -1`，`local_cpulist = 0-639`（全部 CPU）
- `npu-smi info -t topo` 只给互联拓扑，**没有 CPU 亲和性列**，
  所以 `deep_cluster/hardware/ascend.py::_npu_smi_numa_nodes()` 返回空
- 于是 `device_numa_node()` 走到 `raise RuntimeError("NUMA metadata is unavailable")`

**不是阻断** —— `worker/base.py::_bind_numa` 本来就裹了 `try/except Exception`，
失败只打 warning、退化成 first-touch。但日志里会反复出现，
且 `ascend.py` 里 `(raw_numa // 2) * 2 + index % 2` 这个映射式在这台机器上无从验证。

### 互联拓扑

`npu-smi info -t topo` 显示 die `2k` 与 `2k+1` 之间是 `SIO`，其余为 `HCCS_SW`
—— 即一卡双 die 的物理结构。**没有 UnifiedBus**，所以
`deep_cluster/hardware/ascend.py` 首选的 `/sys/class/ubcore` + `ub_totalues`
发现路径在 A2/A3 上恒为空，会落到 `/dev/davinci*` 回退分支（该分支工作正常）。

## 不依赖 torch 拿 SoC 版本

`deep_cluster/hardware/*` 只用 procfs / sysfs / ctypes，不能 import torch。
两条候选实测结果：

| 方式 | 返回 | 可用性 |
|---|---|---|
| `libascendcl.so` → `aclrtGetSocName()` | **`Ascend910_9382`** | ✅ 不需要 `aclInit`，不建 device context |
| `libdcmi.so` → `dcmi_get_device_chip_info()` | `Ascend910`（chip_name） | ❌ 粒度不够，区分不了核数不同的 9382 / 9372 |

`aclrtGetSocName()` 返回的字符串正好是 `platform_config/<SoC_version>.ini` 的索引键。
复核用 [`tools/probe_soc_version.py`](../tools/probe_soc_version.py)。

## torch_npu 侧的设备属性

`torch.npu.get_device_properties(0)`（需要 device context，仅供对照）：

```
cube_core_num = 24      vector_core_num = 48      multi_processor_count = 48
L2_cache_size = 201326592    total_memory = 65787658240    name = 'Ascend910_9382'
```

⚠️ **`multi_processor_count` 等于 vector 核数，不是 cube 核数。**
`deepinfer/model_executor/parallel_utils/userbuffer.py:86-90` 把它当 NVIDIA 的 SM 数
用 `min(32, device_num_sms // 4)`，在昇腾上语义不对应。

## 工具链版本

| | 实测 |
|---|---|
| 内核 | `5.10.0-216.0.0.115.oe2203sp4.aarch64`（openEuler 22.03 SP4） |
| 驱动 / 固件 | `25.5.0` / `7.8.0.5.216` |
| 容器栈 | `torch 2.7.1+cpu` / `torch_npu 2.7.1.post8` / CANN 9.0.1 / Python 3.12.13 / GCC 11.4 |
| 官方配套上限（A2/A3） | CANN 9.1.X + `torch_npu 2.12.0.post2`（`Ascend/pytorch` 的 COMPATIBILITY.md） |
| 上游 DeepInfer 要求 | CANN 9.2 + `torch 2.13.0` + `torch-npu 2.13.0rc1` + GCC ≥ 13 |

**CANN 9.2 不是 950 专属** —— A3 上实测可安装可用，且华为为 9.2 发了
`Ascend-cann-A3-ops` 专用算子包。950 专属的是 tilelang 的 codegen 与
DeepSeek 那几个 kernel 库，不是 CANN 版本本身。
