# 实测结论

全部结论都在 `Ascend910_9382`（910C / A3，`NpuArch=2201`）上实测得到，
每条都附可复核的手段。日期：2026-10-21。

---

## 1. FP8/FP4 在 A2/A3 上不存在 —— 四重独立证据

这是整个移植的核心障碍。注意这**不是**「某个 kernel 没写」，是硬件与运行时层面就没有。

### 1.1 CANN 头文件的编译期门控

`asc/include/simt_api/asc_fp8.h`：

| CANN 版本 | 门控 |
|---|---|
| 9.0.1 | `#if (__NPU_ARCH__ == 3510) \|\| (__NPU_ARCH__ == 5102)` |
| 9.2.0 | `#if (__NPU_ARCH__ == 3510)` |

`2201` 两版都不在内，9.2 反而收得更紧。
→ [`tools/probe_fp8_gating.sh`](../tools/probe_fp8_gating.sh)

### 1.2 mmad 支持的 dtype 表

`platform_config/Ascend910_9382.ini` 的 `[AICoreintrinsicDtypeMap]`：

```
25=Intrinsic_mmad|u32u8u8,s32s8s8,s32u8s8,f16f16f16,f32f16f16,f16f16u2,
                  u8,s8,f162f16,f162f32,f16u2,u8s8,h322f32,f32f32f32,f322f32
```

无任何 `fp8` / `e4m3` / `e2m1` 项。

### 1.3 加速形状表只给整数低精度

同文件 `[DtypeMKN]`：

```
Default = 16,16,16
DT_INT8 = 16,32,16     ← 2× N 维吞吐
DT_INT4 = 16,64,16     ← 4×
DT_INT2 = 16,128,16    DT_UINT1 = 16,256,16
```

**没有 FP8/FP4 条目。** → A2/A3 的低精度路线是 **INT8 / INT4**，
与 vLLM-Ascend 在 A2/A3 上走 W8A8 / W4A8 的做法一致。

### 1.4 运行时连 FP8 张量清零都拒绝（最直观的一条）

```
RuntimeError: zero_: call aclnnInplaceZero failed, error code is 161002
AclNN_Parameter_Error(EZ1001): Tensor self not implemented for DT_FLOAT8_E4M3FN,
  should be in dtype support list [DT_INT8,DT_INT32,DT_INT64,DT_UINT8,DT_FLOAT16,
  DT_FLOAT,DT_BOOL,DT_DOUBLE,DT_INT16,DT_UINT16,DT_COMPLEX128,DT_COMPLEX64,DT_BFLOAT16,]
```

不是缺某个 kernel —— `torch.zeros` 都不行。

---

## 2. tilelang 的昇腾后端在 A2/A3 上不可用

**不是版本问题，是 VF（vector function）编程模型 arch 2201 不支持。**

tilelang 的 arch 是环境变量可配的，950 只是默认值：

```python
# tilelang/contrib/bisheng.py
def get_npu_arch(npu_arch: str | None = None) -> str:
    return npu_arch or os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
```

用官方算例 `examples/ascend/example_vecadd.py`，在 **CANN 9.2.0 + tilelang 0.1.15** 下：

| 组合 | 实际传入的 arch | 结果 |
|---|---|---|
| SimtVF `dav-3510` | `dav-3510` | ✅ 编译通过 |
| SimdVF `dav-3510` | `dav-3510` | ✅ 编译通过 |
| **SimtVF `dav-2201`** | `dav-2201` | ❌ 失败 |
| **SimdVF `dav-2201`** | `dav-2201` | ❌ 失败 |

华为自己的头文件与编译器给出的诊断：

```
asc/include/simt_api/math_functions.h:22: error:
    attribute __simd_callee__/__simt_callee__/__simd_vf__/__simt_vf__ is not [supported]
asc/include/simt_api/device_functions.h:132: error: unknown type name 'long2'
bisheng: error: __simt_vf__ is not supported in function: asc_syncthreads
    on current architecture
```

→ [`tools/probe_tilelang_arch.sh`](../tools/probe_tilelang_arch.sh)

### 对 DeepInfer 的含义

`deepinfer/model_executor/layers/tilelang_ops/asc/` 共 30 个文件，其中 **24 个**
使用 `T.SimdVF`（66 处）/ `T.SimtVF`（43 处）。
**kernel 层需要用 Ascend C（非 VF）或 Triton 重写，不是适配。**

### 🔴 这一轮差点得出两次相反的错误结论

| | 假象 | 真相 |
|---|---|---|
| 第一次 | 8/8 编译失败 →「910C 不支持」 | 是我在脚本顶上加了 `from __future__ import annotations`，导致 `T.prim_func` 的 `get_type_hints()` 取不到闭包变量。**`dav-3510` 对照组同样失败**才看出是脚本自身问题 |
| 第二次 | CANN 9.2 下 8/8 编译成功 →「910C 可以！」 | tilelang 的**进程内存缓存**。`ASCEND_NPU_ARCH` 是编译时读的环境变量，不进 cache key，各组合 prim_func 结构又相同，于是 `dav-2201` 命中了 `dav-3510` 的产物 |

证伪手段（三重，已固化进 `probe_tilelang_arch.sh`）：
一个组合一个独立进程；每轮清磁盘缓存；**monkey-patch `bisheng.compile_ascend`
记录实际传入的 arch** —— 命中缓存时该列表为空，一眼能看出没真编。

---

## 3. DeepInfer 的 FP8 假设渗透到哪一层

```
fp8_attn   6 处（含构造期硬 assert）    mxfp4          13 处
fp8_mlp   10 处                         use_fp8_act    12 处
fp8_embed  7 处    fp8_proj  4 处       use_fp4_weight  5 处
```

另有：

- `layers/tilelang_ops/cache_common.py` —— `SUPPORTED_KV_CACHE_FORMATS` 五种格式
  全是 `e4m3` / `e2m1`，**全框架（含 CUDA 侧）没有非量化的 KV cache 选项**
- `v4_modules/index_attn.py` 四处 —— indexer 无条件调
  `deep_gemm.fp8_fp4_{,sparse_,paged_,paged_sparse_}mqa_logits`，无 BF16 对应物
- `v4_modules/containers.py:82` —— `assert get_fp8_casting_granularity() == 32`
- `layers/linear.py:164` —— `assert x[1].dtype == weight[1].dtype == PLATFORM.fp8_sf_type`
- `hugging_face/config.py:253-262` —— **在配置层**就强制 `quantization_config`
  精确等于 `{quant_method:fp8, expert_dtype:fp4, activation_scheme:dynamic,
  weight_block_size:[32,32], scale_fmt:ue8m0}`，否则 `ValueError`。
  因此后面那行 `fp8 = quant["quant_method"] == "fp8"` 是**死代码**，恒为 True

### 唯一发布的 checkpoint 本身就是 FP8+FP4

`deepseek-ai/DeepSeek-V4.1-Flash-1017` 的 `config.json`：

```json
"quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32],
                        "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
```

510.3 GB（48 个 safetensors 分片）。反量化到 BF16 约 1.0–1.6 TB，
而一个 A3 节点是 8 卡 × 2 die × 64 GB = 1024 GB。
→ A2/A3 上只能走 **INT8 W8A8 / W4A8** 重量化。

---

## 4. 引擎层本身在 A2/A3 上是能跑的

这是好消息，也是继续投入的理由。

| 层 | 状态 | 代价 |
|---|---|---|
| C++ 扩展 `deepinfer._C` | ✅ 编译通过 | 1 个 GCC-11 `std::format` 回退（patch 0001） |
| Python 完整模块图 | ✅ import 通过 | 卸 `torch_c_dlpack_ext` + 4 个 vendor 占位模块 |
| 引擎运行时 | ✅ 启动 | `LLMEngine` → tokenizer → multimodal 子进程 → `Worker` → `init_model` → `cuda_graph_batch_sizes` |
| 配置层 / 权重实体化 | ✅ 可绕 | patch 0002 |
| 模型模块构造 | ❌ 停在 `v4_modules/attn.py::_init_wv_o_proj` | 需要模型层 BF16/INT8 变体 |

→ [`tools/probe_import_chain.py`](../tools/probe_import_chain.py)、
[`tools/run_first_token.py`](../tools/run_first_token.py)

### 记账式占位模块的发现：真正挡路的比想象中少

跑到模型层为止，**被真正调用的 vendor 函数只有 2 个**（另有 13 个只是 import 期取属性）：

| 函数 | 位置 | 性质 |
|---|---|---|
| `deep_gemm.use_deterministic_algorithms` | `worker/runtime_config.py:68` | 配置开关，可直接 no-op |
| `deep_gemm.transform_sf_into_required_layout` | `v4_modules/attn.py:134` | FP8 scale-factor 布局变换，真算子 |

**引擎层几乎不碰 vendor 包，碰的全在模型层。** 这意味着要替换的面比
`grep` 出来的小得多。→ [`tools/vendor_stubs/`](../tools/vendor_stubs/)

---

## 5. `pin_host_memory()` 在 A2/A3 上是静默空操作

按 `deepinfer/shared_memory.py` 的真实路径（`multiprocessing.shared_memory` +
`torch.from_numpy` + `PLATFORM.pin_host_memory`）实测，2 GiB buffer、写满真实页、
每项 5 轮，H2D 中位带宽：

| 模式 | GB/s | 每轮全部值 |
|---|---|---|
| 【参照】`torch pin_memory=True` | **59.3** | 59.3 ×5（零抖动） |
| 【参照】`aclrtMallocHost` | **59.3** | 59.3 ×5 |
| shm 未 pin | 15.8 | 13.9 14.2 15.8 17.7 19.3 |
| shm + `ACL_HOST_REG_PINNED`（上游现状） | 17.1 | 16.9 17.0 17.1 17.4 17.5 |
| shm + `ACL_HOST_REG_MAPPED` | 24.0 | 17.2 20.6 24.0 24.1 24.1 |
| shm + `MAPPED\|PINNED` | 17.5 | 17.4 17.5 17.5 20.5 24.1 |

`aclrtHostRegisterV2` 返回 0、`is_pinned()` 翻 `True`，但带宽只有真锁页内存的四成。
且因为 `is_pinned()` 为真，`non_blocking=True` 的拷贝会被当作真异步发出。
引擎日志里 `shared_memory.py` 打印的 `is_pinned=False` 与此一致。

**换 flag 解决不了**（`MAPPED` 看似好些但呈双峰，且两者都远低于 59.3）。
方向是 shm 能否改用 `aclrtMallocHost`，涉及跨进程共享语义，不是一行改动。
本机内核是 5.10，CANN 文档称内核 ≤5.10 时必须注册 `aclrtMallocHost` 分配的锁页内存。

→ [`tools/bench_host_pin.py`](../tools/bench_host_pin.py)

⚠️ 上游 DeepSpeed PR #8532 在 910B4 + CANN 8.5.0 上的
「PINNED 8-9 GB/s vs MAPPED 23 GB/s」结论在这套栈上**没有复现**。

---

## 6. torch 版本差（上游要 2.13，A2/A3 官方配套到 2.12）

实测 `torch 2.7.1` 相对 DeepInfer 要求只缺 3 个接口：

| 缺失 | 影响 | 处理 |
|---|---|---|
| `torch._C._accelerator_setAllocatorSettings` | `worker/base.py::_init_torch` 直接 `AttributeError` | 加 `hasattr` 守卫；昇腾上同一开关已由 `PYTORCH_NPU_ALLOC_CONF` 设置 |
| `torch.compiler.config.force_cudagraph_gc` | 同上 | 加 `hasattr` 守卫 |
| `torch.float4_e2m1fn_x2` | 无 —— DeepInfer 的 Python 代码未引用 | 不处理 |

CUDA graph、symmetric memory、`mem_get_info`、`cudart`、`float8_e4m3fn` 等
经 `transfer_to_npu` 映射后**均可用**。
→ [`tools/probe_torch_api_gaps.py`](../tools/probe_torch_api_gaps.py)

---

## 7. 替换 vendor 算子的目标面（torch_npu 已有）

`torch_npu 2.7.1.post8` 提供 329 个 `npu_*` 融合算子，schema 实测存在，
几乎正面覆盖 DeepInfer 的需求：

| DeepInfer 现用 | A2/A3 替代 |
|---|---|
| `asc/norm.py`（tilelang VF） | `npu_add_rms_norm_quant` / `npu_rms_norm_dynamic_mx_quant` |
| `deep_gemm.fp8_fp4_mega_moe` | `npu_grouped_matmul_swiglu_quant` + `npu_grouped_matmul_finalize_routing` |
| `deep_gemm.fp8_fp4_*_mqa_logits`（DSA indexer） | `npu_quant_lightning_indexer`（含 `block_table` / `sparse_count` / `sparse_mode`） |
| `DeepEP-Ascend`（依赖灵衢 UB） | `npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_combine_v2` |
| `FlashMLA` ascend kernel | `npu_fusion_attention_v3` / `npu_quant_fusion_attention` |
| `asc/store_kv_cache.py` | `npu_kv_rmsnorm_rope_cache_v2` |
| `asc/kv_quant.py` | `npu_dynamic_quant` / `npu_dynamic_block_quant` |
| `asc/fused_transform.py`（swiglu+cast） | `npu_dequant_swiglu_quant` |
| 路由 topk | `npu_moe_gating_top_k`（带 `k_group` / `group_select_mode` / `routed_scaling_factor`，正对 `topk_method=noaux_tc`） |
| 其余索引 / gather 类 | FlagGems（Triton） |

**扩展点是上游自己设计好的**：`tilelang_ops/ops.py` 按
`PLATFORM.tilelang_ops_backend` 选目录，`KernelUnavailableError` 已用于处理缺算子。
加一个后端目录、实现同样的 `*_tl` 工厂接口但底下换成 torch_npu 融合算子，
不需要动引擎。

→ [`tools/probe_npu_op_surface.py`](../tools/probe_npu_op_surface.py)
