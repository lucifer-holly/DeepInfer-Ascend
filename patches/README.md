# Bring-up 补丁序列

针对 [`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer) 的补丁，
让引擎能在**昇腾 A2/A3（`NpuArch=2201`）** 上构建并启动。

基线提交记录在 [`BASE_COMMIT`](BASE_COMMIT)。

```bash
git clone https://github.com/deepseek-ai/DeepInfer && cd DeepInfer
git checkout "$(cat ../patches/BASE_COMMIT)"
git am ../patches/*.patch
```

## 🔴 这些补丁不适合提给上游

每一条都是为了绕开上游的**明示前提**，不是修复上游的缺陷：

- 上游 README 要求 **GCC ≥ 13**，补丁 0001 是为了在只有 GCC 11 的镜像里编译
- 上游**设计上只支持一个量化 checkpoint**（MXFP8 注意力 + MXFP4 专家），
  补丁 0002 放开了这个前提

向上游提的改动另走单独分支，目前有一个：
[deepseek-ai/DeepInfer#170](https://github.com/deepseek-ai/DeepInfer/pull/170)
（BF16 cube 峰值改为从 CANN 的 platform_config 读取，而非硬编码 432）——
**该改动不在本目录的补丁序列里**。

## 0001 — `std::format` 回退，让 csrc 在 GCC 11 下能编

| | |
|---|---|
| 触发的失败 | `csrc/utils/log.hpp: fatal error: format: No such file or directory` |
| 为什么 | `<format>` 从 GCC 13 才进 libstdc++。昇腾 A3 的官方训练镜像（Ubuntu 22.04）只带 GCC 11，容器内无网络、apt 源也没有 g++-13 |
| 范围 | csrc 里 5 处 `std::format`，全部是裸 `{}` 占位符，无 `{:spec}` 格式说明符 |

⚠️ **一个非显然的坑**：`format_string<Args...>` 必须经 `std::type_identity_t`
变成**非推导上下文**（libstdc++ 对 `std::format_string` 就是这么做的）。
少了这一层，编译器会试图从格式串字面量 `const char[N]` 去推 `Args`，
于是 `log::write(...)` 的**每个调用点**都报
`no matching function for call to write(...)`。

## 0002 — 让引擎推进到 FP8 边界

四处改动，每一处都对应一个在 `Ascend910_9382` 上实测到的失败：

| 文件 | 触发的失败 | 改法 |
|---|---|---|
| `worker/base.py` | `AttributeError: module 'torch._C' has no attribute '_accelerator_setAllocatorSettings'` | 加 `hasattr` 守卫。该 API 是 torch 2.13 引入的；昇腾上同一个开关已由 `PlatformNPU.__init__` 里的 `PYTORCH_NPU_ALLOC_CONF` 设置。`torch.compiler.config.force_cudagraph_gc` 同样处理 |
| `hugging_face/config.py` | 任何非 MXFP8+MXFP4 的 checkpoint 直接 `ValueError` | 仅在 checkpoint 确为量化时才强校验那五个布局键 |
| `models/deepseek.py` | `RuntimeError: zero_: call aclnnInplaceZero failed, error code 161002` —— `Tensor self not implemented for DT_FLOAT8_E4M3FN` | 非 GPU 平台上把 FP8 参数按 BF16 实体化 |
| `v4_modules/mla_attn.py` | `AssertionError` at `assert cfg.fp8_attn` | 去掉该断言。`attn.py:105` 本来就对这个 flag 分支处理（`convert_rmsnorm(..., use_fp8=...)`），所以模块至少是可构造的 |

### 打完补丁后停在哪

```
v4_modules/attn.py::_init_wv_o_proj
    wv_sf = attn_module.Wv.weight.sf.view(...)
    wv_sf = deep_gemm.transform_sf_into_required_layout(wv_sf, ..., is_sfa=False)
```

无条件读 FP8 scale factor，**没有任何非 FP8 分支**。越过它需要实现
V4.1 模型层的 BF16/INT8 变体，属于重写而非 bring-up，因此刻意停在这里。

硬推（把 `transform_sf_into_required_layout` 和
`fused_norm_rope_attn_rope_cast.permute_wv_proj` 都 stub 成恒等变换）能跑出 token，
但那个 token 不能证明任何事 —— 不做。
