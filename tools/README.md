# 探测脚本

产出 [`docs/findings.md`](../docs/findings.md) 里每条结论的脚本。
路径全部 env 参数化，可直接在任意装了昇腾驱动的主机或容器里复跑。

| 脚本 | 回答什么 | 关键 env |
|---|---|---|
| `collect_platform_facts.sh` | 内核版本、NPU 占用、容器、CANN 版本、`ACL_HOST_REG_*` 真实枚举值 | `WORKDIR` |
| `probe_soc_version.py` | 不依赖 torch 拿 SoC 版本：ACL vs DCMI 两条路径对比 | — |
| `probe_fp8_gating.sh` | FP8 在 2201 上的门控；9.2 与旧版 CANN 的对照 | `CANN_PREFIX`、`OLD_CANN` |
| `probe_bisheng_arch.sh` | bisheng 实际接受哪些 `--npu-arch`（两套命名的坑） | `ASCEND_HOME_PATH` |
| `probe_tilelang_arch.sh` | tilelang 能否编到 `dav-2201`。**一组合一进程 + 拦截编译调用**，否则会被缓存骗 | `CANN_PREFIX`、`TL_VENV` |
| `probe_torch_api_gaps.py` | torch 2.7.1 相对上游要求的 2.13 缺哪些接口 | — |
| `probe_npu_op_surface.py` | torch_npu 融合算子的真实 schema（替换 vendor 算子的目标面） | — |
| `probe_import_chain.py` | DeepInfer 的模块图逐层 import 到哪一步 | `DEEPINFER_SRC` |
| `make_tiny_model.py` | 造极小 V4.1 配置（配 `load_format="dummy"`，不需要权重） | — |
| `run_first_token.py` | bring-up 驱动：起引擎、记录被真正调用的 vendor 算子 | `DEEPINFER_SRC`、`TINY_MODEL`、`DI_STUB_LOG`、`DI_STUB_NOOP` |
| `bench_host_pin.py` | `aclrtHostRegisterV2` 各 flag 的 H2D 带宽，对照真锁页内存 | — |
| `vendor_stubs/` | 记账式 vendor 占位模块（见下） | `DI_STUB_LOG`、`DI_STUB_NOOP` |

## 记账式 vendor 占位模块

DeepInfer 在模块顶层硬 import 了 `deep_ep` / `deep_gemm` / `flash_mla` / `deep_select`，
它们在昇腾上由 DeepEP-Ascend / DeepGEMM-Ascend / FlashMLA 提供，而那几个库只支持 950。
A2/A3 上装不出来，于是**引擎连 import 都过不去**。

占位模块可以 import、但被真正调用时报错并**记账**：

```
PYTHONPATH=tools/vendor_stubs DI_STUB_LOG=/tmp/vendor.txt python3 ...
```

日志格式 `<类型>\t<qualname>\t<调用方>`，类型为：

- `ATTR` —— 只是 import 期取属性，没真用
- `CALL` —— **被真正调用，需要一个 A2/A3 实现**
- `NOOP` —— 在 `DI_STUB_NOOP` 白名单里，返回 None 放行（只用于配置/开关类接口）

这份日志就是待实现清单，比 `grep` 准得多：跑到模型层为止，
**被真正调用的只有 2 个**，而 `grep` 能数出几十处引用。

`DI_STUB_NOOP` 的名单本身也是产出 —— 它记录了「为了推进下去必须假装成功的东西」。
