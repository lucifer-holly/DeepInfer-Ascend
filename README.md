# DeepInfer-Ascend

> 让 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 支持**昇腾 950 以外的昇腾加速卡**。

上游 DeepInfer 的昇腾后端是围绕**昇腾 950**（`NpuArch=3510`）构建的 ——
kernel DSL、GEMM 库、通信库、注意力库、数值格式、互联协议，每一层都假定 950 的能力集。
README 写明的硬件支持是 "NVIDIA SM100 GPUs / Huawei Ascend 950 NPUs"，
依赖是 CANN 9.2 + `torch-npu 2.13`。

而昇腾在用的卡远不止 950。本仓库把「昇腾」这一层从「950」里拆出来，
让同一套引擎能落到 **A2（910B 系列）/ A3（910C、`Ascend910_93`）** 及后续型号上。

> **状态：bring-up 与可行性验证完成，尚不能用于推理。**
> 当前第一目标是 **910C（A3 / `NpuArch=2201`）**，实测平台 `Ascend910_9382`。

## 结论速览

**引擎层能跑，模型层不能。** 这是本轮最重要的结果：

| 层 | A2/A3 上 | 障碍 |
|---|---|---|
| C++ 扩展 `deepinfer._C` | ✅ 编译通过 | 镜像只有 GCC 11，`<format>` 要 GCC 13 |
| Python 完整模块图 | ✅ import 通过 | 四个 vendor 包顶层硬 import |
| 引擎运行时（调度器 / worker / tokenizer / 多模态子进程） | ✅ 启动 | torch 2.13 专有的两个 API |
| 模型模块构造 | ❌ | **FP8 —— 硬件层面就没有** |
| kernel 层（30 个 tilelang 算子） | ❌ | **VF 编程模型 arch 2201 不支持** |

两个硬障碍都有来自华为自己的证据，不是推测：

- **FP8 缺失**有四重独立证据（CANN 头文件门控、mmad dtype 表、加速形状表、
  以及运行时连 `torch.zeros` 一个 FP8 张量都报 `aclnnInplaceZero` 失败）
- **tilelang** 在 CANN 9.2 下对 `dav-3510` 两种模式都编得过、对 `dav-2201` 两种都编不过，
  诊断原文是 `__simd_vf__/__simt_vf__ is not supported on current architecture`

详见 **[docs/findings.md](docs/findings.md)**（每条都附可复核的手段）。

## 这不是「加个新硬件」的问题

`get_bf16_tflops_limit()` 对所有昇腾硬编码 `432`，而 432 只是 950 的 **32 核 @ 1650 MHz**
那一支；同代系内 28 核是 378、36 核是 530。**上游这个常量在 950 上就已经不准。**
已向上游提交修复：[deepseek-ai/DeepInfer#170](https://github.com/deepseek-ai/DeepInfer/pull/170)。

## 设计原则

1. **不新增加速器类型。** 910B/910C 就是 `ascend`，不该变成第四种 `AcceleratorKind`。
   要加的是 **ascend 内部的 soc 维度** —— 上游 `PlatformEnum` 目前只有 `GPU`/`NPU`，
   没有芯片粒度，这是扩展点。
2. **型号参数从 CANN 读，不写死。** 核数、UB 容量、峰值算力都在
   `$ASCEND_HOME_PATH/*-linux/data/platform_config/<SoC>.ini` 里，运行时可查。
3. **按特性判断而非按型号判断。** 「有没有原生 FP8」比「是不是 910C」稳定，
   新卡出来不用再改一遍。
4. **通用修正回上游，型号专有实现留本仓库。**

## 仓库结构

```
patches/      针对上游的 bring-up 补丁序列（含 BASE_COMMIT 与「为什么不提上游」的说明）
docs/
  findings.md       实测结论全集，每条附复核手段
  platform-910c.md  Ascend910_9382 平台事实、两套 arch 命名的坑、规格对照
tools/        产出上述结论的探测脚本（路径均 env 参数化，可直接复跑）
  vendor_stubs/     记账式 vendor 占位模块
reference/    CANN 对本机的权威描述原文（platform_config INI）
```

## 快速开始

### 复核平台事实

```bash
bash tools/collect_platform_facts.sh      # 内核 / NPU 占用 / CANN / ACL_HOST_REG_* 枚举值
python3 tools/probe_soc_version.py        # 不依赖 torch 拿 SoC 版本的两条路径对比
bash tools/probe_fp8_gating.sh            # FP8 门控与 2201 的支持情况
bash tools/probe_bisheng_arch.sh          # bisheng 到底接受哪些 --npu-arch
```

### 复核 tilelang 结论

```bash
CANN_PREFIX=/opt/cann-9.2 bash tools/probe_tilelang_arch.sh
```

一个组合一个独立进程，并拦截 `bisheng.compile_ascend` 验证实际传入的 arch ——
**不这么做会被 tilelang 的进程内存缓存骗过去**（我们被骗过一次，见 findings 第 2 节）。

### 应用补丁并跑 bring-up

```bash
git clone https://github.com/deepseek-ai/DeepInfer && cd DeepInfer
git checkout "$(cat ../patches/BASE_COMMIT)"
git am ../patches/*.patch
DEEPINFER_ACCELERATOR=ascend bash ./build.sh        # 或 scripts/build.sh build

# 极小配置 + 随机权重（load_format=dummy），不需要 510 GB 的真权重
python3 ../tools/make_tiny_model.py --out /tmp/tinymodel --layers 6 --experts 8

# vendor 包在 A2/A3 上装不出来，用记账式占位让 import 通过
DEEPINFER_SRC=$PWD PYTHONPATH=../tools/vendor_stubs python3 ../tools/probe_import_chain.py
DEEPINFER_SRC=$PWD TINY_MODEL=/tmp/tinymodel PYTHONPATH=../tools/vendor_stubs \
  DI_STUB_NOOP=deep_gemm.use_deterministic_algorithms \
  python3 ../tools/run_first_token.py
```

占位模块会把**被真正调用**的 vendor 函数记进 `DI_STUB_LOG`（默认 `/tmp/di_vendor_calls.txt`）。
跑到模型层为止只有 2 个被真正调用 —— 这份日志就是待实现清单。

## 下一步

实现 V4.1 模型层的 BF16/INT8 变体。`torch_npu 2.7.1.post8` 的 329 个 `npu_*` 融合算子
几乎正面覆盖所需（含 `npu_quant_lightning_indexer`、`npu_moe_distribute_dispatch_v2`、
`npu_grouped_matmul_swiglu_quant`），映射表见 [findings 第 7 节](docs/findings.md)。
扩展点是上游自己设计好的：`tilelang_ops/ops.py` 按 `PLATFORM.tilelang_ops_backend`
选后端目录。

## 关系

- **上游**：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- **通用修正**走 fork [lucifer-holly/DeepInfer](https://github.com/lucifer-holly/DeepInfer)
  的分支向上游提 PR；**型号专有实现**留本仓库
- 上游另有一份方向相同的 draft RFC：deepseek-ai/DeepInfer#46（纯文档、无实现）

## License

随上游，Apache License 2.0。
