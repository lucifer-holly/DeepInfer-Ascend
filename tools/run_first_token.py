#!/usr/bin/env python3
"""910C 上跑出 DeepInfer 的第一个 token —— 极小配置、随机权重、最慢路径。

目的：把「引擎能不能在 910C 上转」与「510 GB 权重 + 性能」解耦。
  - 权重：`load_format="dummy"`（llm_engine.py:241 的 skip_load），不读 safetensors
  - 规模：6 层 / 8 专家 / 关掉 MTP / 关掉 engram / 关掉 candidate topk / EP=CP=DP=1
  - 算子：vendor 包全是记账式占位，被调到就报错并记进 DI_STUB_LOG

每一步单独 try，失败就打完整 traceback 并 dump 占位调用日志 ——
**失败在哪一步、需要哪个 vendor 算子，本身就是这次要的产出。**
"""

import os
import sys
import traceback

MODEL = os.environ.get("TINY_MODEL", "/work/tinymodel")
STUB_LOG = os.environ.get("DI_STUB_LOG", "/tmp/di_vendor_calls.txt")

# ── 环境必须在 import torch / deepinfer 之前设好 ──
os.environ["DEEPINFER_ACCELERATOR"] = "ascend"
# PlatformNPU.__init__ 要求两者相等
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
# 关掉投机解码与 MTP varlen
os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "0"
os.environ["DEEPINFER_ENABLE_MTP_VARLEN"] = "0"
os.environ["DEEPINFER_SAMPLER_FORCE_TOPK"] = "0"
os.environ.setdefault("DI_STUB_LOG", STUB_LOG)
sys.path.insert(0, os.environ.get("DEEPINFER_SRC", "/work/src"))

if os.path.exists(STUB_LOG):
    os.remove(STUB_LOG)


def dump_stub_log(header: str) -> None:
    print(f"\n{'=' * 72}\n{header}\n{'=' * 72}")
    if not os.path.exists(STUB_LOG):
        print("  （没有任何 vendor 占位被触碰）")
        return
    rows = [line.rstrip("\n").split("\t") for line in open(STUB_LOG) if line.strip()]
    calls = [r for r in rows if r[0] == "CALL"]
    noops = [r for r in rows if r[0] == "NOOP"]
    attrs = [r for r in rows if r[0] == "ATTR"]
    print(f"  被真正调用 (CALL) {len(calls)} 个：")
    for r in calls:
        print(f"    🔴 {r[1]}")
        if len(r) > 2 and r[2]:
            print(f"       ← {r[2]}")
    print(f"  被迫假装成功 (NOOP) {len(noops)} 个：")
    for r in noops:
        print(f"    🟡 {r[1]}")
        if len(r) > 2 and r[2]:
            print(f"       ← {r[2]}")
    print(f"  仅被取属性 (ATTR) {len(attrs)} 个：")
    for r in attrs:
        print(f"       {r[1]}")


step = "import torch"
try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    step = "import deepinfer.entrypoints.llm"
    from deepinfer.entrypoints.llm import LLM
    from deepinfer.sampling_params import SamplingParams

    print(f"✅ import 成功  torch={torch.__version__} torch_npu={torch_npu.__version__}")
    print(f"   soc={torch.npu.get_device_name(0)}  device_count={torch.npu.device_count()}")

    step = "构造 LLM（load_format=dummy）"
    print(f"\n── {step} ──", flush=True)
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        trust_remote_code=True,
        load_format="dummy",
        expert_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=1,
        engine_rank=0,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
        chunked_prefill=True,
        decode_sequence=False,
        disable_log_stats=False,
    )
    print("✅ LLM 构造成功", flush=True)

    step = "generate 第一个 token"
    print(f"\n── {step} ──", flush=True)
    outputs = llm.generate(
        prompts=["Hello"],
        sampling_params=[SamplingParams(max_tokens=4, temperature=0.0)],
        use_tqdm=False,
    )
    print("✅ generate 返回", flush=True)
    for i, out in enumerate(outputs):
        if isinstance(out, str):
            print(f"  sample {i}: {out!r}")
        else:
            ids = [x.token_id for x in out]
            print(f"  sample {i}: token_ids={ids}  tokens={len(ids)}")
            print(f"             decoded={llm.llm_engine.tokenizer.decode(ids)!r}")
    print("\n🎉 第一个 token 跑出来了")
    llm.close()
    dump_stub_log("本次用到的 vendor 算子")

except BaseException:
    print(f"\n❌ 失败于：{step}\n")
    traceback.print_exc()
    dump_stub_log(f"失败于「{step}」时已触碰的 vendor 算子")
    sys.exit(1)
