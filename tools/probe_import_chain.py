#!/usr/bin/env python3
"""逐层探 DeepInfer 在 910C 上能 import 到哪一步。

一层一层来，每层单独报结果 —— 一次性 import 顶层只会看到第一个错误，
看不出后面还有几道坎。
"""

import os
import sys
import traceback

os.environ.setdefault("DEEPINFER_ACCELERATOR", "ascend")
sys.path.insert(0, os.environ.get("DEEPINFER_SRC", "/work/src"))

STEPS = [
    ("硬件探测", "from deep_cluster.hardware import accelerator_kind; r = accelerator_kind()"),
    ("设备数", "from deep_cluster.hardware import device as D; r = (D.physical_device_count(), D.visible_device_count())"),
    ("NUMA 映射", "from deep_cluster.hardware import device as D; r = [D.device_numa_node(i) for i in range(2)]"),
    ("TFLOPS 常量", "from deep_cluster.hardware import device as D; r = D.get_bf16_tflops_limit()"),
    ("deepinfer 顶层", "import deepinfer; r = deepinfer.__version__"),
    ("C++ 扩展", "from deepinfer import _C; r = [x for x in dir(_C) if not x.startswith('_')]"),
    ("engine 子模块", "from deepinfer._C import engine; r = len([x for x in dir(engine) if not x.startswith('_')])"),
    ("PLATFORM 构造", "from deepinfer.platform import get_platform; p = get_platform(); r = (p.enum, p.tilelang_ops_backend, p.distributed_backend, p.fp8_sf_type)"),
    ("npu_ops", "import deepinfer.npu_ops; r = 'ok'"),
    ("tilelang_ops.ops", "from deepinfer.model_executor.layers.tilelang_ops import ops; r = len([x for x in dir(ops) if x.endswith('_tl')])"),
    ("KV cache 格式表", "from deepinfer.model_executor.layers.tilelang_ops import cache_common as C; r = [f.note for f in C.SUPPORTED_KV_CACHE_FORMATS]"),
    ("config", "from deepinfer.config import *; r = 'ok'"),
    ("engine/arg_utils", "from deepinfer.engine.arg_utils import EngineArgs; r = 'ok'"),
    ("LLM 入口", "from deepinfer.entrypoints.llm import LLM; r = 'ok'"),
    ("模型定义", "from deepinfer.model_executor.models import deepseek; r = 'ok'"),
    ("worker", "from deepinfer.worker.worker import Worker; r = 'ok'"),
]

width = max(len(n) for n, _ in STEPS)
ok = fail = 0
for name, code in STEPS:
    ns = {}
    try:
        exec(code, ns)
        val = repr(ns.get("r"))
        print(f"✅ {name:<{width}}  {val[:110]}")
        ok += 1
    except BaseException as e:
        tb = traceback.format_exc()
        # 只留最后一个非本脚本的帧 + 异常本身
        frames = [l.strip() for l in tb.splitlines() if l.strip().startswith("File ") and "20-import-probe" not in l]
        where = frames[-1][:110] if frames else ""
        print(f"❌ {name:<{width}}  {type(e).__name__}: {str(e)[:100]}")
        if where:
            print(f"   {'':<{width}}  ↳ {where}")
        fail += 1

print(f"\n通过 {ok} / 失败 {fail}")
