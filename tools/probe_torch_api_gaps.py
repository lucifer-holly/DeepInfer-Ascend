#!/usr/bin/env python3
"""一次探清 torch 2.7.1 相对 DeepInfer 要求的 2.13 缺哪些 API。

每轮 e2e 跑要几分钟，撞一个修一个太慢。把 DeepInfer 用到的、看起来较新的
torch 接口一次性探完，然后一把加守卫。
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401,E402

print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}\n")

CHECKS: list[tuple[str, str]] = [
    # worker/base.py::_init_torch
    ("torch._C._accelerator_setAllocatorSettings", "torch._C"),
    ("torch.compiler.config.force_cudagraph_gc", "torch.compiler.config"),
    ("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction", "torch.backends.cuda.matmul"),
    ("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction", "torch.backends.cuda.matmul"),
    # cuda graph
    ("torch.cuda.CUDAGraph", "torch.cuda"),
    ("torch.cuda.graph", "torch.cuda"),
    ("torch.cuda.graph_pool_handle", "torch.cuda"),
    # 内存/分配器
    ("torch.cuda.memory_stats", "torch.cuda"),
    ("torch.cuda.mem_get_info", "torch.cuda"),
    ("torch.cuda.caching_allocator_alloc", "torch.cuda"),
    ("torch.cuda.cudart", "torch.cuda"),
    # 对称内存（eplb shuffle 用）
    ("torch.distributed._symmetric_memory", "torch.distributed"),
    # 其他
    ("torch.cuda.Stream", "torch.cuda"),
    ("torch.cuda.Event", "torch.cuda"),
    ("torch.cuda.set_device", "torch.cuda"),
    ("torch.cuda.current_stream", "torch.cuda"),
    ("torch.cuda.synchronize", "torch.cuda"),
    ("torch.cuda.get_device_properties", "torch.cuda"),
    ("torch.float8_e4m3fn", "torch"),
    ("torch.float4_e2m1fn_x2", "torch"),
    ("torch.compiler.config", "torch.compiler"),
]


def resolve(path: str):
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


missing = []
for path, _ in CHECKS:
    try:
        resolve(path)
        print(f"  ✅ {path}")
    except AttributeError as e:
        print(f"  ❌ {path}   ({e})")
        missing.append(path)

print(f"\n缺失 {len(missing)} 个：")
for m in missing:
    print(f"  {m}")

print("\n--- get_device_properties 返回了什么（userbuffer.py 读 multi_processor_count）---")
try:
    p = torch.cuda.get_device_properties(0)
    print("  ", {k: getattr(p, k) for k in dir(p) if not k.startswith("_")})
except Exception as e:
    print("  取不到:", e)
