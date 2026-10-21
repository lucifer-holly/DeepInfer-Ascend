#!/usr/bin/env python3
"""按 DeepInfer 的真实代码路径测 pin 是否有效。

复刻 deepinfer/shared_memory.py:116-120：
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=N)
    buffer = torch.from_numpy(np.frombuffer(shm.buf, dtype=np.uint8))
    PLATFORM.pin_host_memory(buffer)      # → aclrtHostRegisterV2(ptr, n, ACL_HOST_REG_PINNED)
    log(f"is_shared={buffer.is_shared()}, is_pinned={buffer.is_pinned()}")

修掉 v2 的两个方法学漏洞：
  1. v2 用匿名 mmap，零页可能一直命中同一个物理页 → 这里先写满，确保每页都是真实且互异的
  2. v2 测的是匿名 mmap，不是 POSIX shm → 这里用 DeepInfer 实际用的 SharedMemory

参照系（上界）：aclrtMallocHost 与 torch pin_memory=True —— 真锁页内存能跑多快。
"""

from __future__ import annotations

import argparse
import ctypes
import time
from multiprocessing import shared_memory

ACL_SUCCESS = 0
ERR_ALREADY_REGISTERED = 507910
ERR_NOT_REGISTERED = 507911
# 核实自 175 宿主机 /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl_rt.h
FLAGS = {"PINNED (DeepInfer 现状)": 0x10000000, "MAPPED": 0x2, "MAPPED|PINNED": 0x2 | 0x10000000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(torch.npu.current_device())
    print(f"[env] torch={torch.__version__} torch_npu={torch_npu.__version__} soc={torch.npu.get_device_name(0)}")

    lib = ctypes.CDLL("libascendcl.so")
    for fn, argtypes, restype in (
        ("aclrtHostRegisterV2", (ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32), ctypes.c_int),
        ("aclrtHostUnregister", (ctypes.c_void_p,), ctypes.c_int),
        ("aclrtMallocHost", (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t), ctypes.c_int),
        ("aclrtFreeHost", (ctypes.c_void_p,), ctypes.c_int),
    ):
        f = getattr(lib, fn)
        f.argtypes, f.restype = argtypes, restype

    nbytes = args.mib * 1024 * 1024
    devbuf = torch.empty(nbytes, dtype=torch.uint8, device="npu")

    def bench(host, direction: str) -> list[float]:
        def once() -> float:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                if direction == "H2D":
                    devbuf.copy_(host, non_blocking=True)
                else:
                    host.copy_(devbuf, non_blocking=True)
            torch.npu.synchronize()
            return nbytes * args.iters / 1e9 / (time.perf_counter() - t0)

        once()
        return [once() for _ in range(args.repeats)]

    def report(name: str, host, note: str = "") -> None:
        for d in ("H2D", "D2H"):
            v = sorted(bench(host, d))
            print(f"{name + ' ' + d:<38} {v[len(v)//2]:>7.1f}  [{'  '.join(f'{x:.1f}' for x in v)}]  {note if d=='H2D' else ''}")

    print(f"\n[plan] buffer={args.mib} MiB（已写满，真实页）, 每轮 {args.iters} 次, 每项 {args.repeats} 轮")
    print(f"{'模式/方向':<38} {'中位GB/s':>7}  每轮全部值")
    print("-" * 92)

    # ── 上界参照 ──
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    t.random_(0, 255)
    report("【参照】torch pin_memory=True", t, f"is_pinned={t.is_pinned()}")
    del t

    p = ctypes.c_void_p()
    if lib.aclrtMallocHost(ctypes.byref(p), nbytes) == ACL_SUCCESS and p.value:
        arr = (ctypes.c_uint8 * nbytes).from_address(p.value)
        t = torch.frombuffer(arr, dtype=torch.uint8)
        t.random_(0, 255)
        report("【参照】aclrtMallocHost", t, f"is_pinned={t.is_pinned()}")
        del t, arr
        lib.aclrtFreeHost(p)
    else:
        print("【参照】aclrtMallocHost  ❌ 分配失败")

    # ── 被测：DeepInfer 的真实路径 ──
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        buffer = torch.from_numpy(np.frombuffer(shm.buf, dtype=np.uint8))
        buffer.random_(0, 255)  # 写满：每页都是真实且互异的，排除零页假象
        ptr = buffer.data_ptr()
        print(f"\n   shm 地址 0x{ptr:x}  4K对齐={ptr % 4096 == 0}  is_shared={buffer.is_shared()}")
        report("shm 未 pin", buffer, f"is_pinned={buffer.is_pinned()}")

        for name, flag in FLAGS.items():
            err = lib.aclrtHostRegisterV2(ctypes.c_void_p(ptr), nbytes, flag)
            if err not in (ACL_SUCCESS, ERR_ALREADY_REGISTERED):
                # 内核 ≤5.10 上注册非锁页内存预期 107017 —— 失败本身就是结论
                print(f"{'shm + ' + name:<38} ❌ aclError={err}")
                continue
            report(f"shm + {name}", buffer, f"reg={err} is_pinned={buffer.is_pinned()}")
            e = lib.aclrtHostUnregister(ctypes.c_void_p(ptr))
            if e not in (ACL_SUCCESS, ERR_NOT_REGISTERED):
                print(f"{'':<38} ⚠️ unregister aclError={e}")
        del buffer
    finally:
        shm.close()
        shm.unlink()

    print("-" * 92)
    print("判读：shm + 任一 flag 若都 ≈ shm 未 pin，且远低于两个参照 →")
    print("      aclrtHostRegisterV2 在这套栈上拿不到 DMA 快路径，DeepInfer 的 pin_host_memory() 是静默空操作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
