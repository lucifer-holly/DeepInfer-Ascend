#!/usr/bin/env python3
"""不依赖 torch 拿 soc 版本号 —— 两条 ctypes 路径的实测对比。

为什么要 torch-free：`deep_cluster/hardware/*` 全程只用 procfs/sysfs/ctypes
（见 nvidia.py 的 docstring "NVIDIA discovery using procfs, sysfs, and device nodes only"），
`get_bf16_tflops_limit()` 要符合这个约定。

上游既有写法（nvidia.py:_nvml / compute_capability）：
    ctypes 加载厂商**管理库** → 查询 → 失败返回 None，不建 device context。
昇腾的对应物：
    DCMI（libdcmi.so）—— 管理库，对标 NVML
    ACL （libascendcl.so）—— runtime 库，platform.py 已经在用
"""

from __future__ import annotations

import ctypes
import ctypes.util

print("=" * 70)
print("路径 A：DCMI（对标 NVML 的管理库）")
print("=" * 70)


def try_dcmi() -> str | None:
    for cand in filter(None, (ctypes.util.find_library("dcmi"), "libdcmi.so", "/usr/local/dcmi/libdcmi.so")):
        try:
            lib = ctypes.CDLL(cand)
            print(f"  ✅ 载入 {cand}")
            break
        except OSError as e:
            print(f"  ✗ {cand}: {e}")
    else:
        return None

    syms = [s for s in ("dcmi_init", "dcmi_get_card_num_list", "dcmi_get_device_chip_info",
                        "dcmi_get_device_chip_info_v2", "dcmi_get_device_id_in_card") if hasattr(lib, s)]
    print(f"  可用符号: {syms}")
    if not hasattr(lib, "dcmi_init") or not hasattr(lib, "dcmi_get_device_chip_info"):
        return None

    lib.dcmi_init.restype = ctypes.c_int
    rc = lib.dcmi_init()
    print(f"  dcmi_init() -> {rc}")
    if rc != 0:
        return None

    # struct dcmi_chip_info { char chip_type[32]; char chip_name[32]; char chip_ver[32]; ... }
    class ChipInfo(ctypes.Structure):
        _fields_ = [("chip_type", ctypes.c_char * 32), ("chip_name", ctypes.c_char * 32), ("chip_ver", ctypes.c_char * 32)]

    lib.dcmi_get_device_chip_info.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ChipInfo))
    lib.dcmi_get_device_chip_info.restype = ctypes.c_int
    info = ChipInfo()
    rc = lib.dcmi_get_device_chip_info(0, 0, ctypes.byref(info))
    print(f"  dcmi_get_device_chip_info(0,0) -> {rc}")
    if rc != 0:
        return None
    t = info.chip_type.decode(errors="replace").strip()
    n = info.chip_name.decode(errors="replace").strip()
    v = info.chip_ver.decode(errors="replace").strip()
    print(f"    chip_type={t!r}  chip_name={n!r}  chip_ver={v!r}")
    return f"{t}{n}"


a = try_dcmi()
print(f"  → DCMI 结果: {a!r}")

print()
print("=" * 70)
print("路径 B：ACL aclrtGetSocName（platform.py 已经在用 libascendcl）")
print("=" * 70)


def try_acl(with_init: bool) -> str | None:
    try:
        lib = ctypes.CDLL("libascendcl.so")
    except OSError as e:
        print(f"  ✗ libascendcl.so: {e}")
        return None
    if not hasattr(lib, "aclrtGetSocName"):
        print("  ✗ 无 aclrtGetSocName 符号")
        return None
    if with_init:
        if hasattr(lib, "aclInit"):
            lib.aclInit.argtypes = (ctypes.c_char_p,)
            lib.aclInit.restype = ctypes.c_int
            rc = lib.aclInit(None)
            print(f"  aclInit(NULL) -> {rc}")
    lib.aclrtGetSocName.restype = ctypes.c_char_p
    p = lib.aclrtGetSocName()
    s = p.decode(errors="replace") if p else None
    print(f"  aclrtGetSocName() -> {s!r}   （aclInit={'有' if with_init else '无'}）")
    return s


print("  -- 不调 aclInit --")
b1 = try_acl(with_init=False)
print("  -- 调 aclInit --")
b2 = try_acl(with_init=True)

print()
print("=" * 70)
print("结论")
print("=" * 70)
print(f"  DCMI                      : {a!r}")
print(f"  aclrtGetSocName (无 init) : {b1!r}")
print(f"  aclrtGetSocName (有 init) : {b2!r}")
print()
print("  需要的是能索引 platform_config/<SoC_version>.ini 的那个字符串，")
print("  本机应为 'Ascend910_9382'。")
