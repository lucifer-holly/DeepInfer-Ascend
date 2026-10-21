"""910C bring-up 用的 vendor 占位模块。

DeepInfer 在模块顶层硬 import 了四个 vendor 包（`deep_ep` / `deep_gemm` /
`flash_mla` / `deep_select`），它们在昇腾上由 DeepEP-Ascend / DeepGEMM-Ascend /
FlashMLA 提供，而那几个库都只支持 950。910C 上这些包装不出来，于是**引擎连
import 都过不去**。

这里提供可以 import、但一旦真正被调用就报错并**记账**的占位实现。
目的不是让它跑对，而是回答一个问题：

    「跑出第一个 token，到底需要哪些 vendor 算子？」

跑完看 `DI_STUB_LOG`（默认 /tmp/di_vendor_calls.txt），里面就是精确的待实现清单。
"""

from __future__ import annotations

import os
import threading
import traceback

_LOG_PATH = os.environ.get("DI_STUB_LOG", "/tmp/di_vendor_calls.txt")
# 显式白名单：这些 qualname 被调用时返回 None 而不是报错，并记为 NOOP。
# 只用于**配置/开关类**接口（它们不参与计算），好让 bring-up 推进下去。
# 这份名单本身是产出：它就是「为了跑通第一个 token 必须假装成功的东西」。
_NOOP = {name for name in os.environ.get("DI_STUB_NOOP", "").split(",") if name}
_lock = threading.Lock()
_seen: set[str] = set()


def _record(qualname: str, kind: str) -> None:
    with _lock:
        if qualname in _seen:
            return
        _seen.add(qualname)
        # 记下调用方，方便定位是哪条路径需要它
        caller = ""
        for frame in reversed(traceback.extract_stack()[:-2]):
            if "_di_vendor_stub" not in frame.filename:
                caller = f"{frame.filename}:{frame.lineno} in {frame.name}"
                break
        try:
            with open(_LOG_PATH, "a") as f:
                f.write(f"{kind}\t{qualname}\t{caller}\n")
        except OSError:
            pass


class StubError(NotImplementedError):
    """被占位的 vendor 算子被真正调用了 —— 说明这条路径需要一个 910C 实现。"""


def _make_stub(module: str, name: str) -> type:
    qualname = f"{module}.{name}"
    _record(qualname, "ATTR")

    def _boom(*args: object, **kwargs: object):
        if qualname in _NOOP:
            _record(qualname, "NOOP")
            return None
        _record(qualname, "CALL")
        raise StubError(
            f"{qualname} 在 910C 上没有实现。\n"
            f"  上游由 {module} 提供（昇腾侧是 DeepEP-Ascend / DeepGEMM-Ascend / FlashMLA，均只支持 950）。\n"
            f"  910C 需要换成 torch_npu 融合算子或 FlagGems/Triton 实现。\n"
            f"  已记入 {_LOG_PATH}"
        )

    # 做成 class：既能当函数调用（走 __init__），也能用在类型标注和 isinstance 里
    return type(name, (), {"__init__": _boom, "__call__": _boom, "_di_stub": qualname})


def module_getattr(module: str):
    """给占位模块用的 PEP 562 `__getattr__` 工厂。"""
    cache: dict[str, type] = {}

    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name not in cache:
            cache[name] = _make_stub(module, name)
        return cache[name]

    return __getattr__
