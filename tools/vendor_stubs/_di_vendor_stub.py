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
_seen: set[tuple[str, str]] = set()


def _record(qualname: str, kind: str) -> None:
    # 按 (kind, qualname) 去重，不能只按 qualname：取属性一定先于调用发生，
    # 只按名字去重会让每一条 CALL / IMPL / NOOP 都被先到的 ATTR 吞掉，
    # 日志里 CALL 永远是 0 —— 这个 bug 2026-11-10 才发现。
    with _lock:
        if (kind, qualname) in _seen:
            return
        _seen.add((kind, qualname))
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


# ── 第三类：真·最小实现 ───────────────────────────────────────────────────
# 有些 vendor 接口根本不是算子，只是**全局簿记**（比如分组 GEMM 连续布局的对齐粒度）。
# 它们不参与计算，照搬语义即可，所以这里给的是真实现而不是假装。
# 记为 IMPL，与「需要真实现的算子」(CALL) 和「被迫假装成功」(NOOP) 区分开。
#
# 对齐粒度只决定 token 往 per-expert 连续块里怎么补齐，是个粒度不是数值参数 ——
# 只要生产方与消费方取到同一个值，结果就一致。128 是 bring-up 的占位默认值，
# 等 910C 真正选定分组 GEMM 实现（如 npu_grouped_matmul）后应按其要求改。
_MK_ALIGNMENT = [int(os.environ.get("DI_STUB_MK_ALIGNMENT", "128"))]


def _impl_get_mk_alignment(*_a: object, **_k: object) -> int:
    return _MK_ALIGNMENT[0]


def _impl_set_mk_alignment(value: object = None, *_a: object, **_k: object) -> None:
    if value is not None:
        _MK_ALIGNMENT[0] = int(value)  # type: ignore[arg-type]


_IMPL = {
    "deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout": _impl_get_mk_alignment,
    "deep_gemm.get_mk_alignment_for_contiguous_layout": _impl_get_mk_alignment,
    "deep_gemm.set_mk_alignment_for_contiguous_layout": _impl_set_mk_alignment,
}


def _make_stub(module: str, name: str):
    qualname = f"{module}.{name}"
    _record(qualname, "ATTR")

    if qualname in _IMPL:
        impl = _IMPL[qualname]

        def _real(*args: object, **kwargs: object):
            _record(qualname, "IMPL")
            return impl(*args, **kwargs)

        _real.__name__ = name
        return _real

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
