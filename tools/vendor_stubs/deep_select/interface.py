"""910C bring-up 占位：deep_select.interface。

注意 deepinfer/npu_ops/__init__.py 已经把 `deep_select.interface.topk`
patch 成 `deepinfer.npu_ops.deep_select.topk`，所以正常路径不会落到这里。
"""

from _di_vendor_stub import module_getattr

__getattr__ = module_getattr("deep_select.interface")
