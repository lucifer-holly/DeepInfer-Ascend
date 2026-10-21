"""910C bring-up 占位：deep_gemm（上游由 deep_gemm-Ascend 提供，只支持 950）。详见 _di_vendor_stub.py。"""

from _di_vendor_stub import module_getattr

__getattr__ = module_getattr("deep_gemm")
