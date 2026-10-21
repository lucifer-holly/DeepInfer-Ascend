"""910C bring-up 占位：flash_mla（上游 FlashMLA 的昇腾 kernel 只支持 950）。"""

from _di_vendor_stub import module_getattr

__getattr__ = module_getattr("flash_mla")
