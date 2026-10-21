#!/usr/bin/env python3
"""核对 torch_npu 融合算子的真实签名 —— 替换 DeepInfer tilelang kernel 的候选面。

只读 schema，不执行任何算子（这个容器没挂卡）。
目的是确认「能替换」这个判断建在真实接口上，而不是算子名字看着像。
"""

import torch
import torch_npu  # noqa: F401

CANDIDATES = {
    "indexer (DSA)": ["npu_quant_lightning_indexer"],
    "MoE 分发/聚合": ["npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2"],
    "MoE 分组 GEMM": ["npu_grouped_matmul_swiglu_quant", "npu_grouped_matmul_finalize_routing"],
    "MoE 路由": ["npu_moe_gating_top_k", "npu_moe_init_routing_v2"],
    "MLA / attention": ["npu_fusion_attention_v3", "npu_quant_fusion_attention"],
    "KV cache + rope": ["npu_kv_rmsnorm_rope_cache_v2", "npu_dequant_rope_quant_kvcache"],
    "norm + 量化": ["npu_add_rms_norm_quant", "npu_rms_norm_dynamic_mx_quant"],
    "swiglu + 量化": ["npu_dequant_swiglu_quant"],
    "动态量化": ["npu_dynamic_quant", "npu_dynamic_block_quant"],
    "量化 matmul": ["npu_quant_matmul"],
}


def schema_of(name: str) -> str:
    """从 torch 的算子注册表里取 schema；取不到就退回 Python 签名。"""
    for ns in ("npu", "aten"):
        try:
            packet = getattr(getattr(torch.ops, ns), name)
            return "\n".join(f"      {ov}" for ov in [str(packet._schemas[k]) for k in packet._schemas])
        except Exception:
            pass
    fn = getattr(torch_npu, name, None)
    if fn is None:
        return "      ❌ 取不到"
    doc = (fn.__doc__ or "").strip().splitlines()
    return "      " + (doc[0][:180] if doc else f"<built-in {name}，无 schema/doc>")


for group, names in CANDIDATES.items():
    print(f"\n━━━ {group} ━━━")
    for n in names:
        exists = hasattr(torch_npu, n)
        print(f"  {'✅' if exists else '❌'} {n}")
        if exists:
            print(schema_of(n))
