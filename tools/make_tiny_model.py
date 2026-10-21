#!/usr/bin/env python3
"""从上游 config.json 造一个极小的 V4.1 配置，供 bring-up 用。

配合 `load_format="dummy"`（llm_engine.py 的 skip_load）使用，不需要任何权重文件 ——
目的是把「引擎能不能转」与「510 GB 权重」解耦。

只裁层数与专家数，其余 per-layer 维度保持原值，以免触发与移植无关的断言。

用法：
    python3 make_tiny_model.py --out ./tinymodel \\
        [--config /path/to/upstream/config.json] [--layers 6] [--experts 8]

不给 --config 时从 HF 拉官方 config.json（需要网络）。
tokenizer.json / tokenizer_config.json 也会一并拉取（引擎要 tokenizer）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

HF_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
HF_RAW = f"https://huggingface.co/{HF_REPO}/resolve/main"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def fetch(name: str, dest: Path) -> None:
    if dest.exists():
        print(f"  已存在，跳过 {dest.name}")
        return
    print(f"  下载 {name} ...", flush=True)
    with urllib.request.urlopen(f"{HF_RAW}/{name}", timeout=180) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  ✅ {dest.name}  {dest.stat().st_size / 1e6:.2f} MB")


def shrink(cfg: dict, n_layer: int, n_experts: int) -> dict:
    text = cfg["text_config"]
    text["num_hidden_layers"] = n_layer
    # 关掉 MTP/DFlash：compress_ratios 的长度须等于 n_layer + num_nextn_predict_layers
    text["num_nextn_predict_layers"] = 0
    text["dspark_target_layer_ids"] = []

    # compress_ratios 必须含一个 1 —— _n_prefill_layer 取的就是第一个 1 的下标。
    # 形如 [0, 0, 2, ..., 2, 1, ..., 1]：前两层 win、中间 compress、其后是 yoco consumer。
    if n_layer < 4:
        raise SystemExit("--layers 至少要 4（需要容纳 win / compress / yoco 三段）")
    n_compress = (n_layer - 2) // 2
    ratios = [0, 0] + [2] * n_compress + [1] * (n_layer - 2 - n_compress)
    assert len(ratios) == n_layer
    text["compress_ratios"] = ratios
    first_one = ratios.index(1)

    # cross_layer.py 断言 layer_kv_map[n_prefill_layer] == n_prefill_layer，
    # 所以 kv_source_layer_ids 必须包含第一个 ratio==1 的层。
    text["kv_source_layer_ids"] = sorted({2, first_one})
    text["index_source_layer_ids"] = sorted({2, first_one})
    text["candidate_source_layer_id"] = None       # 关掉 candidate topk
    text["engram_layer_ids"] = []                  # 关掉 engram（避开 deep_ep.EngramBuffer）

    if n_experts < text["num_experts_per_tok"]:
        raise SystemExit(f"--experts 不能小于 num_experts_per_tok={text['num_experts_per_tok']}")
    text["n_routed_experts"] = n_experts

    cfg["vision_config"]["num_hidden_layers"] = 2  # 视觉塔也裁小，以防被构建

    # A2/A3 无原生 FP8/FP4：按 BF16 构建（需配合 patches/0002）
    cfg["quantization_config"] = {"quant_method": "none", "expert_dtype": "bf16"}
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="上游 config.json；不给则从 HF 下载")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--experts", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.config:
        cfg = json.loads(args.config.read_text())
    else:
        src = args.out / "config.upstream.json"
        fetch("config.json", src)
        cfg = json.loads(src.read_text())
        src.unlink()

    cfg = shrink(cfg, args.layers, args.experts)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in TOKENIZER_FILES:
        fetch(name, args.out / name)

    t = cfg["text_config"]
    print("\n✅ 生成完毕:", args.out)
    print(f"   num_hidden_layers      = {t['num_hidden_layers']}")
    print(f"   compress_ratios        = {t['compress_ratios']}")
    print(f"   _n_prefill_layer 将取   = {t['compress_ratios'].index(1)}")
    print(f"   kv_source_layer_ids    = {t['kv_source_layer_ids']}")
    print(f"   n_routed_experts       = {t['n_routed_experts']} (per_tok={t['num_experts_per_tok']})")
    print(f"   engram                 = {'关闭' if not t['engram_layer_ids'] else '开启'}")
    print(f"   quantization_config    = {cfg['quantization_config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
