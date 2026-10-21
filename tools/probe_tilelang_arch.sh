#!/bin/bash
# 一个组合一个进程重跑 —— 排除 tilelang 的进程内存缓存。
#
# 上一轮 CANN 9.2 下 8/8「编译 OK」，但日志里有
#   [tilelang.cache.kernel_cache:WARNING] Found kernel 'main' in memory cache
# 而 ASCEND_NPU_ARCH 是编译时读的环境变量，很可能不进 cache key。
# 那样 dav-2201 拿到的其实是 dav-3510 编好的产物 —— 必须证伪。
#
# 做法：① 每个组合独立进程；② 环境变量在进程启动前就设好；
#       ③ 每次清空磁盘与内存缓存；④ 直接抓 bisheng 的真实命令行验证 --npu-arch。
set -u
VENV="${TL_VENV:-/tmp/tl-venv}"
source "${CANN_PREFIX:-/opt/cann-9.2}/ascend-toolkit/set_env.sh"

cat > /tmp/one_combo.py <<'PY'
import os, sys, traceback
mode, arch, tile, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert os.environ.get("ASCEND_NPU_ARCH") == arch, "env 没设对"

import tilelang
import tilelang.ascend.language as T
from tilelang.contrib import bisheng

# 证据 1：tilelang 自己认为的 arch
print(f"  get_npu_arch() -> {bisheng.get_npu_arch()}")

# 证据 2：拦截真实的编译调用，记录命令行里的 --npu-arch
seen = []
_orig = bisheng.compile_ascend
def spy(*a, **kw):
    seen.append(kw.get("npu_arch") or bisheng.get_npu_arch())
    return _orig(*a, **kw)
bisheng.compile_ascend = spy

num_blocks, num_stages = 64, 2
N = tile * num_blocks * 2

@T.prim_func
def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
    with T.Kernel(num_blocks) as bx:
        a = T.alloc_shared((tile,), "float32")
        b = T.alloc_shared((tile,), "float32")
        c = T.alloc_shared((tile,), "float32")
        T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
        for step in T.Pipelined(N // (tile * num_blocks), num_stages=num_stages):
            begin = (step * num_blocks + bx) * tile
            T.copy(A[begin:begin + tile], a)
            T.copy(B[begin:begin + tile], b)
            if mode == "simt":
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
            else:
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        ar = T.simd.vld(a[i * 64]); br = T.simd.vld(b[i * 64])
                        T.simd.vsts(c[i * 64], T.simd.vadd(ar, br, mask), mask)
            T.copy(c, C[begin:begin + tile])

try:
    tilelang.compile(main, target="ascend", out_idx=-1)
except Exception:
    tb = traceback.format_exc()
    errs = [l.strip()[:150] for l in tb.splitlines() if "error:" in l]
    # 去重保序
    seen_e, uniq = set(), []
    for e in errs:
        if e not in seen_e:
            seen_e.add(e); uniq.append(e)
    print(f"  实际编译用的 arch: {seen or '（没走到编译）'}")
    print("  RESULT=FAIL")
    for e in uniq[:6]:
        print(f"    {e}")
    sys.exit(1)
print(f"  实际编译用的 arch: {seen or '（命中缓存，没真编）'}")
print("  RESULT=OK" if seen else "  RESULT=CACHED-可疑")
PY

printf '%-34s %s\n' "组合" "结果"
echo "----------------------------------------------------------------"
for mode in simt simd; do
  for arch in dav-3510 dav-2201; do
    for tile in 2048 8192; do
      threads=$([ "$tile" = 2048 ] && echo 512 || echo 2048)
      tag="$mode $arch tile=$tile"
      echo "=== $tag ==="
      rm -rf /tmp/tlcache ~/.tilelang /root/.tilelang 2>/dev/null
      env ASCEND_NPU_ARCH="$arch" TILELANG_CACHE_DIR="/tmp/tlc-$mode-$arch-$tile" \
          "$VENV/bin/python" /tmp/one_combo.py "$mode" "$arch" "$tile" "$threads" 2>&1 \
          | grep -vE "^\[WARNING\]|triton\.backends|^\[ERROR\]|TileLang begins"
      echo
    done
  done
done
