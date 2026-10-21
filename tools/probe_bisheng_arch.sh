#!/bin/bash
# bisheng 到底认哪些 --npu-arch 值？
#
# 背景：CANN 的 platform_config 写 dav-c220-cube / dav-c310-cube，
# 而 tilelang 传给 bisheng 的是 dav-3510 —— 两套命名不是一个体系。
# 所以 "910C 不支持" 和 "我传错了名字" 必须先分开。
# 做法：① 从二进制里抠出 dav-* 字符串表；② 拿空 kernel 逐个试编。
set -u
ASCEND="${ASCEND_HOME_PATH:-$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -r | head -1)}"
BIN="$ASCEND/tools/bisheng_compiler/bin/bisheng"
LIBDIR="$ASCEND/tools/bisheng_compiler/lib"

echo "###### 1. bisheng 及其插件里出现的 dav-* 字符串 ######"
for f in "$BIN" "$LIBDIR"/*.so*; do
    [ -f "$f" ] || continue
    hits=$(strings -a "$f" 2>/dev/null | grep -oE "^dav-[a-z0-9-]+$" | sort -u | tr '\n' ' ')
    [ -n "$hits" ] && echo "  [$(basename "$f")] $hits"
done
echo "--- 全集 ---"
{ strings -a "$BIN" 2>/dev/null; for f in "$LIBDIR"/*.so*; do strings -a "$f" 2>/dev/null; done; } \
    | grep -oE "^dav-[a-z0-9-]+$" | sort -u | tr '\n' ' '
echo; echo

echo "###### 2. 实编测试：空 kernel 逐个 arch 试 ######"
cat > /tmp/probe.asc <<'ASC'
extern "C" __global__ __aicore__ void probe() {}
ASC

for arch in dav-c220 dav-c220-cube dav-c220-vec dav-2201 dav-c310 dav-c310-cube dav-3510 dav-c100 dav-m200 dav-c220-cube-vec; do
    out=$("$BIN" -O2 -fPIC -std=c++20 --npu-arch="$arch" -c /tmp/probe.asc -o /tmp/probe.o 2>&1)
    if [ $? -eq 0 ]; then
        printf '  ✅ %-20s 编译通过\n' "$arch"
    else
        first=$(echo "$out" | grep -iE "error|unsupported" | head -1 | cut -c1-110)
        printf '  ❌ %-20s %s\n' "$arch" "${first:-未知失败}"
    fi
    rm -f /tmp/probe.o
done
echo

echo "###### 3. 本机 soc 对应的 CCEC 版本（权威来源）######"
grep -E "^CCEC_|^SoC_version|^NpuArch|^AIC_version" \
    "$ASCEND"/*-linux/data/platform_config/Ascend910_9382.ini | sed 's/^/    /'
echo

echo "###### 4. 对照：CANN 里所有 platform_config 的 CCEC 版本都有哪些 ######"
grep -h "^CCEC_AIC_version" "$ASCEND"/*-linux/data/platform_config/*.ini \
    | sort | uniq -c | sort -rn | sed 's/^/    /'
