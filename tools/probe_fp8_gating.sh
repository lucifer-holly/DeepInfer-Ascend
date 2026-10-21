#!/bin/bash
# CANN 9.2 下 dav-2201(910C) 到底行不行 —— 绕开 tilelang 直接问 bisheng。
#
# 9.0.1 上的两个结论要在 9.2 上重验：
#   ① asc_fp8.h 的门控是否仍排除 2201（硬件决定，预期不变）
#   ② simt_api/device_functions.h 的 long2/ulong2/char2 等类型在 2201 下是否补上了
#      ← 这一条决定 tilelang 路线在 910C 上的死活
set -u
C92="${CANN_PREFIX:-/opt/cann-9.2}/cann-9.2.0"
ASC=$C92/aarch64-linux/asc
BIN=$C92/tools/bisheng_compiler/bin/bisheng

echo "###### 0. 版本确认 ######"
"$BIN" --version 2>&1 | head -2
ls -d $C92 2>/dev/null
echo

echo "###### 1. FP8 门控在 9.2 下变了吗（预期：没变，硬件不会变）######"
echo "--- 9.2 的 asc_fp8.h ---"
grep -nE "__NPU_ARCH__ *==" "$ASC/include/simt_api/asc_fp8.h" 2>/dev/null | head -5 | sed 's/^/    /'
echo "--- 对照 9.0.1 ---"
grep -nE "__NPU_ARCH__ *==" "${OLD_CANN:-/usr/local/Ascend/cann-9.0.1}"/*-linux/asc/include/simt_api/asc_fp8.h 2>/dev/null | head -5 | sed 's/^/    /'
echo

echo "###### 2. device_functions.h 里 long2/char2 的定义来源 ######"
echo "--- 9.2 里这些类型在哪定义 ---"
grep -rn "long2\b" "$ASC/include/simt_api/"*.h 2>/dev/null | head -5 | sed 's/^/    /'
echo "--- 有没有按 arch 门控 ---"
grep -nE "__NPU_ARCH__" "$ASC/include/simt_api/device_functions.h" 2>/dev/null | head -8 | sed 's/^/    /'
echo

echo "###### 3. 实编：最小 kernel，只 include device_functions.h ######"
cat > /tmp/t_simt.asc <<'ASC'
#include <simt_api/device_functions.h>
extern "C" [aicore] void probe_simt() {}
ASC
cat > /tmp/t_simd.asc <<'ASC'
#include <c_api/asc_simd.h>
extern "C" [aicore] void probe_simd() {}
ASC
cat > /tmp/t_bare.asc <<'ASC'
extern "C" [aicore] void probe_bare() {}
ASC

for arch in dav-2201 dav-3510; do
    echo "═══ $arch ═══"
    for t in bare simt simd; do
        out=$("$BIN" -O2 -fPIC -std=c++20 --npu-arch="$arch" \
              -I"$ASC/include" -I"$ASC/impl" -I"$C92/aarch64-linux/include" \
              -c /tmp/t_$t.asc -o /tmp/t_$t.o 2>&1)
        if [ $? -eq 0 ]; then
            printf '  ✅ %-6s 编译通过\n' "$t"
        else
            n=$(echo "$out" | grep -c "error:")
            printf '  ❌ %-6s %s 个 error，首条: %s\n' "$t" "$n" "$(echo "$out" | grep "error:" | head -1 | sed 's|.*/||' | cut -c1-90)"
        fi
        rm -f /tmp/t_$t.o
    done
done
echo

echo "###### 4. 9.2 的 Ascend910_9382.ini 关键字段（对比 9.0.1）######"
grep -E "^NpuArch|^CCEC_|^ub_size|^cube_core_cnt|^vector_core_cnt|^cube_freq|^support_" \
    "$C92/aarch64-linux/data/platform_config/Ascend910_9382.ini" 2>/dev/null | sed 's/^/    /'
echo "--- mmad 支持的 dtype（9.2）---"
grep -oE "Intrinsic_mmad\|[^ ]*" "$C92/aarch64-linux/data/platform_config/Ascend910_9382.ini" 2>/dev/null | head -1 | fold -w 110 | sed 's/^/    /'
echo "--- DtypeMKN（9.2）---"
sed -n '/\[DtypeMKN\]/,/^\[/p' "$C92/aarch64-linux/data/platform_config/Ascend910_9382.ini" 2>/dev/null | head -12 | sed 's/^/    /'
