#!/bin/bash
# 175 号机环境事实采集 —— 在宿主机上跑，只读，不改任何东西。
#   bash collect_platform_facts.sh          # 在装了昇腾驱动的主机或容器里直接跑
#
# 关注点（每一条都对应一个已知的移植判据）：
#   uname -r        → 内核 ≤5.10 时 aclrtHostRegisterV2 注册非锁页内存直接 107017
#   npu-smi         → 卡被占着就让开
#   docker ps       → 别人的容器只读，绝不停
#   CANN 版本       → tilelang 950 后端真实下限是 9.2.0 weekly；910C 官方上限 9.1.X
#   acl_rt.h        → ACL_HOST_REG_* flag 的真实枚举值（不靠猜）

echo "###### 1. 主机 ######"
hostname; uname -r; uptime
echo "--- 在线用户 ---"; who

echo; echo "###### 2. NPU 占用 ######"
npu-smi info 2>&1 | head -40

echo; echo "###### 3. 容器（只看，不动） ######"
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1 | head -30
echo "--- 本地镜像 ---"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>&1 | head -30

echo; echo "###### 4. CANN / 驱动 ######"
ls -d /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/* 2>/dev/null
cat /usr/local/Ascend/driver/version.info 2>/dev/null
npu-smi info -t board -i 0 2>&1 | grep -iE "chip name|software version|firmware" | head

echo; echo "###### 5. ACL_HOST_REG_* 真实枚举值 ######"
# 不硬编码 flag，从头文件读出来。找不到就说找不到。
for h in $(find /usr/local/Ascend -name 'acl_rt.h' 2>/dev/null | head -3); do
    echo "--- $h ---"
    grep -nE "ACL_HOST_REG_[A-Z_]+" "$h" | head -20
done

echo; echo "###### 6. 盘与自己的目录 ######"
df -h /data /public-flash 2>/dev/null
echo "--- 工作目录 ---"; ls -la "${WORKDIR:-$PWD}" 2>/dev/null | head -5

echo; echo "###### 7. 卡能不能真的打开（npu-smi 说 OK 不算数） ######"
python3 -c "import os; fd=os.open('/dev/davinci0', os.O_RDWR); os.close(fd); print"'("davinci0 可打开")' 2>&1 | tail -2
dmesg -T 2>/dev/null | grep -i "Conflict open udevid" | tail -3 || echo "（无 namespace 冲突记录）"
