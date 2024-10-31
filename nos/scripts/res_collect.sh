#!/bin/sh

USER="/opt/usr"                                           # user分区路径
LOG_BAK="/opt/usr/log_bak"                                # log_bak分区路径
RES_COLLECT="/opt/usr/log_bak/res_collect"                # 收集路径

echo "------------------------------Start------------------------------"
echo -e "\033[43;30m Please wait a few minutes, don't use ctrl+C to stop it! \033[0m"
echo ""

# 首先删除收集路径目录下旧的内容
rm -rf ${RES_COLLECT}/*
rm -rf ${LOG_BAK}/res_collect.*.gz
echo "Collect start."
echo ""

# 判断收集目录所在分区剩余空间是否足够
usage=$(df -P ${LOG_BAK} | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$usage" -gt 90 ]; then
        echo "NO space left on log_bak partition!!!!! exit"
        exit 1
fi

# 版本信息
echo "Version file collecting......"
echo ""
VERSION_DIR="/app/ver"
VERSION_FILE="/app/version.json"
cp -rf ${VERSION_DIR} ${RES_COLLECT}
cp -f ${VERSION_FILE} ${RES_COLLECT}

# LOG文件
echo "Log file collecting......"
echo ""
mkdir -p ${RES_COLLECT}/log
cp -rf ${USER}/log/mcu_log ${RES_COLLECT}/log
cp -rf ${USER}/log/ota_log ${RES_COLLECT}/log
cp -rf ${USER}/log/soc_log ${RES_COLLECT}/log
cp -rf /svp_log ${RES_COLLECT}/log
cp -rf ${USER}/log/system_monitor_log ${RES_COLLECT}/log
cp -rf /cfg/conf_app/file_monitor ${RES_COLLECT}/log/system_monitor_log

# COL文件
echo "Col file collecting......"
echo ""
mkdir -p ${RES_COLLECT}/col
cp -rf ${USER}/col/fm ${RES_COLLECT}/col

# RUNTIME_SERVICE下进程信息
echo "Hz_Bin info collecting......"
echo ""
find /app/runtime_service -name bin | xargs ls -l > ${RES_COLLECT}/hz_bin_info

# 根据输入参数判断是否打包压缩文件
if [ "$1" != all ]; then
    find "$RES_COLLECT" -type f \( -name "*.zip" -o -name "*.tar.gz" \) -delete
fi

# 所有数据收集完成
echo "Collect end."
echo ""

# 将收集到的数据打包
echo "File compress start."
echo ""
tar -zcPf ${LOG_BAK}/res_collect.`date +%Y-%m-%d-%H-%M-%S`.tar.gz ${RES_COLLECT}
chmod 777 ${LOG_BAK}/res_collect.*.gz
echo "File compress end."
echo ""

# 打包后删除收集路径目录下的内容
rm -rf ${RES_COLLECT}/*
echo "-------------------------------End-------------------------------"
