#!/bin/bash
ROOT_PATH=`realpath ../../`

# OPT ["DelayRun" - DR - 延迟启动上报，  "DelayExit" -DE- 延迟退出上报,  "Both" - BO - 启动和退出都延迟上报]
OPT=""
if [ -n "$1" ];then
    OPT=$1
fi

if [ -d "$ROOT_PATH/em_test_common/emproc/em_proc_b" ];then

    if [ "$OPT" = "DR" ];then
        sed -i 's/\"REPORT_RUN_DELAY_TIME=0\",/\"REPORT_RUN_DELAY_TIME=10\",/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
    fi

    if [ "$OPT" = "DE" ];then
        sed -i 's/\"REPORT_TER_DELAY_TIME=0\",/\"REPORT_TER_DELAY_TIME=10\"/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
    fi

    if [ "$OPT" = "BO" ];then
        sed -i 's/\"REPORT_RUN_DELAY_TIME=0\",/\"REPORT_RUN_DELAY_TIME=10\",/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
        sed -i 's/\"REPORT_TER_DELAY_TIME=0\",/\"REPORT_TER_DELAY_TIME=10\"/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
    fi

    if [ "$OPT" = "" ];then
        sed -i 's/\"REPORT_RUN_DELAY_TIME=10\",/\"REPORT_RUN_DELAY_TIME=0\",/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
        sed -i 's/\"REPORT_TER_DELAY_TIME=10\",/\"REPORT_TER_DELAY_TIME=0\"/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
    fi

    sync
else
    echo -e "\033[1;31m ERROR: em_proc_b dir not found \033[0m"
fi





