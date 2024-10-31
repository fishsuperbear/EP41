#!/bin/bash
app_names=("em_proc_a" "em_proc_b")

for i in ${app_names[*]}; do
    killall -9 $i
done

killall -9 execution_manager
sleep 5

ROOT_PATH=`realpath ../../`
ROOT_EM_PATH="/app"

export EM_DEV_APP_DIR=$ROOT_PATH/test/emproc_test
export EM_DEV_CONF_DIR=$ROOT_EM_PATH/conf/
export LD_LIBRARY_PATH=$ROOT_EM_PATH/lib:/usr/lib:/lib64:../lib:$LD_LIBRARY_PATH

../../sm_test_common/start_apps.sh "${app_names[*]}"