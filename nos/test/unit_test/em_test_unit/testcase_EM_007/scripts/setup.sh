#!/bin/bash
app_names=("em_proc_a" "em_proc_b" "em_proc_c" "em_proc_d" "em_proc_e" \
           "em_proc_f" "em_proc_g" "em_proc_h" "em_proc_i" "em_proc_j" \
           "em_proc_k" "em_proc_l" "em_proc_m" "em_proc_n"  "em_proc_o")

source ../../em_test_common/setup.sh "${app_names[*]}"

ROOT_PATH=`realpath ../../`
ROOT_EM_PATH=/app

#确保启动模式是Normal
startmode=`grep Normal $ROOT_EM_PATH/conf/startup_manifest.json|wc -l`
if [ $startmode -ne 1 ]
then
    echo "change start mode to Normal in $ROOT_EM_PATH/conf/startup_manifest.json"
    sed -i 's/\"[a-zA-Z]*\"$/\"Normal\"/g' $ROOT_EM_PATH/conf/startup_manifest.json
fi
#
sed -i 's/\"REPORT_TER_DELAY_TIME=0\",/\"REPORT_TER_DELAY_TIME=10\"/' $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json
sync

cat $ROOT_PATH/em_test_common/emproc/em_proc_b/etc/MANIFEST.json

../../em_test_common/start_apps.sh "${app_names[*]}"


