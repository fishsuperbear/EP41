#!/bin/bash
app_names=("em_proc_a" "em_proc_b")
source ../../sm_test_common/setup.sh "${app_names[*]}"

ROOT_PATH=`realpath ../../`
ROOT_EM_PATH="/app"

sed -i "/Driving/d" $ROOT_PATH/test/emproc_test/em_proc_a/etc/MANIFEST.json
sed -i "/Normal/d" $ROOT_PATH/test/emproc_test/em_proc_b/etc/MANIFEST.json
sed -i "/Parking/d" $ROOT_PATH/test/emproc_test/em_proc_b/etc/MANIFEST.json
sync


#确保启动模式是Normal
startmode=`grep Normal $ROOT_EM_PATH/conf/startup_manifest.json|wc -l`
if [ $startmode -ne 1 ]
then
    echo "change start mode to Normal in $ROOT_EM_PATH/conf/startup_manifest.json"
    sed -i 's/\"[a-zA-Z]*\"$/\"Normal\"/g' $ROOT_EM_PATH/conf/startup_manifest.json
fi

../../sm_test_common/start_apps.sh "${app_names[*]}"