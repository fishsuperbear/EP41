#!/bin/bash
app_names=("em_proc_a" "em_proc_b" "sm_proc_b")
source ../../sm_test_common/setup.sh "${app_names[*]}"

ROOT_PATH=`realpath ../../`
ROOT_EM_PATH=/app

sed -i "/Driving/d" $ROOT_PATH/test/emproc_test/em_proc_a/etc/MANIFEST.json
sed -i "/Normal/d" $ROOT_PATH/test/emproc_test/em_proc_b/etc/MANIFEST.json
sed -i "/Parking/d" $ROOT_PATH/test/emproc_test/sm_proc_b/etc/MANIFEST.json
sync

# em_a	
# 	    "Normal.1",
# 	    "Parking.3",
# 	    "OTA.1"
# em_b	
# 	    "Driving.1",
# 	    "Parking.2",
# 	    "OTA.2"
# sm_B	
# 	    "Driving.99",
# 	    "Normal.99"

#删除proB的log文件。
rm /log/proB_*.log
#确保启动模式是Normal
startmode=`grep Normal $ROOT_EM_PATH/conf/startup_manifest.json|wc -l`
if [ $startmode -ne 1 ]
then
    echo "change start mode to Normal in $ROOT_EM_PATH/conf/startup_manifest.json"
    sed -i 's/\"[a-zA-Z]*\"$/\"Normal\"/g' $ROOT_EM_PATH/conf/startup_manifest.json
fi

../../sm_test_common/start_apps.sh "${app_names[*]}"