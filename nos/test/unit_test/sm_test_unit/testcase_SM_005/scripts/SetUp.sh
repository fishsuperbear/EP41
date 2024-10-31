#!/bin/bash
app_names=("em_proc_a" "em_proc_b")
source ../../sm_test_common/setup.sh "${app_names[*]}"

ROOT_PATH=`realpath ../../`

sed -i "/Driving/d" $ROOT_PATH/test/emproc_test/em_proc_a/etc/MANIFEST.json
sed -i "/Normal/d" $ROOT_PATH/test/emproc_test/em_proc_b/etc/MANIFEST.json
sed -i "/Parking/d" $ROOT_PATH/test/emproc_test/em_proc_b/etc/MANIFEST.json
sync

../../sm_test_common/start_apps.sh "${app_names[*]}"