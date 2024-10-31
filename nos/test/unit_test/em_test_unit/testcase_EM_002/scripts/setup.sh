#!/bin/bash
app_names=("em_proc_a" "em_proc_b" "em_proc_c" "em_proc_d" "em_proc_e" \
           "em_proc_f" "em_proc_g" "em_proc_h" "em_proc_i" "em_proc_j" \
           "em_proc_k" "em_proc_l" "em_proc_m" "em_proc_n"  "em_proc_o" "em_proc_ax")

source ../../em_test_common/setup.sh "${app_names[*]}"

ROOT_PATH=`realpath ../../`

if [ -d "$ROOT_PATH/em_test_common/emproc" ];then
    if [ ! -d "$ROOT_PATH/em_test_common/emproc/em_proc_ax" ]
    then
        cp -rf $ROOT_PATH/em_test_common/emproc/em_proc_a $ROOT_PATH/em_test_common/emproc/em_proc_ax
        sed -i 's/\"hz_app_aProcess\"/\"hz_app_axProcess\"/' $ROOT_PATH/em_test_common/emproc/em_proc_ax/etc/MANIFEST.json
        sed -i 's/\"-s a\",/\"-m 200\"/' $ROOT_PATH/em_test_common/emproc/em_proc_ax/etc/MANIFEST.json
        sed -i '/\"-c \/opt\/usr\/app\/em_proc_a\/conf\/proc_a.conf\"/d' $ROOT_PATH/em_test_common/emproc/em_proc_ax/etc/MANIFEST.json
        sync
    fi
fi

cat  $ROOT_PATH/em_test_common/emproc/em_proc_ax/etc/MANIFEST.json

../../em_test_common/start_apps.sh "${app_names[*]}"
