#! /usr/bin/bash

set -e

curpath=$(readlink -f "$(dirname "$0")")
PLATFORM=x86_2004
if [ -n "$1" ];then
    PLATFORM=$1
fi

SRC_DIR=output
SRC_ETC_CONF=$SRC_DIR/$PLATFORM/config

SRC_BIN=$SRC_DIR/$PLATFORM/bin
SRC_CONF=$SRC_DIR/$PLATFORM/conf
SRC_LIB=$SRC_DIR/$PLATFORM/lib
SRC_SCRIPT=$SRC_DIR/$PLATFORM/scripts
SRC_TOOL=$SRC_DIR/$PLATFORM/tools

RELEASE_DIR=$SRC_DIR/"nos_"$PLATFORM
RELEASE_APP=$RELEASE_DIR/runtime_service
RELEASE_LIB=$RELEASE_DIR/lib
RELEASE_BIN=$RELEASE_DIR/bin
RELEASE_CONF=$RELEASE_DIR/conf
RELEASE_SCRIPT=$RELEASE_DIR/scripts
RELEASE_TOOL=$RELEASE_DIR/tools


filter=(
    "execution_manager"
    "config_server"
    "diag_server"
    "remote_diag"
    "phm_server"
    "update_manager"
    "system_monitor"
    "state_machine"
    "log_server"
    "devm_server"
    "sys_statemgr"
    "can_tsync_center"
    "udp_tsync_master"
    "udp_tsync_slave"
    "adf-lite-process"
    "aldbg"
    "bag"
    "topic"
    "devm"
    "fm"
    "cfg"
    "stmm"
    "stmm_info"
    "smdbg"
    "log"
    "camera_venc"
    "neta_someipd"
    "soc_to_mcu"
    "sensor_trans"
    "network_capture"
    "neta_lidar"
    "data_collection"
    "dc_trigger"
    "dc_mcu"
    "remote_config"
    "cm_freqchecker_tool"
    "crypto_server"
    "pki_service"
    "tsync"
    "event_test"
    "soc_to_hmi"
    "hz_time"
    "lidar_test"
    "pointcloud"
    "log_collector"
    "cm_event_test"
    "method_test_client"
    "method_test_server"
    # "hz_dvr"
    "notify_main"
    "ihbc_tsr_tlr"
    "extwdg"
)

if [[ $PLATFORM == "orin" ]]; then
    filter=("${filter[*]}" "nvs_producer")
    filter=("${filter[*]}" "ifdata")
    filter=("${filter[*]}" "iostat")
    filter=("${filter[*]}" "mpstat")
    filter=("${filter[*]}" "pidstat")
    filter=("${filter[*]}" "gdb")
    filter=("${filter[*]}" "killall")
    filter=("${filter[*]}" "perf")
    filter=("${filter[*]}" "memtester")
fi



# Clean "orin_soc" dir
if [ -d "$curpath/${RELEASE_DIR}" ]
then
    echo "clean  ${RELEASE_DIR} dir  "
    rm -rf  $curpath/${RELEASE_DIR}/*
fi
# Clean "config" dir
if [ -d "$curpath/${SRC_ETC_CONF}" ]
then
    echo "clean  ${SRC_ETC_CONF} dir  "
    rm -rf  $curpath/${SRC_ETC_CONF}/*
fi


function create_proc_image(){
    echo "create proc image($1)"
    # Copy all process files to release folder
    if [ -d "$curpath/$RELEASE_BIN" ]
    then
        ProcessName=$(ls "$curpath/$RELEASE_BIN")
        for i in $ProcessName
        do
            if [ "$1" = "$i" ]
            then
                # create the process folder first
                #echo "$curpath/${RELEASE_APP}/$i"
                #echo  "Create Process $i release folder Start"
                mkdir -p "$curpath/$RELEASE_APP/$i"
                mkdir -p "$curpath/$RELEASE_APP/$i/bin"
                mkdir -p "$curpath/$RELEASE_APP/$i/etc"
                mkdir -p "$curpath/$RELEASE_APP/$i/conf"
                sync
                mv "$curpath/$RELEASE_BIN/$i"  "$curpath/$RELEASE_APP/$i/bin"
                copy_proc_etc $i
                #echo -e "Create Process $i release \033[32m success \033[0m"
            fi
        done
    else
        echo -e "\033[1;31m ERROR: $curpath/$RELEASE_BIN dir not found \033[0m"
        exit
    fi
}

function copy_proc_conf(){
    #echo "copy process $1 conf"
    if [ $# -ge 3 ]
    then
        if [ ! -d "$curpath/$RELEASE_APP/$1/conf" ]
        then
            mkdir -p "$curpath/$RELEASE_APP/$1/conf"
        fi

        if [ "$2" = "-d" ];then   #dir
            if [ -d "$3" ];then
                cp -rf $3/* "$curpath/$RELEASE_APP/$1/conf"
            else
                echo -e "\033[1;31m ERROR: $3 dir not found \033[0m"
                exit
            fi
        elif [ "$2" = "-f" ];then #file
            num=0
            for arg in "$@"
            do
                num=$(($num+1))
                if [ $num -ge 3 ];then
                    #echo "cp $1 conf file: $arg"
                    if [ -f "$arg" ];then
                        cp -rf "$arg" "$curpath/$RELEASE_APP/$1/conf"
                    else
                        cp -rf "$curpath/$RELEASE_CONF/$arg" "$curpath/$RELEASE_APP/$1/conf"
                    fi
                fi
            done
        fi
    else
        echo -e"\033[1;31m ERROR:invalid param, copy process conf fail \033[0m"
        exit
    fi
}

function copy_proc_etc(){
    #echo "copy process $1 etc config"
    if [ $# -ge 3 ]
    then
        if [ "$2" = "-d" ];then   #dir
            if [ -d "$3" ];then
                cp -rf $3/* "$curpath/$RELEASE_APP/$1/etc"
            else
                echo -e "\033[1;31m ERROR: $3 dir not found \033[0m"
                exit
            fi
        fi
    else
        if [ -d "$curpath/$SRC_ETC_CONF/$1/" ]
        then
            cp -rf $curpath/$SRC_ETC_CONF/$1/* "$curpath/$RELEASE_APP/$1/etc"
        else
            echo -e "\033[1;31m ERROR: $1 etc config not found \033[0m"
            exit
        fi
    fi
}

function create_algorithm_etc(){
    AlgoProcArry=(fisheye_perception hz_perception_state_machine parking_fusion parking_slam perception planning mapping uss_perception mapping_plugin)
    for pname in ${AlgoProcArry[@]}; do
        mkdir -p "$curpath/$RELEASE_APP/$pname"
        mkdir -p "$curpath/$RELEASE_APP/$pname/bin"
        mkdir -p "$curpath/$RELEASE_APP/$pname/etc"
        mkdir -p "$curpath/$RELEASE_APP/$pname/conf"
        cp -rf $curpath/$SRC_ETC_CONF/$pname/* "$curpath/$RELEASE_APP/$pname/etc"
    done
}

echo -e "\033[0;36m start to create the release image ... \033[0m"
# Create "/app/runtime_service,lib,bin,conf,script"  dir
# echo  "mkdir  -p ${RELEASE_APP}
mkdir  -p ${RELEASE_APP}
mkdir  -p ${RELEASE_LIB}
mkdir  -p ${RELEASE_BIN}
mkdir  -p ${RELEASE_CONF}
mkdir  -p ${RELEASE_SCRIPT}
mkdir  -p ${RELEASE_TOOL}

#echo -e "\033[0;36m copy em child process etc config \033[0m"
if [ -d "$curpath/config" ]
then
    cp -rf "$curpath/config" "$curpath/$SRC_DIR/$PLATFORM"
else
    echo -e "\033[1;31m ERROR: process etc config not found \033[0m"
    exit
fi

# copy src bin lib conf script
# cp -rf "$curpath/$SRC_BIN"  "$curpath/$RELEASE_DIR"

filter_list=${filter[*]}
src_bin_list=`ls $curpath/$SRC_BIN`
for i in $src_bin_list
do
    for j in $filter_list
    do
        if [ "$i" == "$j" ]
        then
            cp $curpath/$SRC_BIN/$i $curpath/$RELEASE_BIN
            break
        fi
    done
done

cp -rf "$curpath/$SRC_LIB"  "$curpath/$RELEASE_DIR"
cp -rf "$curpath/$SRC_CONF"  "$curpath/$RELEASE_DIR"
cp -rf "$curpath/$SRC_SCRIPT"  "$curpath/$RELEASE_DIR"
cp -rf "$curpath/$SRC_TOOL"  "$curpath/$RELEASE_DIR"
cp -f "$curpath/version.json" "$curpath/$RELEASE_DIR"
chmod +x $curpath/$RELEASE_SCRIPT/*
chmod +x $curpath/$RELEASE_BIN/*

# create runtime_service

#create algorithm etc config
create_algorithm_etc

# phm_server
create_proc_image phm_server
copy_proc_conf phm_server -d "$curpath/service/phm_server/conf"

# diag_server
create_proc_image diag_server
copy_proc_conf diag_server -d "$curpath/middleware/diag/conf/$PLATFORM"

# update_manager
create_proc_image update_manager
copy_proc_conf update_manager -d "$curpath/service/update_manager/conf/$PLATFORM"

# config_server
create_proc_image config_server
copy_proc_conf config_server -d "$curpath/service/config_server/conf/"

# state_machine
create_proc_image state_machine
copy_proc_conf state_machine -d "$curpath/service/state_machine/conf"

# log_server
create_proc_image log_server
copy_proc_conf log_server -d "$curpath/service/log_server/conf/$PLATFORM"

# devm_server
create_proc_image devm_server
copy_proc_conf devm_server -d "$curpath/service/devm_server/conf"

# sys_statemgr
if [[ $(echo $PLATFORM | grep "orin") != "" ]];then
    create_proc_image sys_statemgr
    copy_proc_conf sys_statemgr -d "$curpath/service/sys_statemgr/conf"
fi

# camera_venc
if [[ $(echo $PLATFORM | grep "orin") != "" ]];then
    create_proc_image camera_venc
    copy_proc_conf camera_venc -f "$curpath/service/camera_venc/conf/$PLATFORM/camera_venc_conf.yaml"
fi

# nvs_producer
if [[ $(echo $PLATFORM | grep "orin") != "" ]];then
    create_proc_image nvs_producer
    copy_proc_conf nvs_producer -f "$curpath/middleware/sensor/nvs_producer/conf/nvs_producer.yaml"
fi

# network_capture
create_proc_image network_capture
copy_proc_conf network_capture -d "$curpath/service/network_capture/conf"

# data_collection
create_proc_image data_collection
copy_proc_conf data_collection -d "$curpath/service/data_collection/conf"

# dc_mcu
create_proc_image dc_mcu

# remote_config_server
create_proc_image remote_config

# crypto_server
create_proc_image crypto_server
copy_proc_conf crypto_server -d "$curpath/middleware/crypto/crypto_server/conf"
copy_proc_etc crypto_server -d "$curpath/middleware/crypto/crypto_server/etc"

# pki_service
create_proc_image pki_service
copy_proc_conf pki_service -d "$curpath/service/pki_service/conf"
copy_proc_etc pki_service -d "$curpath/service/pki_service/etc"

# neta_someipd
create_proc_image neta_someipd
copy_proc_conf neta_someipd -d "$curpath/middleware/someip/conf"
# neta_lidar
create_proc_image neta_lidar
copy_proc_conf neta_lidar -d "$curpath//service/ethstack/lidar/conf"

# soc_to_hmi
create_proc_image soc_to_hmi
copy_proc_conf soc_to_hmi -d "$curpath/service/ethstack/soc_to_hmi/server/conf"

# sensor_trans
create_proc_image sensor_trans
copy_proc_conf sensor_trans -d "$curpath/service/ethstack/sensor_trans/server/conf/"

create_proc_image adf-lite-process
copy_proc_conf adf-lite-process -d "$curpath/test/sample/adf_lite_sample/conf"
mkdir -p $curpath/$RELEASE_APP/adf-lite-process/lib
cp -rf $SRC_DIR/$PLATFORM/test/emproc_adf_test/adf-lite-sample/lib/* "$curpath/$RELEASE_APP/adf-lite-process/lib"

# hz_dvr
# create_proc_image hz_dvr
# copy_proc_conf hz_dvr -d "$curpath/service/hz_dvr/conf"

# hz_time
create_proc_image hz_time
copy_proc_conf hz_time -d "$curpath/service/hz_time/conf"

create_proc_image ihbc_tsr_tlr
copy_proc_conf ihbc_tsr_tlr -d "$curpath/middleware/fsm/ihbc_tsr_tlr/conf"

# extwdg
copy_proc_conf extwdg -d "$curpath/service/extwdg/conf/"

sync
echo -e "\033[1;32m Create the release image success \033[0m"
