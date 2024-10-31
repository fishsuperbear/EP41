#!/bin/bash

echo "kill nos poc start"

ProcArray=(hz_perception_state_machine \
           fisheye_perception \
           parking_slam \
           parking_fusion \
           perception \
           planning \
           mapping \
           uss_perception \
           phm_server \
           log_server \
           devm_server \
           update_manager \
           neta_someipd \
           crypto_server \
           camera_venc \
           data_collection \
           diag_server \
           hz_time \
           dc_mcu \
           remote_config_server \
           hz_dvr \
           network_capture \
           nvs_producer \
           neta_lidar \
           pki_service \
           state_machine \
           sensor_trans \
           soc_to_hmi)

ProcPath="/app/runtime_service/"
KillPath="/app/bin/"

${KillPath}killall -9 system_monitor
${KillPath}killall -9 remote_diag
${KillPath}killall -9 config_server
${KillPath}killall -9 sys_statemgr
${KillPath}killall -9 extwdg
${KillPath}killall -9 execution_manager
${KillPath}killall -9 notify_main
${KillPath}killall -9 core_file_monitor.sh

for proc in ${ProcArray[@]}; do
    ${KillPath}killall -9 ${proc}
done

echo "$(ps -ef | grep "/app")"

echo "kill nos poc finish"

exit