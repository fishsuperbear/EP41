#!/bin/bash

script_dir=$(cd "$(dirname "$0")"; pwd)
apConfDir=${script_dir}/../manual_service/camera_mviz/etc/CameraMvizProcess/
runDir=${script_dir}/../manual_service/camera_mviz/bin

export LD_LIBRARY_PATH=${script_dir}/../lib/:${LD_LIBRARY_PATH}
export CM_CONFIG_FILE_PATH=${apConfDir}
${runDir}/camera_mviz  $*
