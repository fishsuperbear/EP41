#!/bin/bash

# producer connect two consumer, cuda index is 0 and enc index is 1.
# this shell will launch produer and enc consumer.
# bash scripts/start_camera.sh -m 2 -e

# start single consumer. cuda or enc need set index is 0.
# this shell only launch produer.
# bash scripts/start_camera.sh

consumer_num=1
with_enc=false
dump_265=""

function kill_process() {
    local pids=$(ps aux |grep $1 |grep -v grep |awk '{print $2}')

    for pid in ${pids}
    do
        echo "kill pid: $pid, $1"
        kill -9 ${pid}
    done
    sleep 1
}

while getopts ':m:akw' OPT &> /dev/null;do
    case $OPT in
    m)
    consumer_num=$OPTARG ;;
    a)
    nos dbg restart nvs_producerProcess
    nos dbg restart camera_vencProcess
    nos dbg restart fisheye_perceptionProcess
    nos dbg restart perceptionProcess
	exit 7 ;;
    k)
    echo "kill all process."
    kill_process camera_venc
    kill_process nvs_producer
    kill_process fisheye_perception
    kill_process perception
    exit 7 ;;
    w)
    dump_265="-w" ;;
    *)
    echo "Wrong Options"
    exit 7 ;;
    esac
done

cur_dir=$(pwd)

if [[ ${BASH_SOURCE} == /* ]]; then
    shell_root="${BASH_SOURCE}"
else
    shell_root="${cur_dir}/${BASH_SOURCE}"
fi

shell_root="${shell_root%/scripts/start_camera.sh}"

source ${shell_root}/scripts/env_setup.sh
echo "Restart Camera Process."
nos dbg restart nvs_producerProcess
nos dbg restart camera_vencProcess
