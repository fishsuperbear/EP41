#!/bin/bash

ProcArray=(UpdateService)
ProcPath="/svp/bin/"
KillPath="/usr/bin/"

echo "stop procs"
for proc in ${ProcArray[@]}; do
    BinPath=$ProcPath$proc
    echo $BinPath
    for pid in $(ps -ef | grep $BinPath | grep -v grep | awk '{print $2}'); do
        ${KillPath}kill -9 $pid
    done
done

cd /svp/etc/
. ./env_setup.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/svp/lib:/app/lib:/svp/lib/hozon_lib:

cd /svp/bin
/svp/bin/UpdateService &

exit
