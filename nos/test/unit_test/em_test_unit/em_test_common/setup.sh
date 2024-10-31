#!/bin/bash
app_names=(`echo $1`)
ROOT_PATH=`realpath ../../em_test_common`
ROOT_EM_PATH=/app

if [ -f /app/bin/killall ]; then
    /app/bin/killall -9 execution_manager
    sleep 1
    for i in ${app_names[*]}; do
        /app/bin/killall -9 $i
    done
else
    killall -9 execution_manager
    sleep 1
    for i in ${app_names[*]}; do
        killall -9 $i
    done
fi

BinPath="/app/runtime_service"
for pid in $(ps -ef | grep $BinPath | grep -v grep | awk '{print $2}'); do
    echo "===>> $pid"
    /usr/bin/kill -9 $pid
done

export EM_DEV_APP_DIR=$ROOT_PATH/emproc
export EM_DEV_CONF_DIR=/app/conf
export LD_LIBRARY_PATH=/app/lib:/usr/lib:/lib64:../lib:$LD_LIBRARY_PATH

echo "=== $EM_DEV_APP_DIR ==="
echo "=== $EM_DEV_CONF_DIR ==="
echo "=== $LD_LIBRARY_PATH ==="

cd ../../em_test_common/emproc

for i in ${app_names[*]}; do
    echo $i $EM_DEV_APP_DIR
    cp -r $i $EM_DEV_APP_DIR/
done

sync

cd -

