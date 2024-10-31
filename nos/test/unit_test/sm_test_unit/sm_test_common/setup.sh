#!/bin/bash
app_names=(`echo $1`)
ROOT_PATH=`realpath ../../`

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

echo "=== start execution_manager ==="
export EM_DEV_APP_DIR=$ROOT_PATH/test/emproc_test
export EM_DEV_CONF_DIR=/app/conf/
export LD_LIBRARY_PATH=/app/lib:/usr/lib:/lib64:../lib:$LD_LIBRARY_PATH

echo "=== $EM_DEV_APP_DIR ==="
echo "=== $EM_DEV_CONF_DIR ==="
echo "=== $LD_LIBRARY_PATH ==="
if [ -d "$ROOT_PATH/test/emproc_test" ];then
    rm -rf $ROOT_PATH/test/emproc_test
fi
mkdir -p $ROOT_PATH/test/emproc_test

cd ../../sm_test_common/emproc

for i in ${app_names[*]}; do
    echo $i $EM_DEV_APP_DIR
    cp -r $i $EM_DEV_APP_DIR/
done

sync

cd -

