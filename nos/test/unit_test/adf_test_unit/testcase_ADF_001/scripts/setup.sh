#!/bin/bash
ROOT_PATH=`realpath ../../../../../`
echo "==========ROOT_PATH:${ROOT_PATH}========="
ps -ef |grep chassis | grep -v grep | awk '{print $2}' |xargs kill -9 

export LD_LIBRARY_PATH=$ROOT_PATH/lib:/usr/lib:/lib64:../lib:$LD_LIBRARY_PATH
echo "==========LD_LIBRARY_PATH:${LD_LIBRARY_PATH}============="