#!/bin/bash

app_names=("em_proc_a")

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
