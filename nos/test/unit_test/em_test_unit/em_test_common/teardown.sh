#!/bin/bash
app_names=(`echo $1`)

if [ -f /app/bin/killall ]; then
    /app/bin/killall -15 execution_manager
    sleep 10
    for i in ${app_names[*]}; do
        /app/bin/killall -9 $i
    done
else
    killall -15 execution_manager
    sleep 10
    for i in ${app_names[*]}; do
        killall -9 $i
    done
fi