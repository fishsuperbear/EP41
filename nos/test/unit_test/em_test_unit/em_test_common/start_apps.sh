#!/bin/bash
app_names=(`echo $1`)

/app/bin/execution_manager &
sleep 15

em_status=`ps -ef |grep execution_manager|grep -v grep|wc -l`
if [ $em_status -ne 1 ]
then
    echo "em_status is $em_status, Error:execution_manager not launched"
else
    echo "em_status is $em_status, execution_manager launched"
fi

for i in ${app_names[*]}; do
    app_status=`ps -ef |grep $i|grep -v grep|grep -v common|wc -l`
    echo "process [$i] count is [$app_status]"
done