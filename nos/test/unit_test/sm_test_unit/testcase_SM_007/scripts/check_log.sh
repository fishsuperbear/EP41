#!/bin/bash


expect_keyword=$1
expect_num=$2

#log生成在/log目录中
count=`sed -n "/\[BB\].*$expect_keyword/p" /log/proB_*.log|wc -l`

echo "==expect_keyword is [$expect_keyword], expect_num is [$expect_num], sed count is [$count] =="
if [ $count -eq $expect_num ]; then
    echo 0
else
    echo 1
fi
