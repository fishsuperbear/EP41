#!/bin/bash

export LD_LIBRARY_PATH=/home/xiaoyu/work/netaos/nos/output/x86_2004/lib/:$LD_LIBRARY_PATH


# 循环遍历执行进程
for ((i=1; i<=10; i++))
do
    # 构建进程名称
    process_name="proc_$(printf "%03d" $i)"
    
    # 执行进程
    echo "执行进程 $process_name"
    ./$process_name &
    
    # 可选：等待一段时间，再执行下一个进程
    sleep 0.01
done
