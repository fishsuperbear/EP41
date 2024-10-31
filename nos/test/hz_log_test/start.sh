#!/bin/bash

print_help() {
    echo -e "$0 [test sequence] [thread count] [is sleep] [time_ms] [logto] [mode]"
    echo -e "\ttest sequence: 1-6， 11-13， 21-24"
    echo -e "\tthread count: total thread number"
    echo -e "\tis sleep: whether to limit the speed of writing logs or not"
    echo -e "\ttime_ms: milliseconds, write log time"
    echo -e "\tlogto: default:2, kernel:4"
    echo -e "\tmode: spdlog/kernel"
    echo -e "Examples:"
    echo -e "\t$0 12 10 1 10 2 spdlog"
}

if [[ $# == 0 || "$1" == "-h" || "$1" == "help" ]]; then
    print_help
    exit 0
fi

export LD_LIBRARY_PATH=/opt/usr/ytx/hz/lib:$LD_LIBRARY_PATH
# 检查是否提供了整数参数
#if  ! [[ $1 =~ ^[0-9]+$ ]]; then
#    echo "请提供一个整数参数作为测试次数。"
#    exit 1
#fi

# 整数参数，用作测试次数
test_count=$1
thread_count=$2
is_sleep=$3
time_ms=$4
logto=$5
mode=$6

# 初始化数组来保存每次的平均值
avg_time_arr=()
avg_cpu_arr=()
p_avg_cpu_arr=()
avg_mem_vsz_arr=()
p_avg_mem_vsz_arr=()
avg_mem_rss_arr=()
p_avg_mem_rss_arr=()
avg_mem_mem_arr=()
p_avg_mem_mem_arr=()
avg_disk_read_arr=()
avg_disk_write_arr=()

producer_pid=""
consumer_pid=""

test_result_path="./resultdir/"

start_proc() {
    # 假设您的进程A的启动命令是 "./hz_log_performance_test $test_count"
    if [[ "$mode" == "spdlog" ]]; then
        #./hz_log_performance_test $test_count $thread_count >>./performance.txt &
        ./hz_log_performance_test $test_count $thread_count >>/dev/null &
        producer_pid=$!
    else
        #./logcollector -max_file_size=10000 >/dev/null &
        ./log_collector >/dev/null &
        consumer_pid=$!
        sleep 3 
        #./hz_log_performance_test $test_count $thread_count $is_sleep $time_ms >>./performance.txt &
        ./hz_log_performance_test $test_count $thread_count $is_sleep $time_ms $logto >>/dev/null &
        producer_pid=$!
    fi
}

stop_proc() {
    kill -9 `ps axu|grep hz_log_performance_test|grep -v grep|awk '{print $2}'`
    kill -9 `ps axu|grep log_collector|grep -v grep|awk '{print $2}'`
    sleep 5

    rm fetchlog/*
}

rm -rf resultdir/*

# 循环运行进程3次
for ((i=1; i<=3; i++))
do
    # 启动进程A，并获取其PID
    stop_proc
    start_proc

    if [[ "$producer_pid" == "" ]]; then
        echo "no producer process, exit";
        exit -1;
    fi

    if [[ "$mode" != "spdlog" ]]; then
        if [[ "$consumer_pid" == "" ]]; then
            echo "no consumer process, exit"
            exit -1;
        fi
    fi
    pidlist=$producer_pid","$consumer_pid

    # 记录开始时间
    start_time=$(date +%s.%N)

    # 等待进程A运行0.3秒
    sleep 0.3

    # 初始化变量来记录采样数据
    cpu_usage=()
    mem_vsz_usage=()
    mem_rss_usage=()
    mem_mem_usage=()

    p_cpu_usage=()
    p_mem_vsz_usage=()
    p_mem_rss_usage=()
    p_mem_mem_usage=()

    disk_read=()
    disk_write=()

    # 采样间隔
    sample_interval=0.5

    while true
    do
		
        # 检查进程A是否已经退出，如果退出则结束记录
        if ! kill -0 $producer_pid 2>/dev/null; then
            break
        fi

        # 使用top命令获取CPU消耗
        cpu_data=$(top -b -n 1 -p $pidlist | awk 'NR>7 { sum += $9; } END { print sum; }')
        p_cpu_data=$(top -b -n 1 -p $producer_pid | awk 'NR>7 { sum += $9; } END { print sum; }')
        # 使用pidstat命令获取MEM占用情况
        mem_vsz_data=$(pidstat -r -p $pidlist | awk 'NR>3 { sum += $6 / 1024; } END { print sum; }') # 转换为MB
        p_mem_vsz_data=$(pidstat -r -p $producer_pid | awk 'NR>3 { sum += $6 / 1024; } END { print sum; }') # 转换为MB
        mem_rss_data=$(pidstat -r -p $pidlist | awk 'NR>3 { sum += $7 / 1024; } END { print sum; }') # 转换为MB
        p_mem_rss_data=$(pidstat -r -p $producer_pid | awk 'NR>3 { sum += $7 / 1024; } END { print sum; }') # 转换为MB
        mem_mem_data=$(pidstat -r -p $pidlist | awk 'NR>3 { sum += $8; } END { print sum; }') # 转换为MB
        p_mem_mem_data=$(pidstat -r -p $producer_pid | awk 'NR>3 { sum += $8; } END { print sum; }') # 转换为MB

        # 使用pidstat命令获取磁盘读取和写入速率
        disk_data=$(pidstat -d -p $pidlist | awk 'NR>3 { sum_read += $4; sum_write += $5; } END { print sum_read, sum_write; }')

	    # echo "$cpu_data"
        # echo "$mem_vsz_data"
        # echo "$mem_rss_data"
        # echo "$mem_mem_data"
        # echo "$disk_data" | awk '{print $1}'
        # echo "$disk_data" | awk '{print $2}'

        # 添加数据到相应的数组
        cpu_usage+=("$cpu_data")
        p_cpu_usage+=("$p_cpu_data")
        mem_vsz_usage+=("$mem_vsz_data")
        p_mem_vsz_usage+=("$p_mem_vsz_data")
        mem_rss_usage+=("$mem_rss_data")
        p_mem_rss_usage+=("$p_mem_rss_data")
        mem_mem_usage+=("$mem_mem_data")
        p_mem_mem_usage+=("$p_mem_mem_data")

        read_speed=$(echo "$disk_data" | awk '{print $1}')
        write_speed=$(echo "$disk_data" | awk '{print $2}')
        disk_read+=("$read_speed")
        disk_write+=("$write_speed")

        # 等待下一个采样点
        sleep $sample_interval
    done

    # 记录结束时间
    end_time=$(date +%s.%N)

    # 计算运行时间
    #run_time=$((end_time - start_time))
    run_time=$(python -c "print(($end_time - $start_time) * 1)")

    # 计算平均值并保存到数组
    avg_cpu=$(echo "${cpu_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    p_avg_cpu=$(echo "${p_cpu_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    avg_mem_vsz=$(echo "${mem_vsz_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    p_avg_mem_vsz=$(echo "${p_mem_vsz_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    avg_mem_rss=$(echo "${mem_rss_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    p_avg_mem_rss=$(echo "${p_mem_rss_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    avg_mem_mem=$(echo "${mem_mem_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    p_avg_mem_mem=$(echo "${p_mem_mem_usage[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    avg_disk_read=$(echo "${disk_read[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
    avg_disk_write=$(echo "${disk_write[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')

    avg_time_arr+=("$run_time")
    avg_cpu_arr+=("$avg_cpu")
    p_avg_cpu_arr+=("$p_avg_cpu")
    avg_mem_vsz_arr+=("$avg_mem_vsz")
    p_avg_mem_vsz_arr+=("$p_avg_mem_vsz")
    avg_mem_rss_arr+=("$avg_mem_rss")
    p_avg_mem_rss_arr+=("$p_avg_mem_rss")
    avg_mem_mem_arr+=("$avg_mem_mem")
    p_avg_mem_mem_arr+=("$p_avg_mem_mem")
    avg_disk_read_arr+=("$avg_disk_read")
    avg_disk_write_arr+=("$avg_disk_write")

    # 输出结果到result.txt
	echo "" >> $test_result_path/result.txt
    echo "Producer&Consumer 运行的测试用例: $test_count" >> $test_result_path/result.txt
    echo -e "\t运行时间: $run_time 秒" >> $test_result_path/result.txt
    echo -e "\tCPU消耗(平均值): $avg_cpu %" >> $test_result_path/result.txt
    echo -e "\t虚拟内存占用(平均值): $avg_mem_vsz MB" >> $test_result_path/result.txt
    echo -e "\t物理内存占用(平均值): $avg_mem_rss MB" >> $test_result_path/result.txt
    echo -e "\t内存占比(平均值): $avg_mem_mem %" >> $test_result_path/result.txt
    echo -e "\t磁盘读取速率(平均值): $avg_disk_read KB/s" >> $test_result_path/result.txt
    echo -e "\t磁盘写入速率(平均值): $avg_disk_write KB/s" >> $test_result_path/result.txt

    # 输出进程A的各项指标
    echo "P&C 进程的各项指标:"
    echo -e "\t运行时间: $run_time 秒"
    echo -e "\tCPU消耗(平均值): $avg_cpu %"
    echo -e "\t虚拟内存占用(平均值): $avg_mem_vsz MB"
    echo -e "\t物理内存占用(平均值): $avg_mem_rss MB"
    echo -e "\t内存占比(平均值): $avg_mem_mem %"
    echo -e "\t磁盘读取速率(平均值): $avg_disk_read KB/s"
    echo -e "\t磁盘写入速率(平均值): $avg_disk_write KB/s"

    echo "测试$test_count 运行第 $i次,已完成"

    if [ "$i" -ne 3 ]; then
        # 如果$i不等于3，删除指定目录下的所有.log文件
        rm -rf /opt/usr/hz_map/hzlog/*.log
    fi

    # bak logs
    mv fetchlog $test_result_path/fetchlog$i
    mkdir -p fetchlog

    if [[ "$mode" != "spdlog" ]]; then
        echo "" >> $test_result_path/result.txt
        echo "Producer 运行的测试用例: $test_count" >> $test_result_path/result.txt
        echo -e "\t 运行时间: $run_time 秒" >> $test_result_path/result.txt
        echo -e "\tCPU消耗(平均值): $p_avg_cpu %" >> $test_result_path/result.txt
        echo -e "\t虚拟内存占用(平均值): $p_avg_mem_vsz MB" >> $test_result_path/result.txt
        echo -e "\t物理内存占用(平均值): $p_avg_mem_rss MB" >> $test_result_path/result.txt
        echo -e "\t内存占比(平均值): $p_avg_mem_mem %" >> $test_result_path/result.txt
        echo -e "\t磁盘读取速率(平均值): $p_avg_disk_read KB/s" >> $test_result_path/result.txt
        echo -e "\t磁盘写入速率(平均值): $p_avg_disk_write KB/s" >> $test_result_path/result.txt

        # 输出进程A的各项指标
        echo "Producer 进程的各项指标:"
        echo -e "\t运行时间: $run_time 秒"
        echo -e "\tCPU消耗(平均值): $p_avg_cpu %"
        echo -e "\t虚拟内存占用(平均值): $p_avg_mem_vsz MB"
        echo -e "\t物理内存占用(平均值): $p_avg_mem_rss MB"
        echo -e "\t内存占比(平均值): $p_avg_mem_mem %"
        echo -e "\t磁盘读取速率(平均值): $p_avg_disk_read KB/s"
        echo -e "\t磁盘写入速率(平均值): $p_avg_disk_write KB/s"

        echo "测试$test_count 运行第 $i次,已完成"
    fi

    # 等待一段时间，以确保资源得到释放
    sleep 5
done

# 计算3次平均值
avg_time_total=$(echo "${avg_time_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_cpu_total=$(echo "${avg_cpu_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_mem_vsz_total=$(echo "${avg_mem_vsz_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_mem_rss_total=$(echo "${avg_mem_rss_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_mem_mem_total=$(echo "${avg_mem_mem_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
p_avg_cpu_total=$(echo "${p_avg_cpu_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
p_avg_mem_vsz_total=$(echo "${p_avg_mem_vsz_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
p_avg_mem_rss_total=$(echo "${p_avg_mem_rss_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
p_avg_mem_mem_total=$(echo "${p_avg_mem_mem_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_disk_read_total=$(echo "${avg_disk_read_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')
avg_disk_write_total=$(echo "${avg_disk_write_arr[@]}" | tr ' ' '\n' | awk '{ sum += $1; } END { print sum / NR; }')

# 输出3次平均值
echo "Producer&Consumer 3次平均值:"
echo -e "\t耗时(平均值): $avg_time_total 秒"
echo -e "\tCPU消耗(平均值): $avg_cpu_total %"
echo -e "\t虚拟内存占用(平均值): $avg_mem_vsz_total MB"
echo -e "\t物理内存占用(平均值): $avg_mem_rss_total MB"
echo -e "\t内存占比(平均值): $avg_mem_mem_total %"
echo -e "\t磁盘读取速率(平均值): $avg_disk_read_total KB/s"
echo -e "\t磁盘写入速率(平均值): $avg_disk_write_total KB/s"

echo " " >> $test_result_path/result.txt
echo "3次平均值:" >> $test_result_path/result.txt 
echo -e "\t耗时(平均值): $avg_time_total 秒" >> $test_result_path/result.txt
echo -e "\tCPU消耗(平均值): $avg_cpu_total %" >> $test_result_path/result.txt
echo -e "\t虚拟内存占用(平均值): $avg_mem_vsz_total MB" >> $test_result_path/result.txt
echo -e "\t物理内存占用(平均值): $avg_mem_rss_total MB" >> $test_result_path/result.txt
echo -e "\t内存占比(平均值): $avg_mem_mem_total %" >> $test_result_path/result.txt
echo -e "\t磁盘读取速率(平均值): $avg_disk_read_total KB/s" >> $test_result_path/result.txt
echo -e "\t磁盘写入速率(平均值): $avg_disk_write_total KB/s" >> $test_result_path/result.txt

if [[ "$mode" != "spdlog" ]]; then
    # 输出3次平均值
    echo "Producer 3次平均值:"
    echo -e "\t耗时(平均值): $avg_time_total 秒"
    echo -e "\tProducer CPU消耗(平均值): $p_avg_cpu_total %"
    echo -e "\tProducer 虚拟内存占用(平均值): $p_avg_mem_vsz_total MB"
    echo -e "\tProducer 物理内存占用(平均值): $p_avg_mem_rss_total MB"
    echo -e "\tProducer 内存占比(平均值): $p_avg_mem_mem_total %"
    echo -e "\t磁盘读取速率(平均值): $avg_disk_read_total KB/s"
    echo -e "\t磁盘写入速率(平均值): $avg_disk_write_total KB/s"

    echo " " >> $test_result_path/result.txt
    echo "Producer 3次平均值:" >> $test_result_path/result.txt 
    echo -e "\t耗时(平均值): $avg_time_total 秒" >> $test_result_path/result.txt
    echo -e "\tCPU消耗(平均值): $p_avg_cpu_total %" >> $test_result_path/result.txt
    echo -e "\t虚拟内存占用(平均值): $p_avg_mem_vsz_total MB" >> $test_result_path/result.txt
    echo -e "\t物理内存占用(平均值): $p_avg_mem_rss_total MB" >> $test_result_path/result.txt
    echo -e "\t内存占比(平均值): $p_avg_mem_mem_total %" >> $test_result_path/result.txt
    echo -e "\t磁盘读取速率(平均值): $avg_disk_read_total KB/s" >> $test_result_path/result.txt
    echo -e "\t磁盘写入速率(平均值): $avg_disk_write_total KB/s" >> $test_result_path/result.txt
fi

stop_proc
echo "脚本执行完成"

if [ -f parser ]; then
    echo -e "\n\n"
    echo "================================result data information====================================="
    #./parser $test_result_path/fetchlog1/ #| grep -v "magic number don't match"
    #./parser $test_result_path/fetchlog2/ #| grep -v "magic number don't match"
    #./parser $test_result_path/fetchlog3/ #| grep -v "magic number don't match"
    #ls -l $test_result_path/fetchlog* -rht
fi

