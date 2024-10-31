#!/bin/bash

# need execute by root
if [ "$USER" != "root" ]; then
    echo -e "\e[91merror: need execute by root\e[0m"
    exit 1
fi

# decide outut directory according to platform
folder_name="perfdata-`date +%g-%m-%d_%H.%M.%S`"
if [ "$(uname -p)" = "aarch64" ]; then
    folder_name="/opt/usr/data/perf/"$folder_name
fi

function Prepare() {
    mkdir -p $folder_name
    if [ ! -d "$folder_name" ]; then
        echo -e "\e[91mfolder $folder_name does not exists\e[0m"
        exit 1
    fi
    echo "perf data folder: "$folder_name
    trap 'kill $(jobs -pr)' EXIT
}

function Stat() {
    mpstat -P ALL 1 > $folder_name/mpstat.log &

    if [ $1 = "background" ]; then
        pidstat -uhRwr 1 > $folder_name/pidstat.log &
    else
        pidstat -uhRwr 1 > $folder_name/pidstat.log 
    fi
}

function Perf() {

    local pids=""
    local commands=""
    local time=10
    while getopts ":p:e:t:" opt
    do
        case $opt in
            p)
                pids="$pids $OPTARG"
                ;;
            e)
                commands="$commands $OPTARG"
                ;;
            t)
                time=$OPTARG
                ;;
            ?)
                Usage
                exit 1
                ;;
        esac
    done

    local perf_names=""

    # get all pids
    for arg in $commands
    do
        local tmp=`pidof $arg`
        if [ "$tmp" == "" ]; then
            echo -e "\e[91merror: process "$arg" is not running.\e[0m"
        else
            pids="$pids $tmp"
        fi
    done

    if [ -z "$pids" ]; then
        echo -e "\e[91merror: no process is specified.\e[0m"
        Usage
        exit 1
    fi

    # perf each pid
    local perf_data_files=""
    local perf_svg_files=""
    for pid in $pids
    do
        local cmd_name=`ps -p $pid -o comm=`
        if [ -z "$cmd_name" ]; then
            echo -e "\e[91merror: process of pid $pid is not running\e[0m"
            exit 1
        fi
        local tmp_perf_name="${cmd_name}_${pid}"
        perf_names=$perf_names" "$tmp_perf_name
        # echo "perf record -F 1000 -g --call-graph fp -p $pid -o $folder_name/$tmp_perf_name.data -- sleep $time"
        echo "start to capture perf data of $tmp_perf_name for "$time" seconds into: $folder_name/$tmp_perf_name.data"
        perf record -F 1000 -g --call-graph fp -p $pid -o $folder_name/$tmp_perf_name.data -- sleep $time &
    done

    sleep 1
    echo -ne "\e[92mcapturing, please wait for "$time" seconds...\e[0m"
    # wait perf
    for ((i=$time-1; i>0; i--))
    do
        echo -ne "\r\e[92mcapturing, please wait for "$i" seconds...\e[0m"
        sleep 1
    done
    echo ""

    sleep 10

    # convert to flame graph
    echo "convert perf data to flame graph"
    for name in $perf_names
    do
        echo "$folder_name/$name.data : $folder_name/$name.svg"
        perf script -i  $folder_name/$name.data | /app/tools/FlameGraph/stackcollapse-perf.pl | /app/tools/FlameGraph/flamegraph.pl > $folder_name/$name.svg
    done

    echo -e "\e[92mperf finished.\e[0m"
    echo "To check report by command line:"
    for name in $perf_names
    do
        echo "    perf report -i $folder_name/$name.data"
    done

    echo "To check report by flame graph:"
    for name in $perf_names
    do
        echo "    copy to x86 and open svn on browser: $folder_name/$name.svg"
    done
}

function Sched() {
    local sleep_time=""
    if [ $1 ]; then
        sleep_time=$1
    else
        sleep_time="1"
        
    fi

    echo "perf for $sleep_time(sec)"
    perf sched record -o $folder_name/sched.data -- sleep $sleep_time
    perf sched timehist -i $folder_name/sched.data > $folder_name/timehist.perf
    perf sched latency -i $folder_name/sched.data > $folder_name/latency.perf
}

function Usage() {
    echo "Usage:"
    echo "        version: v1.5"
    echo "        [nos] perfhelper stat : collect system status"
    echo "        [nos] perfhelper perf [-p <pid>] [-e <executable name>] : perf specific processes, separated by space"
    echo "        [nos] perfhelper sched [time] : perf scheduler, default 1s"
}

case $1 in 
    "stat"):
        Prepare
        Stat foreground
        ;;
    "perf"):
        Prepare
        Stat background
        shift
        Perf $*
        ;;
    "sched"):
        Prepare
        Stat background
        shift
        Sched $*
        ;;
    *):
        Usage
esac
