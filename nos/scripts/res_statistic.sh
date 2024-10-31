#!/bin/sh

export PATH=$PATH:/usr/bin/busybox/
log_version="1.0"

# the script is called by fault manager by default.
monitor_type=fault
log_dir=/log/col/fm
emmc_log_file=/tmp/plog/emmc.log
dev_statinfo_log_file=/tmp/plog/dev_statinfo.log
system_runtime_file=/opt/usr/col/system_runtime/system_running_time.txt
emmc_life_file=/opt/usr/col/log/emmc_life/emmc_life.txt

# set dir path for fm and longterm type
if [ "$1" = "longterm" ]
then
    log_dir=/opt/usr/col/log/plog
    monitor_type="longterm"
fi

# echo "monitor_type: $monitor_type"
# echo $@
fault_id=$2
fault_obj=$3
collect_level=$4
is_fault_trigger=$5

# make sure log dir path
if [ ! -d $log_dir ]
then
    mkdir -p $log_dir
    chown -R mdc:mdc $log_dir
fi

# output system monitor log once.
# paramter: count, log_file
function log_output()
{
    local cnt=$1
    local log_file=$2
    # echo "-------------------- timestamp count=${cnt} --------------------" >> $log_file
    # mdc-tool tsync DP_AND_MP -g >> $log_file
    if [ -f $emmc_log_file ]
    then
        echo "-------------------- emmc count=${count} --------------------" >> $log_file
        echo "$(cat $emmc_log_file)" >> $log_file
    fi
    if [ -f $dev_statinfo_log_file ]
    then
        echo "-------------------- dev_statinfo count=${count} --------------------" >> $log_file
        echo "$(cat $dev_statinfo_log_file)" >> $log_file
    fi
    echo "-------------------- df count=${count} --------------------" >> $log_file
    df -h >> $log_file
    echo "-------------------- free count=${count} --------------------" >> $log_file
    free >> $log_file
    echo "-------------------- mpstat count=${count} --------------------" >> $log_file
    mpstat -P ALL >> $log_file
    echo "-------------------- top count=${count} --------------------" >> $log_file
    top_out="$(/usr/bin/top -b -N 60 -n 3 -d 1 -1 -w 300)"; top_out2="${top_out##*top -}"; echo "top -${top_out2}" >> $log_file
    echo "-------------------- ifconfig count=${count} --------------------" >> $log_file
    ifconfig >> $log_file
    echo "-------------------- buddyinfo count=${count} --------------------" >> $log_file
    echo "$(cat /proc/buddyinfo)" >> $log_file
    if [ -f $system_runtime_file ]
    then
        echo "-------------------- system runtime count=${count} --------------------" >> $log_file
        echo "$(cat $system_runtime_file)" >> $log_file
    fi
    if [ -f $emmc_life_file ]
    then
        echo "-------------------- emmc life count=${count} --------------------" >> $log_file
        echo "$(cat $emmc_life_file)" >> $log_file
    fi
}

# monitor system performance long term
function longterm_monitor()
{
    #echo "longterm_monitor start"
    #echo "log_dir: $log_dir"

    # format log file name with timestamp.
    local start_unixtime=$(mdc-tool tsync MP -g |grep MP | awk '{print $2}')
    local start_utctime=$(date -d @$start_unixtime +"%Y%m%d-%H%M%S")

    local sys_monitor_log=$log_dir/sys_monitor-$start_utctime.log

    # move all existing files into "old" directory.
    if [ ! -d $log_dir/old ]
    then
        mkdir -p $log_dir/old
        chown -R mdc:mdc $log_dir/old
    fi
    mv $log_dir/*.log $log_dir/old

    # delete oldest files to keep only the latest 5 files.
    file_list=($(ls --sort=time $log_dir/old)); file_count=${#file_list[*]};
    while (( $file_count > 5 ))
    do
        rm $log_dir/old/${file_list[$file_count-1]}
        let file_count--
    done

    # output log_version as file header.
    echo "log_version: $log_version" >> $sys_monitor_log

    #echo "sys_monitor_log: $sys_monitor_log"
    # output log per 5 minites
    local count=0
    while true
    do
        let count++
        log_output $count, $sys_monitor_log
        sleep 300
    done
    echo "longterm_monitor stop"
}

# monitor system performance by trigger of trigger.
function fault_monitor()
{
    local res_statistic_file_tmp=$log_dir/res_statistic_tmp.txt
    local res_statistic_file=$log_dir/res_statistic.txt

    # output log_version as file header.
    echo "log_version: $log_version" >> $res_statistic_file_tmp
    echo "is_fault_trigger: $is_fault_trigger | fault info: $fault_id $fault_obj | collect_level: $collect_level" >> $res_statistic_file_tmp

    local count=0

    if [ -e ${res_statistic_file} ];then
        rm -f ${res_statistic_file}
    fi

    while (( $count < 3))
    do
        let count++
        sleep 3
        log_output $count, $res_statistic_file_tmp
    done
    mv -f ${res_statistic_file_tmp} ${res_statistic_file}
}

# start monitor
case $monitor_type in
    "fault")
        fault_monitor
    ;;
    "longterm")
        longterm_monitor
    ;;
    *)
        echo "monitor type error!"
esac
