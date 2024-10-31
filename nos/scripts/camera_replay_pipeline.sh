original_producer="nvs_producer"
consumers=("camera_venc" "fisheye_perception" "perception")  # fisheye_perception must precede perception
# consumers=("camera_venc" "desay_nvs_recv")

# define association between executable and process name in execution manager.
declare -A consumer_exe_procname_map
consumer_exe_map["camera_venc"]="/app/runtime_service/camera_venc/bin/camera_venc"
consumer_exe_map["fisheye_perception"]="/app/runtime_service/fisheye_perception/bin/fisheye_perception"
consumer_exe_map["perception"]="/app/runtime_service/perception/bin/perception"
consumer_exe_map["desay_nvs_recv"]="/app/bin/desay_nvs_recv"

# query em status
processes_em_status=""
em_pid="$(ps aux |grep execution_manager |grep -v grep |awk '{print $2}')"
if [[ -n $em_pid ]]
then
    processes_em_status="$(neta-dbg query processStatus)"
fi

function is_process_exist() {
    local pids="$(ps aux |grep $1 |grep -v grep |awk '{print $2}')"
    if [[ -n $pids ]]
    then
        return 0
    fi

    return 1
}

function kill_process() {
    local pids=$(ps aux |grep $1 |grep -v grep |awk '{print $2}')

    for pid in ${pids}
    do
        echo "kill pid: $pid, $1"
        kill -9 ${pid}
    done
}

function start_process() {
    local em_status=$(echo "$processes_em_status" | grep " $1" |grep -v grep)
    local is_started_by_em=false
    if [[ -n $em_status ]]
    then
        is_started_by_em=true
    fi

    if [ $is_started_by_em ]
    then
        em_start_status="$(neta-dbg restart $1Process)"
        if [[ $em_start_status =~ "failed" ]]
        then
            # echo "Start $1 by em failed."
            echo 1
        fi
    else
        if [[ -n "${consumer_exe_map[${1}]}" ]]
        then
            if [ -f "${consumer_exe_map[${1}]}" ]
            then
                ${consumer_exe_map[${1}]} &
            else
                # echo "Excutable file does not exist:  ${consumer_exe_map[${1}]}"
                echo 2
            fi
        else
            # echo "Executable path is not defined: ${1}"
            echo 3
        fi
    fi

    return 0
}

# kill phm_server to avoid [process restart].
$(kill_process "phm_server")

# kill original producer
$(kill_process $original_producer)

# kill all consumers
for consumer in ${consumers}
do
    $(kill_process $consumer)
done

# check original producer is killed
while true
do
    pids="$(ps aux |grep $original_producer |grep -v grep |awk '{print $2}')"
    if [ -z $pids ]
    then
        echo "$original_producer is killed. Continue to play camera"
        break
    fi
    
    sleep 0.5
done

# start phm_server if phm_server existed before.
if [ $phm_server_exist ]
then
    if [ $(start_process "phm_server") -ne 0 ]
    then
        echo "Can not start phm_server."
        exit 1;
    fi
fi

# start consumers
for consumer in ${consumers}
do
    start_proc_res=$(start_process $consumer)
    # echo "start_proc_res: ${start_proc_res}"
    if [ $start_proc_res -ne 0 ]
    then
        echo "Can not start $consumer"
        exit 2
    fi
done