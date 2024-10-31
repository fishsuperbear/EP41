#!/bin/bash
START_LOG=/opt/log/aos_linux/os_log/start.log

function show_start_log()
{
    local add_info="[${BASH_SOURCE[0]}:${FUNCNAME[1]}]"
    echo "[$(date +%Y-%m-%d" "%T)] ${add_info} $1" | tee -a "${START_LOG}"
    return 0
}

function create_mdc_path()
{
    local_path=$1
    username=$2
    usergroup=$3
    mod=$4
    mkdir -p "${local_path}"
    chown "${username}":"${usergroup}" "${local_path}"
    chmod "${mod}" "${local_path}"
}

function init_mdc_log_directory()
{
    echo "start to create mdc log directory "
    create_mdc_path "/opt/log/aos_linux/platform_log" root root 753
    create_mdc_path "/opt/log/aos_linux/os_log/" root root 753
}

function create_start_dir()
{
    init_mdc_log_directory

    create_mdc_path "/opt/cfg" root root 755

    create_mdc_path "/opt/cfg/conf/" root root 755
    create_mdc_path "/opt/cfg/conf/keyStorage" mdc mdc 700
    create_mdc_path "/opt/cfg/conf/certStorage/" mdc mdc 700

    create_mdc_path "/opt/cfg/conf_run" root root 755
    create_mdc_path "/opt/cfg/conf_run/sm" mdc mdc 740
    create_mdc_path "/opt/cfg/conf_run/cfg" mdc mdc 755
    create_mdc_path "/opt/cfg/conf_run/viz" mdc mdc 740
    create_mdc_path "/opt/cfg/conf_run/log" mdc mdc 700

    create_mdc_path "/opt/cfg/bak" root root 755
    create_mdc_path "/opt/cfg/bak/cfg" mdc mdc 755
    
    create_mdc_path "/opt/usr/" root root 755
    create_mdc_path "/opt/usr/mnt" root root 755
    create_mdc_path "/opt/usr/app" root root 755

    return 0
}

function check_process_running()
{
    PROCESS_NAME=$1
    PROC_NUM=$(pgrep -f "${PROCESS_NAME}" | wc -l)

    if [ "${PROC_NUM}" -gt 0 ];then
        echo "$PROCESS_NAME is already running..."
        return 0
    fi

    return 1
}

function start_platform_process()
{
    echo "======== Start mdc process ========"

    local PLATFORM_SCRIPT_PATH=/opt/platform/mdc_platform/script
    local LOG_DAEMON_PATH=/usr/bin/mdc/base-plat/log_daemon
    local LOG_DEBUG_PATH=/opt/log/aos_linux/platform_log

    create_start_dir

    # mdc用户进程使用cap后无法生产coredump文件,进行处理
    echo 1 > /proc/sys/fs/suid_dumpable

    # clean log
    echo "" > "$START_LOG"

    if [ ! "$(check_process_running log_daemon)" ]; then
        su -c "$LOG_DAEMON_PATH/bin/log_daemon -c $LOG_DAEMON_PATH/conf/log_daemon.conf > $LOG_DEBUG_PATH/log_daemon.log &" - mdc
        if [ $? -ne 0 ]; then
            show_start_log "ERROR: failed to execute log_daemon."
        else
            show_start_log "INFO: success to execute log_daemon."
        fi
    fi

    check_process_running someipd
    [ -e $PLATFORM_SCRIPT_PATH/vsomeipd.sh ] && su -c "$PLATFORM_SCRIPT_PATH/vsomeipd.sh" - mdc &
    if [ $? -ne 0 ]; then
        show_start_log "ERROR: failed to execute vsomeipd.sh."
    else
        show_start_log "INFO: success to execute vsomeipd.sh."
    fi

    # start process maintaind for rtf maintain tool,
    # maintaind must be started befor applications usging CM,
    # otherwise rtf maintain tool will not work properly.
    MAINTAIND_SCRIPT=$PLATFORM_SCRIPT_PATH/maintaind.sh
    [ -e $MAINTAIND_SCRIPT ] && $MAINTAIND_SCRIPT
    if [ $? -ne 0 ]; then
        show_start_log "ERROR: failed to execute maintaind.sh."
    else
        show_start_log "INFO: success to execute maintaind.sh."
    fi

    # 执行命令行工具脚本，进行相关操作
    target_name="cmd_line_tool.sh"
    [ -e ${PLATFORM_SCRIPT_PATH}/${target_name} ] && su -c ${PLATFORM_SCRIPT_PATH}/${target_name} - mdc &
    if [ $? -ne 0 ]; then
        show_start_log "ERROR: failed to execute ${target_name}."
    else
        show_start_log "INFO: success to execute ${target_name}."
    fi

    viz_script_name="viz_init.sh"
    [ -e $PLATFORM_SCRIPT_PATH/${viz_script_name} ] && su -c $PLATFORM_SCRIPT_PATH/${viz_script_name} - mdc &
    if [ $? -ne 0 ]; then
        show_start_log "ERROR: failed to execute ${viz_script_name}."
    else
        show_start_log "INFO: success to execute ${viz_script_name}."
    fi

    em_script_name="exec_manager.sh"
    if [ ! "$(check_process_running execution-manager)" ]; then
        [ -e $PLATFORM_SCRIPT_PATH/${em_script_name} ] && $PLATFORM_SCRIPT_PATH/${em_script_name}
        if [ $? -ne 0 ]; then
            show_start_log "ERROR: failed to execute ${em_script_name}."
        else
            show_start_log "INFO: success to execute ${em_script_name}."
        fi
    fi

    return 0
}

function del_someipd_dependency()
{
    # x86环境someipd由脚本拉起，EM拉起的进程(当前有smn smc cfg_server)删除对someipd的依赖
    file_list="$(find /opt/platform/mdc_platform/runtime_service/*/etc/MANIFEST.json)"
    for file in $file_list; do
        sed -i 's/\"SomeipdProcess.Running\"//g' $file
        if [ $? -ne 0 ]; then
            show_start_log "ERROR: failed to delete someipd dependency ${file}."
        fi
    done
}

main()
{
    sysctl -w fs.protected_regular=0
    sysctl -w fs.protected_fifos=0
    ldconfig
    del_someipd_dependency
    start_platform_process || echo "Start mdc process fail!"
}

main "$@"
exit $?
