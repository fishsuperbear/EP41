#!/bin/bash

core_pattern_file="/proc/sys/kernel/core_pattern"
core_attribute="ELF 64-bit LSB core file"
delete_log_name="core_file.log"

sleep_time=5s
reserved_file_num=20
reserved_log_line_count=1000

function monitor_core_file()
{
    if [ $# -eq 0 ];then
        return 1
    fi

    local core_file_path=$1
    local delete_file_log=${core_file_path}/${delete_log_name}
    cd "${core_file_path:?}" || { echo "ERROR: core_file_path does't exist: ${core_file_path}"; return 1; }

    local files=$(ls -t ${coredump_file_path})
    local core_file_num=0
    local delete_files_info

    for fileName in ${files}
    do
        local fileInfo=$(file "${fileName}" | grep -wE "${core_attribute}")
        if [ -n "${fileInfo}" ];then
            if [ "${core_file_num}" -lt "${reserved_file_num}" ]; then
                let core_file_num++
            else
                rm -f "${fileName}"
                local delete_file_info="[`date +%Y"/"%m"/"%d" "%H":"%M":"%S`] delete core file: ${fileName}"
                delete_files_info=${delete_file_info}'\n'${delete_files_info}
            fi
        fi
    done

    # 记录日志并控制日志文本大小
    if [ -n "${delete_files_info}" ]; then
        echo -e ${delete_files_info} >> ${delete_file_log}
    fi
    if [ -f "${delete_file_log}" ]; then
        local row_num=$(wc -l ${delete_file_log} | awk '{print $1}')
        if [ ${row_num} -gt ${reserved_log_line_count} ]; then
            local delete_files_number=$((row_num-reserved_log_line_count))
            sed -i "1,${delete_files_number}d" ${delete_file_log} &
        fi
    fi
    return 0
}

function main()
{
    # 获取linux 和 coredump文件路径
    local linux_coredump_file_path=$(cat ${core_pattern_file} | awk -F'/[^/]*$' '{print $1}')
    cd "${linux_coredump_file_path:?}" || { echo "ERROR: The linux coredump path does't exist: ${linux_coredump_file_path}"; return 1; }

    while true
    do
        monitor_core_file ${linux_coredump_file_path}

        sleep ${sleep_time}
    done

    return 0
}

main $@
exit $?
