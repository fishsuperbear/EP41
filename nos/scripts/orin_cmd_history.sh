#!/bin/sh
#######################################################################################
#
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
# security-tool licensed under the Mulan PSL v1.
# You can use this software according to the terms and conditions of the Mulan PSL v1.
# You may obtain a copy of Mulan PSL v1 at:
#     http://license.coscl.org.cn/MulanPSL
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v1 for more details.
# Description: Append the history list to the history file.
#
#######################################################################################

DriveOS_history()
{
    result=$?
    log_path="/opt/usr/log/system_monitor_log/"
    if [ ! -d "${log_path}" ];then
        mkdir -p ${log_path}
        if [ ! -f "${log_path}""history.log" ];then
            touch "${log_path}""history.log"
            chmod 777 "${log_path}""history.log"
        fi
    fi

    cli_his_tmp=$(history 1)
    his_num=$(echo "$cli_his_tmp" | awk '{print $1}')
    cli_his=$(echo $cli_his_tmp | sed "s/$his_num //")
    loginer=$(who -m | awk '{print $1}')
    loginer_id=$(id -u "$loginer")
    cmd_where=$(who -m | awk '{print $2" "$NF}')
    loginer_num=$(who -q | awk -F '=' '{print $2}')
    pwd_cmd=$(pwd)
#    cmd_cnt=1
    if [ "${LastComandNum_for_history}" = "" ];then
        cmd_cnt=0
    fi

    cmd_time=$(date "+%Y-%m-%d %H:%M:%S")
    if [ "${his_num}" != "${LastComandNum_for_history}" ] && { [ "${LastComandNum_for_history}" != "" ] || [ "${his_num}" = "1" ]; };then
        if [ ${cmd_cnt} -eq 0 ] || [ ${loginer_num} -ne 1 ];then
            echo "[$his_num][$LastComandNum_for_history][$cmd_cnt][${cmd_time}][${loginer}][${loginer_id}][${cmd_where}][$cli_his][${pwd_cmd}]""[${result}]" >> "$log_path""history.log"
            cmd_cnt=1
        else
            echo "[${cmd_time}][$cli_his][${pwd_cmd}]""[${result}]" >> "$log_path""history.log"
        fi
    fi
    LastComandNum_for_history=${his_num}
}

DriveOS_variable_readonly()
{
    _local_var="$1"
    _local_val="$2"
    _local_ret=$(readonly -p | grep -w "${_local_var}" | awk -F "${_local_var}=" '{print $NF}')
    if [ "${_local_ret}" = "\"${_local_val}\"" ]
    then
        return
    else
        export "${_local_var}"="${_local_val}"
        readonly "${_local_var}"
    fi
}

export HISTCONTROL=''
DriveOS_variable_readonly HISTTIMEFORMAT ""
DriveOS_variable_readonly PROMPT_COMMAND DriveOS_history
