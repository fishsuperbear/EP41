#!/bin/sh

# 创建viz配置文件，初始化上位机地址模板
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

function generate_viz_address_conf_file()
{
    local viz_address_conf_file=/opt/cfg/conf_run/viz/viz_address.conf
    local file_auth_user=mdc
    local file_auth_pri=640

    if [ ! -f ${viz_address_conf_file} ]; then
        touch ${viz_address_conf_file}
        echo "127.0.0.1 7000" > ${viz_address_conf_file}
        chown ${file_auth_user}:${file_auth_user} ${viz_address_conf_file}
        chmod ${file_auth_pri} ${viz_address_conf_file}
    fi

    return 0
}

function main()
{
    generate_viz_address_conf_file || return 1
}

main "$@"
exit 0
