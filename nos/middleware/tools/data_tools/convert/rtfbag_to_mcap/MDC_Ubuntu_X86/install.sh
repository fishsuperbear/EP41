#! /bin/bash

username="mdc"
usergroup="mdc"

SCRIPT_DIR=$(cd "$(dirname "$0")" || return; pwd)

function create_user_group()
{
    #判断mdc用户和用户组是否存在
    if [ "$( < /etc/group grep -c ${usergroup} )" -eq 0 ]; then
        groupadd ${usergroup}  || return 1
        echo "creat group ${usergroup}"
    else
        echo "group \"mdc\" already exists."
    fi
    if [ "$( < /etc/passwd grep -c ${username} )" -eq 0 ]; then
       useradd -m -s /bin/bash  -d /home/mdc -g ${usergroup} ${username}  || return 1
       echo "creat user ${usergroup}"
    else
        echo "user \"mdc\" already exists."
    fi
}

function chang_file_permission()
{
    echo "start change \"$1\" user and group"
    local user
    while IFS= read -r file
    do
        if [ -f "$file" ]; then
            user=$(ls -l "$file" | awk -F " " '{print $3}')
        else
            user=$(ls -ld "$file" | awk -F " " '{print $3}')
        fi
        if [ "$user" != "root" ]; then
            chown -h mdc:mdc "$file"
        fi
    done <    <( find "$1" )
    echo "finish change \"$1\" user and group"
}

function change_permission()
{
    chang_file_permission "$SCRIPT_DIR"/etc || return 1
    chang_file_permission "$SCRIPT_DIR"/home || return 1
    chang_file_permission "$SCRIPT_DIR"/mdc_platform || return 1
    chang_file_permission "$SCRIPT_DIR"/usr || return 1
}

function copy_file()
{
    echo "Start copy file."
    mkdir -p /opt/platform
    cp -af "$SCRIPT_DIR"/etc / || return 1
    cp -af "$SCRIPT_DIR"/home / || return 1
    cp -af "$SCRIPT_DIR"/usr / || return 1
    cp -af "$SCRIPT_DIR"/mdc_platform /opt/platform/ || return 1

    echo "Finish copy file."
}

function set_mdc_env()
{
    sed -i '/\/etc\/mdc_env.bash/d' /etc/profile
    sed -i '/\/opt\/platform\/mdc_platform\/bin\/setup.bash/d' /etc/profile
    echo "source /etc/mdc_env.bash" >> /etc/profile
    echo "source /opt/platform/mdc_platform/bin/setup.bash 1>/dev/null" >> /etc/profile
    return 0
}

function main()
{
    rm -rf /etc/ld.so.conf.d/mdc_platform.conf
    rm -rf /etc/mdc/base-plat/log_daemon
    rm -rf /etc/mdc_env.bash
    rm -rf /home/etc/rtf/rtf.json
    rm -rf /opt/platform/mdc_platform
    rm -rf /usr/bin/mdc/base-plat/log_daemon
    if [ "$(whoami)" != "root" ]; then
        echo "pleace use \"root\" to run the script"
    fi
    create_user_group || echo "Create user group error!"
    change_permission || echo "Change permission error!"
    copy_file || echo "Copy file error!"
    set_mdc_env || echo "Set mdc environment error!"
}

main "$@"
source /etc/profile
