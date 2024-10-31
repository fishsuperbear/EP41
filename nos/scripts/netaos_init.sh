#!/bin/bash

set -e

mount -o remount,rw /
mount -o remount,rw /app

ROOTUID="0"
if [ "$(id -u)" -ne "$ROOTUID" ] ; then
    echo -e "\033[1;31m This script must be executed as root privileges.\033[0m"
    exit 1
fi

PLATFORM=orin
if [ "$PLATFORM" = "orin" ];then
    if [ ! -f "/etc/ld.so.conf.d/neta_platform.conf" ];then
        touch /etc/ld.so.conf.d/neta_platform.conf
        chmod 644 /etc/ld.so.conf.d/neta_platform.conf
    fi
    echo "/app/lib" > /etc/ld.so.conf.d/neta_platform.conf
    echo "/usr/lib" >> /etc/ld.so.conf.d/neta_platform.conf
    echo "/lib64" >> /etc/ld.so.conf.d/neta_platform.conf
    echo "/app/conf/bag/lib" >> /etc/ld.so.conf.d/neta_platform.conf
    echo "/svp/lib" >> /etc/ld.so.conf.d/neta_platform.conf
fi

if [ "$PLATFORM" = "orin" ];then
    if [ -f "/app/conf/nvsciipc.cfg" ];then
        cp -f /app/conf/nvsciipc.cfg /etc/
        cp  /app/scripts/orin_cmd_history.sh /etc/profile.d/
    else
        echo "/app/conf/nvsciipc.cfg is not exist!!"
    fi
fi

if [ -f /etc/bash.bashrc ]; then
    if [ -z "$(cat /etc/bash.bashrc | grep '/app/scripts/env_setup.sh')" ];then
        echo "source /app/scripts/env_setup.sh" >> /etc/bash.bashrc
        ln -sf /etc/bash.bashrc ~/.bashrc
    fi
fi

NODE_LINK="/usr/sbin/nos"
NODE="/app/scripts/nos_tool.sh"
if [ ! -e $NODE_LINK ]; then
    ln -sf $NODE $NODE_LINK
    if [ $? -ne 0 ]; then
        echo "create link: $NODE failed"
    fi
fi

