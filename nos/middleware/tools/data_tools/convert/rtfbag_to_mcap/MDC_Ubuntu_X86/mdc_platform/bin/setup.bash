#!/bin/sh
# ===============================================================================
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
# Description: rtftoolsbash - functions to support VRTF_RTFTOOLS users
# Create: 2020-03-05
# ===============================================================================
# source rtftools_setup.sh from the same directory as this file
CUR_DIR=$(builtin cd $(dirname "${BASH_SOURCE[0]}") > /dev/null && pwd)
if [ ! -f "${CUR_DIR}/rtfevent" ]
then
    echo "[WARN]: rtfevent not found and completion init may failed"
fi
if [ ! -f "${CUR_DIR}/rtfbag" ]
then
    echo "[WARN]: rtfbag not found and completion init may failed"
fi
if [ ! -f "${CUR_DIR}/rtfnode" ]
then
    echo "[WARN]: rtfnode not found and completion init may failed"
fi
if [ ! -f "${CUR_DIR}/rtfmethod" ]
then
    echo "[WARN]: rtfmethod not found and completion init may failed"
fi
export PATH=${CUR_DIR}:${PATH}
echo "[INFO]: set PATH down"
if [ ! -f "${CUR_DIR}/setup_rtftools.sh" ]
then
    echo "[ERROR]: setup_rtftools.sh not found and completion init failed"
else
    source "${CUR_DIR}/setup_rtftools.sh"
    echo "[INFO]: source setup_rtftools.sh"
    echo "[INFO]: bash completion init down"
fi
