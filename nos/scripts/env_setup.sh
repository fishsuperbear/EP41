#!/bin/bash
cur_dir=$(pwd)

if [[ ${BASH_SOURCE} == /* ]]; then
    tool_root="${BASH_SOURCE}"
else
    tool_root="${cur_dir}/${BASH_SOURCE}"
fi
tool_root="${tool_root%/scripts/env_setup.sh}"
export LD_LIBRARY_PATH="${tool_root}/lib:${tool_root}/conf/bag/MDC_Ubuntu_X86/mdc_platform/lib:/app/lib:/app/conf/bag/lib:/svp/lib:${LD_LIBRARY_PATH}"
export PATH="${tool_root}/bin:${tool_root}/scripts:/app/bin:/app/scripts:${PATH}"
export AMENT_PREFIX_PATH="${tool_root}/conf/bag"
export CONF_PREFIX_PATH="${tool_root}/conf/"
export TERMINFO=/usr/lib/terminfo
export TOOL_ROOT_PATH=${tool_root}
export SOMEIP_DESERIALIZ_JSON_PATH="${tool_root}/conf/bag/someip/"
export NOS_TOOL_PATH=/opt/usr/sbin

export NOS_ROOT=${tool_root}
alias perfhelper="bash $tool_root/scripts/perf_helper.sh"
alias nos="bash $tool_root/scripts/nos_tool.sh"

source nos_bash_complt.sh
