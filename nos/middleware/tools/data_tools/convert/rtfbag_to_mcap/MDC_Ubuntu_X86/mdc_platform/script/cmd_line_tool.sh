#!/bin/sh

# 启动命令行工具的代理进程，代理进程用于中转命令行工具与应用程序的通信消息
CLT_ROOT_PATH=/opt/platform/mdc_platform/manual_service/command_line_tool
${CLT_ROOT_PATH}/bin/clt_agent &
echo " start clt agent, ret $?."

exit 0
