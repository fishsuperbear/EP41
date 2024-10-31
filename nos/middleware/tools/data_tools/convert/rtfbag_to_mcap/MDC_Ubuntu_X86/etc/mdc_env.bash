#!/bin/bash

export PATH=/opt/platform/mdc_platform/bin:${PATH}
export ASCEND_VECTOR_OBJ_PATH=/lib64/pg1

# add path and bash completion scripts of command line tool
CLT_ROOT_PATH=/opt/platform/mdc_platform/manual_service/command_line_tool
export PATH=${CLT_ROOT_PATH}/bin:${PATH}
source ${CLT_ROOT_PATH}/conf/clt_completion
