#!/bin/bash

APP_NAME=$1
NUM=$2
if [ $# -gt 3 ]
then
    return 1
fi
CURR_PATH=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
rm -rf "${CURR_PATH}"/gcda
rm -rf "${CURR_PATH}"/*.xml
rm -rf "${CURR_PATH}"/*.log
export GCOV_PREFIX=${CURR_PATH}/gcda;
export GCOV_PREFIX_STRIP=$NUM;
export GTEST_OUTPUT=xml:${CURR_PATH}/${APP_NAME}.xml

"${CURR_PATH}"/bin/"${APP_NAME}";
