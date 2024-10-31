#!/usr/bin/env bash
set -e
set -o pipefail
clear
# 解析workspace
WORKSPACE=$(builtin cd $(dirname ${BASH_SOURCE[0]}); pwd)
cd $WORKSPACE
# 设置python解释器
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(which python3 || which python || true)
fi

export LD_LIBRARY_PATH=${WORKSPACE}/third_party/third_party/x86/protobuf/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORKSPACE
$PYTHON_BIN tools/compile.py $@ --workspace $WORKSPACE
