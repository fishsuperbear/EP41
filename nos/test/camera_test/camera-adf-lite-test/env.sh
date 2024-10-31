#!/bin/bash

prjpath=${TOOL_ROOT_PATH}/../..
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TOOL_ROOT_PATH}/lib:${prjpath}/netaos_thirdparty/x86_2004/zmq/lib/:${prjpath}/netaos_thirdparty/x86_2004/cuda/targets/x86_64-linux/lib/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TOOL_ROOT_PATH}/lib:${prjpath}/netaos_thirdparty/x86_2004/zmq/lib/:${prjpath}/netaos_thirdparty/x86_2004/cuda/targets/x86_64-linux/lib/
echo "export ADFLITE_ROOT_PATH=${prjpath}/output/x86_2004/test/emproc_adf_test/adf-lite-sample"
export ADFLITE_ROOT_PATH=${prjpath}/output/x86_2004/test/emproc_adf_test/adf-lite-sample