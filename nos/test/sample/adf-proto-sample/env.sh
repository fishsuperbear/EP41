#!/bin/bash

curpath=$(pwd)
echo "path $curpath"
prjpath=../../../../
export LD_LIBRARY_PATH=$prjpath/output/x86_2004/lib:$LD_LIBRARY_PATH:$prjpath/netaos_thirdparty/x86_2004/zmq/lib/