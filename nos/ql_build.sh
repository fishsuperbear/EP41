#!/bin/bash

set -e

git submodule update --init

./build.sh -p orin -n 20
./build.sh -p x86_2004 -n 20