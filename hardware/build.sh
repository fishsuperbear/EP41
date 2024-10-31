#!/bin/bash

BUILD_TARGET=$1
BOARD_CONFIG_NVIDIA=$2
current_path=$(pwd)

mkdir -p $current_path/out/
mkdir -p $current_path/out/build

mkdir -p $current_path/out/hardware

cd $current_path/out/build

if [ "$BUILD_TARGET" = "board" ]; then
    if [ "$BOARD_CONFIG_NVIDIA" = "nvidia" ]; then
cmake -DCMAKE_INSTALL_PREFIX=$current_path/out/hardware ../..  -DBUILD_ON_DOCKER=OFF -DBOARD_CONFIG_NVIDIA=ON
    else
cmake -DCMAKE_INSTALL_PREFIX=$current_path/out/hardware ../..  -DBUILD_ON_DOCKER=OFF
    fi
else
    if [ "$BOARD_CONFIG_NVIDIA" = "nvidia" ]; then
cmake -DCMAKE_INSTALL_PREFIX=$current_path/out/hardware ../.. -DBOARD_CONFIG_NVIDIA=ON
    else
cmake -DCMAKE_INSTALL_PREFIX=$current_path/out/hardware ../..
    fi
fi

make -j8

make install
