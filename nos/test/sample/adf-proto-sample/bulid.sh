#!/bin/bash

set -e

BUILD_TYPE=release

function Build() {
    module=$1

    mkdir -p build
    mkdir -p output

    cmake $module -B build/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="X86" -DBUILD_TYPE="$BUILD_TYPE" 
    cd build/$module
    make -j8 
    make install
    cd -
}

Build .