#!/bin/bash

ln -sf CMakeLists_board.txt CMakeLists.txt
mkdir -p build
cd build
cmake ..
make -j8
