cmake_minimum_required(VERSION 2.8.3)
project(PandarSwiftSDK)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
set(CMAKE_BUILD_TYPE "Release")

include_directories(
    .
    include
    src/taskflow
    ${3RD_LIBRARY_PATH}/boost/include
    ${3RD_LIBRARY_PATH}/openssl/include
    ${3RD_LIBRARY_PATH}/pcap/include
)

file (GLOB HESAI_SDK_SOURCE_LIST
    src/input.cc
    src/laser_ts.cpp
    src/pandarSwiftDriver.cc
    src/pandarSwiftSDK.cc
    src/platUtil.cc
    src/tcp_command_client.c
    src/util.c
    src/wrapper.cc
    src/fault_message.cc
)

add_library(PandarSwiftSDK SHARED ${HESAI_SDK_SOURCE_LIST})
target_link_libraries(PandarSwiftSDK
    ${THIRDPARTYD_PATH}/lidar/boost/lib/libboost_thread.so
    # boost_thread
    crypto
    ssl
    pcap
)

add_executable(PandarSwiftTest test/test.cc)
target_link_libraries(PandarSwiftTest
    PandarSwiftSDK
    ${PCL_LIBRARIES}
)

install(TARGETS PandarSwiftSDK
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/lidar
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/lidar
)