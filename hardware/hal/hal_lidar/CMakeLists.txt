cmake_minimum_required(VERSION 3.0)
project(hal_lidar_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PCL_SHOW "Use pcl to show pointcloud" OFF)
if(PCL_SHOW)
    add_definitions(-DPCL_SHOW)
    message(STATUS "Open pcl dependence")

    find_package(PCL 1.10 REQUIRED)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|x64|x86|Win64|Win32")
    message(STATUS "x86 platform detected!")
    set(X86_PLATFORM 1)
    add_definitions(-DX86_PLATFORM)
else()
    message(STATUS "arm platform detected!")
    set(ARM_PLATFORM 1)
    add_definitions(-DARM_PLATFORM)
endif()

# 暴露给外部的头文件的路径
set(INTERFACE_HPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../code/interface/hpp/intf_lidar/1.0)
# 逻辑代码的路径
set(IMPL_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../code/hal_impl/cpp/impl_lidar)
# 三方库的路径
set(THIRDPARTYD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../code/thirdParty)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INTERFACE_HPP_PATH}
    ${IMPL_CPP_PATH}/sdk/hesai/include
    ${IMPL_CPP_PATH}/sdk/hesai/src/taskflow
    ${PCL_INCLUDE_DIRS}
    ${THIRDPARTYD_PATH}/lidar/pcap/include
    ${THIRDPARTYD_PATH}/lidar/openssl/include
    ${THIRDPARTYD_PATH}/lidar/boost/include
)

if (ARM_PLATFORM)
    link_directories(
        ${THIRDPARTYD_PATH}/lidar/pcap/lib
        ${THIRDPARTYD_PATH}/lidar/openssl/lib
        ${THIRDPARTYD_PATH}/lidar/boost/lib
    )
endif()

file(GLOB SOURCE_FILE_LIST
    ${IMPL_CPP_PATH}/*.h
    ${IMPL_CPP_PATH}/*.cpp
)

add_subdirectory(${IMPL_CPP_PATH}/sdk/hesai PandarSwiftSDK)

add_library(hal_lidar SHARED ${SOURCE_FILE_LIST})
target_link_libraries(hal_lidar
    pthread
    ${THIRDPARTYD_PATH}/lidar/boost/lib/libboost_thread.so
    # boost_thread
    crypto
    ssl
    pcap
    PandarSwiftSDK
)

add_executable(hal_lidar_test main.cpp)
target_link_libraries(hal_lidar_test
    hal_lidar
    ${PCL_LIBRARIES}
)

install(TARGETS hal_lidar
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/lidar
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/lidar
)
install(FILES ${INTERFACE_HPP_PATH}/hal_lidar.hpp 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lidar
)