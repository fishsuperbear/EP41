cmake_minimum_required(VERSION 3.5.1)
project(hal_lidar VERSION 0.0.1)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()
# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../hal/code/lib)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../
    ${CMAKE_SOURCE_DIR}/../../
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/global
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/global/devtype
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/lidar
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/lidar/moduleid
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/lidar/v0.1
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/video
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/video/moduleid
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/devices/video/v0.1
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/platform
    ${CMAKE_SOURCE_DIR}/../../hal/code/porting
    ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl
    ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl/hal
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices
)

link_libraries(
    ${CMAKE_SOURCE_DIR}/../../hal/code/lib
)

file(GLOB LIDAR_FILE_LIST
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/common/impl/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/common/impl/config/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/common/impl/parser/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/common/impl/protocol/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/normal/common/impl/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/normal/*.cpp
    ${CMAKE_SOURCE_DIR}/../../hal/code/modules/devices/lidar/modules/normal/impl/*.cpp
)
add_library(hw_lidar_vs SHARED ${LIDAR_FILE_LIST})
target_link_libraries(hw_lidar_vs
    pthread
)

# file(GLOB HW_HAL_VS_FILE_LIST
#     ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl/hal/*.cpp
# )
# add_library(hw_hal_vs SHARED ${HW_HAL_VS_FILE_LIST})
# target_link_libraries(hw_hal_vs
#     dl
# )

add_executable(hal_lidar main.cc)
target_link_libraries(hal_lidar
    pthread
    hw_hal_vs
    hw_platform_vs
)
