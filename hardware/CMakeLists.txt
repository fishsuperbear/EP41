cmake_minimum_required(VERSION 3.10)

option(BUILD_ON_DOCKER "enable build on docker"  ON)
if(BUILD_ON_DOCKER)
    message("Now build on Docker env")
    include("/drive/env.cmake")
else()
    message("Now build on Board env")
endif()

# 区分德赛和英伟达板
option(BOARD_CONFIG_NVIDIA "enable camera config for nvidia or desay"  OFF)
if(BOARD_CONFIG_NVIDIA)
    message("Current board is NVIDIA")
    add_definitions(-DBOARD_CONFIG_NVIDIA=1)
else()
    message("Current board is DESAY")
endif()

project(hardware)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")


include(ProjectConfig.cmake)

set(HARDWARE_ROOT_PATH ${CMAKE_SOURCE_DIR} CACHE PATH "Hardware project root path")

# 编译hw_platform项目
add_subdirectory(hal/hw_platform)

# 编译hw_hal项目，依赖hw_platform项目
add_subdirectory(hal/hw_hal)


# 编译gpuutils项目
add_subdirectory(hal/code/thirdParty/gpuutils)

# 编译netacuda项目
add_subdirectory(hal/code/thirdParty/netacuda)

# 编译lidar项目
add_subdirectory(hal/hal_lidar)
add_subdirectory(hal/code/thirdParty/lidar)

# 编译halnode项目
add_subdirectory(hal/halnode)

# 编译hw_nvmedia_multiipc_main项目，依赖hw_hal项目
add_subdirectory(hal/hw_nvmedia/hw_nvmedia_multiipc_main)

# 编译hw_nvmedia_multiipc_consumer_cuda项目，依赖hw_nvmedia_multiipc_main项目
add_subdirectory(hal/hw_nvmedia/hw_nvmedia_multiipc_consumer_cuda)

add_subdirectory(hal/hal_camera)

add_subdirectory(demo/camera_hpp_demo/camera_hpp_main)
