cmake_minimum_required(VERSION 3.5.1)

project(hw_nvmedia_multiipc_main VERSION 1.0.0)

# option(ENABLE_FEATURE_OPENCV "Enable feature openCV" OFF)

FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules)
FIND_INCLUDE_DIR(INCLUDE_DIR_INTERFACE_LIST ${HARDWARE_ROOT_PATH}/hal/code/interface)
FIND_INCLUDE_DIR(INCLUDE_DIR_PORTING_LIST ${HARDWARE_ROOT_PATH}/hal/code/porting)
FIND_INCLUDE_DIR(GPUUTILS_INCLUDE_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/thirdParty/gpuutils/include)
FIND_INCLUDE_DIR(INCLUDE_TENSORRT_DIR_LIST ${HARDWARE_ROOT_PATH}/TensorRT)

#CUDA
find_package(CUDA REQUIRED)

option(ENABLE_FEATURE_OPENCV "Enable feature openCV" OFF)
if (ENABLE_FEATURE_OPENCV)
    add_definitions(-DFEATURE_OPENCV_ENABLED)
    set(CMAKE_MODULE_PATH /home/orin/80043632/projects/opencv/build)
    set(OpenCV_DIR /home/orin/80043632/projects/opencv/build)
    find_package(OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(
    ${INCLUDE_DIR_LIST}
    ${INCLUDE_DIR_INTERFACE_LIST}
    ${INCLUDE_DIR_PORTING_LIST}
    ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include
    ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include/nvmedia_6x
    ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include/cuda
    ${HARDWARE_ROOT_PATH}/hal/code/interface/halnode
    ${GPUUTILS_INCLUDE_DIR_LIST}
    ${HARDWARE_ROOT_PATH}/hal/code/thirdParty/netacuda/include
)

file(GLOB_RECURSE HW_NVMEDIA_MULTIIPC_MAIN_SRCS_MAIN
    "main.cpp"
    "enc_thread.cpp"
    )
file(GLOB_RECURSE HW_NVMEDIA_MULTIIPC_MAIN_SRCS
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/common/*.cpp"
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/common/*.cpp"
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/nvmedia_multiipc_producer/*.cpp"
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/nvmedia_multiipc_consumer_enc/*.cpp"
)

#link directorie of lib
link_directories(
    #    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
)

# 定义一个cmake宏，用于选择编译项
macro(ENABLE_LIB_OR_EXE ENABLE_LIB)
    if(${ENABLE_LIB})
        # 编译为动态库
        add_library(${PROJECT_NAME} SHARED ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS})
    else()
        # 编译为可执行文件
        add_executable(${PROJECT_NAME} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS_MAIN})
        add_executable(client ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp)
    endif()
endmacro()

# 调用cmake宏，默认编译为可执行文件，可以通过参数ENABLE_LIB来选择编译为动态库
option(ENABLE_LIB "Build as library" ON)
ENABLE_LIB_OR_EXE(${ENABLE_LIB})

# add_library(${PROJECT_NAME} SHARED ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS})
# add_executable(${PROJECT_NAME} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS_MAIN})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")
# cuda_add_executable(${PROJECT_NAME} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS})

target_link_libraries(${PROJECT_NAME}
    hw_hal
    hw_platform
    nvsipl
    nvsipl_query
    nvmedia_iep_sci
    nvscistream
    nvscibuf
    nvscisync
    nvscievent
    nvsciipc
    nvscicommon
    cuda
    cudart
    pthread
    cudart
    pthread
    # nvinfer
    # myplugins
    nvmedia2d
    gpuutils
    halnode
    #yolov5
    netacuda
)
if (ENABLE_FEATURE_OPENCV)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)
endif()

#install(TARGETS ${PROJECT_NAME}
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)

add_definitions(-D__HW_NVMEDIA_PROJ_MULTIIPC_PRODUCER)

