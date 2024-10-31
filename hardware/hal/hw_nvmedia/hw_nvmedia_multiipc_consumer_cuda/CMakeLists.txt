cmake_minimum_required(VERSION 3.5.1)

project(hw_nvmedia_multiipc_cuda VERSION 1.0.0)

# option(ENABLE_FEATURE_OPENCV "Enable feature openCV" OFF)

FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules)
FIND_INCLUDE_DIR(INCLUDE_DIR_INTERFACE_LIST ${HARDWARE_ROOT_PATH}/hal/code/interface)
FIND_INCLUDE_DIR(INCLUDE_DIR_PORTING_LIST ${HARDWARE_ROOT_PATH}/hal/code/porting)
FIND_INCLUDE_DIR(GPUUTILS_INCLUDE_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/thirdParty/gpuutils/include)
FIND_INCLUDE_DIR(INCLUDE_TENSORRT_DIR_LIST ${HARDWARE_ROOT_PATH}/TensorRT)

#CUDA
find_package(CUDA REQUIRED)
# Find YOLOv5 include directory
# find_path(YOLOV5_INCLUDE_DIR yololayer.h
#     HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/yolov5/include/
#     PATH_SUFFIXES include)

# # Find GPUUtils include directory
# find_path(GPUUTILS_INCLUDE_DIR gpu_convert.hpp
#     HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/gpuutils/include/
#     PATH_SUFFIXES include)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/yolov5/res/yolov5.engine ${CMAKE_CURRENT_BINARY_DIR}/yolov5.engine COPYONLY)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

set(LIB_DIR ../../code/lib)
set(DRIVE_OS_DIR /home/jiangxing/driveos)
option(ENABLE_FEATURE_OPENCV "Enable feature openCV" OFF)
if (ENABLE_FEATURE_OPENCV)
    add_definitions(-DFEATURE_OPENCV_ENABLED)
    set(CMAKE_MODULE_PATH /home/orin/80043632/projects/opencv/build)
    set(OpenCV_DIR /home/orin/80043632/projects/opencv/build)
    find_package(OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(
    # /home/jiangxing/driveos/drive/drive-linux/include/nvmedia_6x
    # /home/jiangxing/driveos/drive/drive-linux/include
    # /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/include
    ${INCLUDE_DIR_LIST}
    ${INCLUDE_DIR_INTERFACE_LIST}
    ${INCLUDE_DIR_PORTING_LIST}
    # ${INCLUDE_TENSORRT_DIR_LIST}
    #${INCLUDE_DRIVE_OS_DIR_LIST}
     ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include
     ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include/nvmedia_6x
     ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include/cuda
    # ${YOLOV5_INCLUDE_DIR}
     ${GPUUTILS_INCLUDE_DIR_LIST}
     ${HARDWARE_ROOT_PATH}/hal/code/thirdParty/netacuda/include
)

file(GLOB_RECURSE HW_NVMEDIA_IPC_CONSUMER_CUDA_SRCS
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/common/*.cpp"
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/common/*.cpp"
    "${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/nvmedia_multiipc_consumer_cuda/*.cpp"
)

#link directorie of lib
link_directories(
    /usr/local/cuda/targets/aarch64-linux/lib
)

add_library(${PROJECT_NAME} SHARED ${HW_NVMEDIA_IPC_CONSUMER_CUDA_SRCS})
# cuda_add_executable(${PROJECT_NAME} ${HW_NVMEDIA_IPC_CONSUMER_CUDA_SRCS})
# add_executable(${PROJECT_NAME} ${HW_NVMEDIA_IPC_CONSUMER_CUDA_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")

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
    #yolov5
    halnode
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

add_definitions(-D__HW_NVMEDIA_PROJ_MULTIIPC_CONSUMER_CUDA)

