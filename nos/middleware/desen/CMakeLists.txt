cmake_minimum_required(VERSION 3.16)
#module名称
set(MODULE_NAME "neta_desen")

project(${MODULE_NAME} CXX)

# set(3RD_ROOT ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM})
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/netaos_thirdparty/cmake)
# find_package(CUDA REQUIRED)
# set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -arch=sm_87 -Xcompiler -fPIC" CACHE STRING "" FORCE)
# enable_language(CUDA)

# setup cuda
set(CUDA_ROOT ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/aarch64-linux)
set(CMAKE_CUDA_COMPILER ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -arch=sm_87 -Xcompiler -fPIC" CACHE STRING "" FORCE)
enable_language(CUDA)
# setup tensorrt
set(TENSORRT_ROOT ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/tensorrt/targets/aarch64-linux)

# find YAMLCPP
set(3RD_ROOT ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/netaos_thirdparty/cmake)
# find_package(YAMLCPP REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_SKIP_RPATH  TRUE)


if (${TARGET_PLATFORM} MATCHES "orin")
    set(CODEC_PRIVATE_INC_DIRS 
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/nvmedia/
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/nvmedia/utils 
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/nvmedia/iep 
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/nvmedia/ide 
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/orin
        ${CMAKE_CURRENT_SOURCE_DIR}/../codec/src/
        ${CUDA_ROOT}/include
        ${TENSORRT_ROOT}/include
    )       
    add_definitions(-DPLAT_ORIN -DNVMEDIA_NVSCI_ENABLE -D_FILE_OFFSET_BITS=64 -DNV_IS_SAFETY=0)
endif()

include_directories(    
    # ${OPENCV_INCLUDE_DIRS}
    ${COMPILER_SYSROOT}/include
    ${CMAKE_SOURCE_DIR}/middleware
    ${CODEC_PRIVATE_INC_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../codec/include/
    # # 2 yuv2bgr 
    ${CMAKE_CURRENT_SOURCE_DIR}/yuv_rgb_convert/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/YuvToH265/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/configManager
    # process 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # rtr module 
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/trt_c++/include/
    
)

add_library(${MODULE_NAME} SHARED 

    ${CMAKE_CURRENT_SOURCE_DIR}/src/yuv_rgb_converter.cu
    # ${config}
    # ${IDL_PROTO_SRC}
    # ${SOURCES} 
    # trt  module 
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/trt_c++/src/detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/trt_c++/src/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/trt_c++/src/preprocess_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/trt_c++/src/util.cpp

    # # 1 h265 convert to yuv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/desen_process.cpp
    # # 2 yuv2bgr 
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/src/yuv_to_rgb_kernel.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/src/rgb2yuv.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/src/yuv_to_rgb.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/src/Yuv2Rgb.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToRgb/configManager/configManager.cpp
    # 4 yuvto H265
    # ${CMAKE_CURRENT_SOURCE_DIR}/YuvToH265/src/YuvToH265.cpp
)

target_link_directories(${MODULE_NAME} 
    PUBLIC
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
    # ${YAMLCPP_LIBRARY_DIRS}
    ${CUDA_ROOT}/lib
    ${CUDA_ROOT}/lib/stubs
    ${TENSORRT_ROOT}/lib
)

target_link_libraries(${MODULE_NAME} 
    PUBLIC    
    # ${OPENCV_LIBRARIES}
    pthread
    opencv_imgcodecs
    opencv_core
    opencv_imgproc
    neta_log
    neta_codec
    # yaml-cpp
    cudart
    nvinfer
    cudla
    nvdla_compiler
)

# add_executable(main_desen ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# target_link_libraries(main_desen neta_desen)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${TARGET_PLATFORM}/include/desen/include)

install(TARGETS ${MODULE_NAME} LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib)

# TBD for 3rd deploy
set(DESEN_3RD ${3RD_ROOT}/opencv/lib/libopencv_imgcodecs.so ${3RD_ROOT}/opencv/lib/libopencv_core.so ${3RD_ROOT}/opencv/lib/libopencv_imgproc.so)
file(COPY ${DESEN_3RD} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib FOLLOW_SYMLINK_CHAIN)

# for engine file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/trt_infer/weights/picodet_l_416_nonms.engine DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)