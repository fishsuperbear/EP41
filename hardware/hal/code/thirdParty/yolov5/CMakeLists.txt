cmake_minimum_required(VERSION 3.5)
include("/drive/env.cmake")

project(yolov5)

set(CMAKE_CXX_STANDARD 11)

# 设置CUDA的版本
find_package(CUDA REQUIRED)
# set(CUDA_ARCH "35" CACHE STRING "CUDA Architecture (default is '35')" )

#设置opencV
set(CMAKE_MODULE_PATH /home/orin/80043632/projects/opencv/build)
set(OpenCV_DIR /home/orin/80043632/projects/opencv/build)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${CMAKE_SOURCE_DIR}/../../../../TensorRT/include
    PATH_SUFFIXES include)


# 添加头文件路径
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../modules/devices/video/modules/nvmedia/common
    ${TENSORRT_INCLUDE_DIR}
)

# 添加源文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
link_directories(
     /home/jiangxing/driveos/drive/drive-linux/lib-target
     /home/jiangxing/driveos/usr/lib/aarch64-linux-gnu
     /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/lib
     ${LIB_DIR}
     #    /usr/local/cuda/lib64
     /usr/local/cuda/targets/aarch64-linux/lib
 )


# 编译为动态库
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBS}
    nvinfer
    )
# 指定安装路径
install(TARGETS yolov5 DESTINATION ${CMAKE_SOURCE_DIR}/../../lib)
# install(DIRECTORY include/ DESTINATION include)

