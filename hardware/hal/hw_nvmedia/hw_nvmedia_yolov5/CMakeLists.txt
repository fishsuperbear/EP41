cmake_minimum_required(VERSION 3.5.1)
include("/drive/env.cmake")
project(hw_nvmedia_yolov5 VERSION 1.0.0)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#自动查找头文件路径函数(没有去重)
macro(FIND_INCLUDE_DIR result curdir)  #定义函数,2个参数:存放结果result；指定路径curdir；
    file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h" )	#遍历获取{curdir}中*.hpp和*.h文件列表
    # message(STATUS "children= ${children}")								#打印*.hpp和*.h的文件列表
    set(dirlist "")														#定义dirlist中间变量，并初始化
    foreach(child ${children})											#for循环
        string(REGEX REPLACE "(.*)/.*" "\\1" LIB_NAME ${child})			#字符串替换,用/前的字符替换/*h
        if(IS_DIRECTORY ${LIB_NAME})									#判断是否为路径
            LIST(APPEND dirlist ${LIB_NAME})							#将合法的路径加入dirlist变量中
        endif()															#结束判断
    endforeach()														#结束for循环
    set(${result} ${dirlist})											#dirlist结果放入result变量中
endmacro()																#函数结束

FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../code)
FIND_INCLUDE_DIR(INCLUDE_TENSORRT_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../../TensorRT)

find_package(CUDA REQUIRED)
# Find YOLOv5 include directory
#find_path(YOLOV5_INCLUDE_DIR yololayer.h
#    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/yolov5/include/
#    PATH_SUFFIXES include)

# Find GPUUtils include directory
#find_path(GPUUTILS_INCLUDE_DIR gpu_convert.hpp
#    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/gpuutils/include/
#    PATH_SUFFIXES include)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../code/thirdParty/yolov5/res/yolov5.engine ${CMAKE_CURRENT_BINARY_DIR}/yolov5.engine COPYONLY)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

set(LIB_DIR ../../code/lib)
set(OpenCV_INCLUDE_DIRS /drive/thirdparty/include/opencv4/)
option(ENABLE_FEATURE_OPENCV "Enable feature openCV" ON)
if (ENABLE_FEATURE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(
    ${INCLUDE_DIR_LIST}
    ${INCLUDE_TENSORRT_DIR_LIST}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../driveos/v6.0.6/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../driveos/v6.0.6/include/nvmedia_6x
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../driveos/v6.0.6/include/cuda
    ${YOLOV5_INCLUDE_DIR}
    ${GPUUTILS_INCLUDE_DIR}
)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/res/ DESTINATION ${EXECUTABLE_OUTPUT_PATH})

file(GLOB_RECURSE HW_NVMEDIA_GROUPA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../code/modules/devices/video/*.cpp"
)

#link directorie of lib
link_directories(
    ${LIB_DIR}
    /usr/local/cuda/targets/aarch64-linux/lib
)

#add_library(${PROJECT_NAME} SHARED ${HW_NVMEDIA_MULTI_GROUP_SRCS})
add_executable(${PROJECT_NAME} ${HW_NVMEDIA_GROUPA_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")
target_link_libraries(${PROJECT_NAME}
    hw_hal_vs
    hw_platform_vs
    nvsipl
    nvsipl_query
    nvmedia_iep_sci
    nvscistream
    #tegrawfd
    nvscibuf
    nvscisync
    nvscievent
    nvsciipc
    nvscicommon
    #dl
    cuda
    cudart
    pthread
    cudla
    nvinfer
    nvmedia2d
    gpuutils
    yolov5
)
if (ENABLE_FEATURE_OPENCV)
target_link_libraries(${PROJECT_NAME}
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
)
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

add_definitions(-D__HW_NVMEDIA_PROJ_USING_GROUPA)

