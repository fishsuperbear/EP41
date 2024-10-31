
cmake_minimum_required(VERSION 3.5.1)
# include("/drive/env.cmake")

project(hal_test VERSION 1.0.0)


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

FIND_INCLUDE_DIR(INCLUDE_VIDEO_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/interface)
FIND_INCLUDE_DIR(INCLUDE_GLOBAL_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/porting/)
FIND_INCLUDE_DIR(INCLUDE_IMPL_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/hal_impl)

#CUDA
find_package(CUDA REQUIRED)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/interface/hpp/intf_camera/0.1
    ${INCLUDE_VIDEO_DIR_LIST}
    ${INCLUDE_GLOBAL_DIR_LIST}
    ${INCLUDE_IMPL_DIR_LIST}
)
message(${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/interface/hppintf_camera/0.1)

file(GLOB_RECURSE HW_NVMEDIA_MULTIIPC_MAIN_SRCS_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/code/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/hal_impl/*.cpp"
    )

file(GLOB_RECURSE HW_NVMEDIA_MULTIIPC_MAIN_SRCS_SUB
    "${CMAKE_CURRENT_SOURCE_DIR}/code/gpu_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/hal_impl/*.cpp"
    )
file(GLOB_RECURSE HW_NVMEDIA_MULTIIPC_MAIN_SRCS_CLIENT
    "${CMAKE_CURRENT_SOURCE_DIR}/code/client.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/hal_impl/*.cpp"
    )
#link directorie of lib
link_directories(
    ${LIB_DIR}
    /usr/local/cuda/targets/aarch64-linux/lib
)

add_executable(${PROJECT_NAME} ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS_MAIN})
# add_executable(halTestGPU ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS_SUB})
# add_executable(client ${HW_NVMEDIA_MULTIIPC_MAIN_SRCS_CLIENT})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")
# set_target_properties(halTestGPU PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")
# set_target_properties(client PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")

target_link_libraries(${PROJECT_NAME}
    hw_hal_vs
    hw_platform_vs
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
    dl
)
#target_link_libraries(halTestGPU
#    hw_hal_vs
#    hw_platform_vs
#    nvsipl
#    nvsipl_query
#    nvmedia_iep_sci
#    nvscistream
#    nvscibuf
#    nvscisync
#    nvscievent
#    nvsciipc
#    nvscicommon
#    cuda
#    cudart
#    pthread
#    cudart
#    pthread
#    # nvinfer
#    # myplugins
#    nvmedia2d
#    gpuutils
#    #yolov5
#    dl
#)
#target_link_libraries(client
#    hw_hal_vs
#    hw_platform_vs
#    nvsipl
#    nvsipl_query
#    nvmedia_iep_sci
#    nvscistream
#    nvscibuf
#    nvscisync
#    nvscievent
#    nvsciipc
#    nvscicommon
#    cuda
#    cudart
#    pthread
#    cudart
#    pthread
#    # nvinfer
#    # myplugins
#    nvmedia2d
#    gpuutils
#    #yolov5
#    dl
#)
