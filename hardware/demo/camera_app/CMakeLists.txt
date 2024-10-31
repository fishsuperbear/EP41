cmake_minimum_required(VERSION 3.0)
project(camera_app)

add_definitions(-std=c++14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")

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

FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${CMAKE_SOURCE_DIR}/../../hal/code)

include_directories(
    .
    ./
    # ${INCLUDE_DIR_LIST}
    ${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_SOURCE_DIR}/third_party/framework/include
    ${CMAKE_SOURCE_DIR}/third_party/framework/include/framework/log/spdlog
    ${CMAKE_SOURCE_DIR}/third_party/gflags/include
    ${CMAKE_SOURCE_DIR}/third_party/protobuf/include
    ${CMAKE_SOURCE_DIR}/third_party/fastrtps/include
    ${CMAKE_SOURCE_DIR}/third_party/fastcdr/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyxml2/include
    ${CMAKE_SOURCE_DIR}/../../hal/code/interface/hpp/intf_camera/0.1
)

link_directories(
    ${CMAKE_SOURCE_DIR}/third_party/framework/lib
    ${CMAKE_SOURCE_DIR}/third_party/gflags/lib
    ${CMAKE_SOURCE_DIR}/third_party/protobuf/lib
    ${CMAKE_SOURCE_DIR}/third_party/fastrtps/lib
    ${CMAKE_SOURCE_DIR}/third_party/fastcdr/lib
    ${CMAKE_SOURCE_DIR}/third_party/tinyxml2/lib
    ${CMAKE_SOURCE_DIR}/../../hal/code/lib
)

file(GLOB SOURCE_FILE_LIST
    interface.cc
    camera_app.cc
    camera_device_callback.cc
    proto/*.pb.cc
    # ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl/cpp/impl_camera/*.cpp
    # ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl/cpp/impl_camera/threadroutine/*.cpp
    # ${CMAKE_SOURCE_DIR}/../../hal/code/hal_impl/cpp/impl_camera/pipelineconfig/*.cpp
)

add_library(camera_app SHARED ${SOURCE_FILE_LIST})
target_link_libraries(camera_app
    hal_camera_vs
    # hw_hal_vs
    # hw_platform_vs
    gflags
    protobuf
    fastrtps
    fastcdr
    framework
    tinyxml2
    pthread
)

add_executable(camera_app_bin test/main.cc)
target_link_libraries(camera_app_bin
    camera_app
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

install(TARGETS camera_app camera_app_bin
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY 
    ${CMAKE_SOURCE_DIR}/third_party/framework/lib
    ${CMAKE_SOURCE_DIR}/third_party/gflags/lib
    ${CMAKE_SOURCE_DIR}/third_party/protobuf/lib
    ${CMAKE_SOURCE_DIR}/third_party/fastrtps/lib
    ${CMAKE_SOURCE_DIR}/third_party/fastcdr/lib
    ${CMAKE_SOURCE_DIR}/third_party/tinyxml2/lib
    DESTINATION ${CMAKE_INSTALL_PREFIX} PATTERN "*.so")
install(DIRECTORY ${CMAKE_SOURCE_DIR}/conf DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES 
    "interface.h" 
    "${CMAKE_SOURCE_DIR}/../../hal/code/interface/hpp/intf_camera/0.1/camera_types.hpp" 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
