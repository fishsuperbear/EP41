cmake_minimum_required(VERSION 3.5)
# 添加头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../code/interface/lognode
)
# 添加源文件
file(GLOB_RECURSE SOURCES_WRITER "${CMAKE_CURRENT_SOURCE_DIR}/../code/lognode_impl/logblockwriter_impl.cpp" )
file(GLOB_RECURSE SOURCES_READER "${CMAKE_CURRENT_SOURCE_DIR}/../code/lognode_impl/logblockreader_impl.cpp" )
# 编译为动态库
add_library(logblockwriter SHARED ${SOURCES_WRITER})
add_library(logblockreader SHARED ${SOURCES_READER})
# 指定安装路径
# install(TARGETS halnode DESTINATION /usr/lib)
# install(DIRECTORY include/ DESTINATION include)
