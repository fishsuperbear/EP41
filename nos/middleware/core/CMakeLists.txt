cmake_minimum_required(VERSION 3.5)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(MODULE_NAME core)


# 添加可执行 target
# add_library(${MODULE_NAME} OBJECT ${SRCS} ${PROTO_SRCS})

install(DIRECTORY ../core
        DESTINATION ${TARGET_PLATFORM}/include)