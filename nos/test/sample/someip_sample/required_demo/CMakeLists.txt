cmake_minimum_required(VERSION 2.8)

set(MODULE_NAME "someip_required_demo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")


#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/someip_required_usage_demo.cpp
    )

add_executable(${MODULE_NAME}
    ${SOURCES}
)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/someip/include
        )

#用户可在此添加链接库目录
link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)

target_link_libraries(${MODULE_NAME}
    pthread
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_someip.so
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
)

install(TARGETS ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
