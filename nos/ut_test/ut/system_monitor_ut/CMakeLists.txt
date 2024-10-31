#module名称
set(MODULE_NAME "system_monitor_ut")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

# 代码覆盖率报告编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")

# 设置需要跑单体覆盖率的文件夹
set(TESTED_DIR ${PRJ_ROOT_DIR}/service/system_monitor/src)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
        ${COMPILER_PATH}/../usr/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${TESTED_DIR}/*.c
        ${TESTED_DIR}/*.cxx
        ${TESTED_DIR}/*.cpp
        ${TESTED_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${PRJ_ROOT_DIR}/service/idl/generated/monitor.cxx
        ${PRJ_ROOT_DIR}/service/idl/generated/monitorPubSubTypes.cxx
        ${PRJ_ROOT_DIR}/service/idl/generated/monitorTypeObject.cxx
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/service
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        neta_phm
        gtest
        gtest_main
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        rt
        jsoncpp
        neta_zmq_ipc
        zipper
        uuid
        protobuf
        zmq
        globalproto
        ipc
        neta_sm_client
        yaml-cpp
        )

if(${TARGET_PLATFORM} MATCHES "orin")
        target_link_libraries(${MODULE_NAME}
                PRIVATE
                nvmnand_private
                nvmnand_public
                )
endif()


set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
