#module名称
set(MODULE_NAME "diag_server_ut")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

# 代码覆盖率报告编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")

set(TESTED_DIR ${PRJ_ROOT_DIR}/middleware/diag/diag_server/src)

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
        ${PRJ_ROOT_DIR}/middleware/idl/generated/servicebase.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/servicebasePubSubTypes.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/servicebaseTypeObject.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/diag.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/diagPubSubTypes.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/diagTypeObject.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/phm.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/phmPubSubTypes.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/phmTypeObject.cxx
        )

list(FILTER SOURCES EXCLUDE REGEX "${TESTED_DIR}/transport/ipc_transport/.*")

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${PRJ_ROOT_DIR}/middleware/diag/diag_server/include/transport/cm_transport
        # ${PRJ_ROOT_DIR}/middleware/diag/diag_server/include/transport/ipc_transport
)

if(ENABLE_SOMEIP)
        target_include_directories(${MODULE_NAME}
                PRIVATE
                ${PRJ_ROOT_DIR}/middleware/diag/dosomeip/someip/src-gen
                ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/include
                ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/include
                ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/include
        )
endif()

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        dl
        pthread
        neta_diag_common
        neta_doip
        neta_docan
        neta_diag_sa
        neta_log
        neta_cm
        neta_em_client
        sqlite3
        neta_cfg
        neta_ipc
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
        neta_sttask
        neta_phm
        neta_sm_client
        yaml-cpp
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
