cmake_minimum_required(VERSION 3.5)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
# 设置模块名
set(MODULE_NAME network_capture_test_001)
# 设置中间件头文件路径和lib路径
set(MIDDLEWARE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../output/${TARGET_PLATFORM}/include)
set(MIDDLEWARE_LIB_DIR ${PROJECT_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib)

set(CMAKE_INSTALL_RPATH "MIDDLEWARE_LIB_DIR;${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)


#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${PRJ_ROOT_DIR}/service/network_capture/src/*.c
        ${PRJ_ROOT_DIR}/service/network_capture/src/*.cxx
        ${PRJ_ROOT_DIR}/service/network_capture/src/*.cpp
        ${PRJ_ROOT_DIR}/service/network_capture/src/*.cc
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufTypeObject.cxx
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME} PUBLIC
        ${PRJ_ROOT_DIR}/
        ${PRJ_ROOT_DIR}/service/
        ${PRJ_ROOT_DIR}/middleware/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME} PRIVATE
        stdc++fs
        gtest
        gtest_main
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so
        pthread
        neta_phm
        fastcdr
        fastrtps
        uuid
        protobuf
        neta_cm
        neta_sm_client
        netaos::log
        foonathan_memory-0.7.1
        tinyxml2
        jsoncpp
        dl
        rt
        neta_zmq_ipc
        zipper
        zmq
        yaml-cpp
        globalproto
        gtest
        gtest_main
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed,--disable-new-dtags")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/network_capture_test_unit/network_capture_test_001/bin/
)

# set(SCRIPTS_PATH ${CMAKE_SOURCE_DIR}/unit_test/network_capture_test_unit/network_capture_test_001/scripts)
# set(CONFIG_PATH ${CMAKE_SOURCE_DIR}/unit_test/network_capture_test_unit/network_capture_test_001/conf)
file(COPY ${SCRIPTS_PATH} ${CONFIG_PATH}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/network_capture_test_unit/network_capture_test_001)