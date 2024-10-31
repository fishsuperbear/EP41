#module名称
set(MODULE_NAME "network_capture")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

set(MCU_JSON_SOURCE_DIR ${CMAKE_SOURCE_DIR}/service/ethstack/sensor_trans/gen/src/hozon/netaos/v1)
set(SOC_JSON_SOURCE_DIR ${CMAKE_SOURCE_DIR}/service/ethstack/soc_to_hmi/gen/src/ap_datatype_package/v0)
set(JSON_DESTINATION_DIR ${CMAKE_SOURCE_DIR}/service/network_capture/conf) 
# 查找并复制所有.json文件
file(GLOB_RECURSE JSON_FILES ${MCU_JSON_SOURCE_DIR}/*.json)
file(COPY ${JSON_FILES} DESTINATION ${JSON_DESTINATION_DIR})

file(GLOB_RECURSE JSON_FILES ${SOC_JSON_SOURCE_DIR}/*.json)
file(COPY ${JSON_FILES} DESTINATION ${JSON_DESTINATION_DIR})

link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/middleware/em/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
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
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so
        pthread
        neta_log
        neta_em_client
        neta_zmq_ipc
        neta_cm
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/service/network_capture/conf/${TARGET_PLATFORM}/*")
file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)
