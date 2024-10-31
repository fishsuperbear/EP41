#module名称
set(MODULE_NAME "sys_statemgr")

add_definitions("-D SSM_DEBUG_ON")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -fPIC)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib
    ${CMAKE_SOURCE_DIR}/service/sys_statemgr/spm_svr_gen/lib
    ${CMAKE_SOURCE_DIR}/service/sys_statemgr/mss_cli_gen/lib
)

include_directories(
        ${CMAKE_SOURCE_DIR}/service/sys_statemgr/spm_svr_gen/includes
        ${CMAKE_SOURCE_DIR}/service/sys_statemgr/mss_cli_gen/includes
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        # ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        # ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_SOURCE_DIR}/service/sys_statemgr/src/*.cpp
        # ${CMAKE_SOURCE_DIR}/middleware/em/utils/*.cpp
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/service/sys_statemgr
        ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        neta_sm_client
        neta_phm
        neta_cfg
        neta_zmq_ipc
        dl
        rt
        pps_com
        nos_spm_svr
        nos_mss_cli
        -Wl,--start-group
        ara_com_ics
        someip_e2e_ics
        nlog_ics
        aracore_ics
        dlt_ics
        perftoolkit_ics
        ara_serialize_ics
        ncdump_ics
        someip_ics
        ara_someipbinding_ics
        ncjson_ics
        someip_serializer_ics
        ncore_ics
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libssl.so.1.1
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libcrypto.so.1.1
        -Wl,--end-group
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

set(SSM_CONFIG_FILE_PATH "${CMAKE_SOURCE_DIR}/service/sys_statemgr/conf")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

file(COPY ${SSM_CONFIG_FILE_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)


set(SSM_SPM_LIB_PATH "${CMAKE_SOURCE_DIR}/service/sys_statemgr/spm_svr_gen/lib")
set(SSM_MSS_LIB_PATH "${CMAKE_SOURCE_DIR}/service/sys_statemgr/mss_cli_gen/lib")
file(COPY ${SSM_SPM_LIB_PATH} ${SSM_MSS_LIB_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)
