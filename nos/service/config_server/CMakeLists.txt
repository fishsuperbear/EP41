#module名称
set(MODULE_NAME "config_server")


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib;${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib;${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -pthread -Wall")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -fPIC)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")
link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/st_someip/lib
)
#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/service/config_server/src/cfg_server.cpp
        ${CMAKE_SOURCE_DIR}/service/config_server/src/cfg_server_proto.cpp
        ${CMAKE_SOURCE_DIR}/service/config_server/src/cfg_manager.cpp
        ${CMAKE_SOURCE_DIR}/service/config_server/src/main.cpp  
        ${CMAKE_SOURCE_DIR}/service/config_server/src/phm_client_instance.cpp
        ${CMAKE_SOURCE_DIR}/service/config_server/src/cfg_vehiclecfg_update.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/*.cxx
        ${CMAKE_SOURCE_DIR}/service/config_server/proto/cfg.pb.cc
        ${CMAKE_SOURCE_DIR}/service/config_server/gen/src/hozon/netaos/v1/vehiclecfgservice_skeleton.cpp
        ${CMAKE_SOURCE_DIR}/service/config_server/gen/src/hozon/netaos/v1/vehiclecfgservice_manifest.cpp
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service/config_server
        ${CMAKE_SOURCE_DIR}/service/config_server/proto/cfg
        ${CMAKE_SOURCE_DIR}/service/config_server/include
        ${CMAKE_SOURCE_DIR}/middleware/phm
        ${CMAKE_SOURCE_DIR}/middleware/phm/include
        ${CMAKE_SOURCE_DIR}/middleware/per
        ${CMAKE_SOURCE_DIR}/middleware/per/include
        ${CMAKE_SOURCE_DIR}/middleware/per/proto
        ${CMAKE_SOURCE_DIR}/middleware/per/struct2x
        ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/include
        ${CMAKE_SOURCE_DIR}/service/config_server/gen/includes
        ${CMAKE_SOURCE_DIR}/service/config_server/gen/includes
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
)


set(SOMEIP_LIBRARY
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
        pthread
        -Wl,--end-group
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        protobuf
        neta_log
        neta_phm
        neta_cm
        neta_per
        neta_em_client
        ${SOMEIP_LIBRARY}
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
file(GLOB CONF_FILES  "conf/config_server.json")
file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

file(GLOB PROTO_SOURCE_DIR  ${CMAKE_SOURCE_DIR}/service/config_server/proto/cfg/*.proto)
file(COPY ${PROTO_SOURCE_DIR} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf/bag/proto)
