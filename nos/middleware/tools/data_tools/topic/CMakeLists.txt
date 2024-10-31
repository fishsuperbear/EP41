cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "topic")
#Check C++11
# include(CheckCXXCompilerFlag)
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
#     if(NOT SUPPORTS_CXX11)
#         message(FATAL_ERROR "Compiler doesn't support C++11")
#     endif()
# endif()
if (${BUILD_TYPE} MATCHES "debug")
  set(CMAKE_CXX_FLAGS "-Wl,--as-needed -g -fPIC --std=gnu++14 -O0 -Wall")
endif()
# -----------------------------------neta_topic.so------------------------------
file(GLOB_RECURSE
        MONITOR_SOURCES
        src/monitor/*.cc
        )

message(STATUS "Configuring topic...")
add_library(neta_topic SHARED
    src/echo.cpp
    src/impl/echo_impl.cpp
    src/hz.cpp
    src/impl/hz_impl.cpp
    src/list.cpp
    src/impl/list_impl.cpp
    src/monitor.cpp
    src/impl/monitor_impl.cpp
    src/latency.cpp
    src/impl/latency_impl.cpp
    src/impl/sub_base.cpp
    src/impl/someip_deserialize_impl.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/topic_manager.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_utility.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_factory.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/process_utility.cpp
    ${MONITOR_SOURCES}
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbuf.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx

    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufPubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbuf.cxx

    ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodPubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_method.cxx

    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/../proto/test/soc/dbg_msg.pb.cc

    ${DATA_TOOLS_ROOT_PATH}/common/util/src/adf_lite_utile.cpp
    )

target_include_directories(neta_topic
  PRIVATE
    ./include
    ./src
    ${DATA_TOOLS_ROOT_PATH}/common/
    ${DATA_TOOLS_ROOT_PATH}/common/ament_index_cpp/include
    ${DATA_TOOLS_ROOT_PATH}/common/util/include
    ${DATA_TOOLS_ROOT_PATH}/common/rcpputils/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include/
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/ncurses-6.4/include/
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include/
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/iauto_someip/include/
    ${NETA_MIDDLEWARE_DIR}
    ${NETA_MIDDLEWARE_DIR}/../)

target_link_directories(neta_topic
    PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib)


# 因为商泰反序列json库依赖libjsoncpp.so.24 这里创建一个
set(CP_COMMAND cp ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so.24)
execute_process(COMMAND ${CP_COMMAND})

target_link_libraries(neta_topic
PUBLIC
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/ncurses-6.4/lib/libncurses.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so.24
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/zipper/lib/libzipper.so
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/lib/libprotobuf.so
    ament_index_cpp
    neta_log
    neta_cm
    nlog_ics
    ncore_ics
    dlt_ics
    ncjson_ics
    someipserializer_ics
    )

install(TARGETS neta_topic
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)



# 复制someip反序列化json文件
file(GLOB_RECURSE JSON_FILES ${CMAKE_SOURCE_DIR}/service/ethstack/sensor_trans/gen/json/TotalJsonFormat.json)
set(JSON_DESTINATION_DIR ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf/bag/someip)
file(COPY ${JSON_FILES} DESTINATION ${JSON_DESTINATION_DIR})
set(MV_COMMAND mv ${JSON_DESTINATION_DIR}/TotalJsonFormat.json ${JSON_DESTINATION_DIR}/sensor_trans_TotalJsonFormat.json)
execute_process(COMMAND ${MV_COMMAND})

file(GLOB_RECURSE JSON_FILES ${CMAKE_SOURCE_DIR}/service/ethstack/soc_to_hmi/gen/json/TotalJsonFormat.json)
set(JSON_DESTINATION_DIR ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf/bag/someip)
file(COPY ${JSON_FILES} DESTINATION ${JSON_DESTINATION_DIR})
set(MV_COMMAND mv ${JSON_DESTINATION_DIR}/TotalJsonFormat.json ${JSON_DESTINATION_DIR}/soc_to_hmi_TotalJsonFormat.json)
execute_process(COMMAND ${MV_COMMAND})

  # -----------------------------------topic bin------------------------------
add_executable(topic src/topic_command.cpp)
target_include_directories(topic
  PRIVATE
    ./include
    ${DATA_TOOLS_ROOT_PATH}/common
    ${DATA_TOOLS_ROOT_PATH}/common/util/include
    ${DATA_TOOLS_ROOT_PATH}/common/rcpputils/include)
target_link_libraries(topic PUBLIC neta_topic)
install(TARGETS topic RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

# -----------------------------------install hpp------------------------------
install(
  DIRECTORY include/
  DESTINATION ${TARGET_PLATFORM}/include/data_tools/topic)

  # -----------------------------------install dependencies------------------------------
install(FILES
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/ncurses-6.4/lib/libncurses.so
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/ncurses-6.4/lib/libncurses.so.6
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/ncurses-6.4/lib/libncurses.so.6.4
  DESTINATION ${TARGET_PLATFORM}/lib/)
