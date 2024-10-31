#module名称
set(MODULE_NAME "neta_cm")

#lib名称
set(LIB_NAME neta_cm)

# link_directories(
#     ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
#     ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib
#     ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
#     ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/
#     ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
# )

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_method.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
        )

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${NETA_MIDDLEWARE_DIR}/
        ${CMAKE_SOURCE_DIR}
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_BINARY_DIR}/proto
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        neta_log
        uuid
        jsoncpp
        tinyxml2
        protobuf
        globalproto
)

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/proxy.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/skeleton.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/method.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/method_adapter.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/proto_method.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/cm_logger.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/proto_cm_reader.h
              ${CMAKE_CURRENT_SOURCE_DIR}/include/proto_cm_writer.h
DESTINATION ${TARGET_PLATFORM}/include/cm/include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib/
DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/tools/
DESTINATION ${TARGET_PLATFORM}/tools)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/default_network_list.json
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/default_qos.json
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/topic_monitor_qos.json
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/)
