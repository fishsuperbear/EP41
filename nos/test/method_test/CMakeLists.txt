
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "method_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

file(GLOB_RECURSE
        IDL_SOURCES
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase*.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm_method*.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_method*.cxx)

include_directories(
        ${NETA_MIDDLEWARE_DIR}/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
)

add_executable(method_test_client 
        ${IDL_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/method_test_client.cpp)

add_executable(method_test_server 
        ${IDL_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/method_test_server.cpp)


target_link_libraries(method_test_client
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        neta_cm
        dl
        rt
        netaos::log
        uuid
        protobuf
        globalproto
)

target_link_libraries(method_test_server
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        neta_cm
        dl
        rt
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so
        uuid
        netaos::log
        protobuf
        globalproto
)

install(
    TARGETS method_test_client method_test_server RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
