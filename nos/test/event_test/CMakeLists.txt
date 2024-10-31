
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "event_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/) 

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmTypeObject.cxx)

add_executable(event_test ${SOURCES})

target_include_directories(event_test
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${NETA_MIDDLEWARE_DIR}/
)
target_link_libraries(event_test
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

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
