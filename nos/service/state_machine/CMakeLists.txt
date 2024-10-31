
# cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "state_machine")
project(${MODULE_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wno-multichar -std=c++14)


file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
        )

add_executable(${MODULE_NAME} ${SOURCES})

target_link_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_BINARY_DIR}/proto
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
        # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/map_manage/lib/
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/
        ${COMPILER_PATH}/../usr/lib/)

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${CMAKE_SOURCE_DIR}/middleware/
        ${CMAKE_SOURCE_DIR}/middleware/cfg/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/avp_map_manage/
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include/
        ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/
)

if (${TARGET_PLATFORM} MATCHES "orin")
    set(ADF_DEPENDENCY 
        nvs
        nvscistream
        nvscibuf
        nvscisync
        nvscievent
        nvsciipc
        nvscicommon
        cuda
        cudart
        nvmedia2d
        nvvic)
endif()


target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        dl
        rt
        yaml-cpp
        fastrtps
        fastcdr
        neta_adf
        neta_cfg
        neta_log
        protobuf
        avp_map_manage
        globalproto

        # adf dependency
        ${ADF_DEPENDENCY}
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)