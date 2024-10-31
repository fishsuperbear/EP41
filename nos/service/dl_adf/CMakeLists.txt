#module 名称
set(MODULE_NAME "adf_main")

if (NOT (${TARGET_PLATFORM} MATCHES "mdc-llvm"))
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -fPIC --std=gnu++14 -O2 -Wall -Werror")
endif()

link_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib  
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
)

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(
    ${MODULE_NAME} 
    ${SOURCES})

target_include_directories(${MODULE_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${NETA_MIDDLEWARE_DIR}/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
    ${CMAKE_SOURCE_DIR}/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
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
        neta_log
        nvmedia2d
        nvvic)
endif()


target_link_libraries(${MODULE_NAME}
    pthread
    dl
    rt
    yaml-cpp
    fastrtps
    fastcdr
    foonathan_memory-0.7.1
    tinyxml2
    neta_adf
    neta_cm 
    uuid
    jsoncpp
    zipper
    protobuf
    globalproto
    neta_codec

    # adf dependency
    ${ADF_DEPENDENCY}
)

install(
    TARGETS
    ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

