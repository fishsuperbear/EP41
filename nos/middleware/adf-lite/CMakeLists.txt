project(adf-lite)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
    ${CMAKE_SOURCE_DIR}/middleware/cfg/include/
    ${NETA_MIDDLEWARE_DIR}/
    ${CMAKE_SOURCE_DIR}
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/include/
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
)

### adf-lite library
add_library(adf-lite SHARED 
    src/config.cpp
    src/topology.cpp
    src/bundle.cpp
    src/writer.cpp
    src/writer_impl.cpp
    src/reader.cpp
    src/reader_impl.cpp
    src/classloader.cpp
    src/executor.cpp
    src/executor_impl.cpp
    src/executor_mgr.cpp
    src/topic_manager.cpp
    src/core.cpp
    src/dbg_info.cpp
    src/phm_client_instance.cpp
    ${CMAKE_SOURCE_DIR}/middleware/adf/src/profiler/checkpoint_profiler.cpp
    ${CMAKE_SOURCE_DIR}/middleware/adf/src/profiler/latency_profiler.cpp
    service/rpc/lite_rpc.cpp
)

target_link_directories(adf-lite
    PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lib
)

target_link_libraries(adf-lite
    PUBLIC
    neta_em_client
    neta_cm
    neta_phm
    neta_log
    neta_cfg
    yaml-cpp
    dl
)

install(TARGETS adf-lite LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib)

install(FILES   
    ${CMAKE_CURRENT_SOURCE_DIR}/include/adf_lite_internal_logger.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/adf_lite_logger.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bundle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/writer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/classloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/executor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/dbg_info.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/struct_register.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/phm_client_instance.h
    DESTINATION ${TARGET_PLATFORM}/include/adf-lite/include/)

### lite-process
add_executable(adf-lite-process 
    src/adf_lite_process.cpp
)

target_include_directories(adf-lite-process 
    PRIVATE
    ${NETA_MIDDLEWARE_DIR}/
)

target_link_libraries(adf-lite-process 
    PRIVATE
    adf-lite
    neta_log
    dl
)

install(TARGETS adf-lite-process RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

link_directories(adf-lite-ds
    PUBLIC
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/stubs
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/
)

### adf-lite data_source library
file(GLOB_RECURSE
    SOURCES_DS
    ${CMAKE_CURRENT_SOURCE_DIR}/ds/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ds/ds_recv/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ds/ds_send/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattachPubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattachTypeObject.cxx
)

add_library(adf-lite-ds SHARED ${SOURCES_DS})

if (${TARGET_PLATFORM} MATCHES "orin")
    set(CAMERA_LIBRARY 
        neta_camera 
        nvs
        nvscistream
        nvscibuf
        nvscisync
        nvscievent
        nvsciipc
        nvscicommon
        neta_log
        nvmedia2d
        nvvic
        nvs_consumer
        nvmedia_iep_sci
        nvvideo)
endif()

target_link_libraries(adf-lite-ds
    PUBLIC
    neta_cm
    neta_log
    cuda
    cudart
    ${CAMERA_LIBRARY}
)

install(TARGETS adf-lite-ds LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/data_types/
    DESTINATION ${TARGET_PLATFORM}/include/adf-lite/include/data_types/)

install(FILES   
    ${CMAKE_CURRENT_SOURCE_DIR}/ds/builtin_types.h
    DESTINATION ${TARGET_PLATFORM}/include/adf-lite/include/ds)


add_subdirectory(service)