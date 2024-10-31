#module名称
set(MODULE_NAME "neta_adf")

#lib名称
set(LIB_NAME neta_adf)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        )

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${NETA_MIDDLEWARE_DIR}/
        ${CMAKE_SOURCE_DIR}
        ${COMPILER_SYSROOT}/include
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/include/
        neta_codec
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
        ${COMPILER_SYSROOT}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/stubs
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/aarch64-linux/lib/stubs
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/aarch64-linux/lib/
)


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

target_link_libraries(${MODULE_NAME}
        PUBLIC
        neta_cm
        neta_log
        neta_em_client
        yaml-cpp
        neta_codec
        cuda
        cudart
        ${CAMERA_LIBRARY}
        neta_phm
)

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/data_types/
        DESTINATION ${TARGET_PLATFORM}/include/adf/include/data_types/)

install(FILES   ${CMAKE_CURRENT_SOURCE_DIR}/include/log.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/node_base.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/node_bundle.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/thread_pool.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/node_config.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/node_profiler_token.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/class_register.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/class_loader.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/optional.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/base.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/node_proto_register.h
        DESTINATION ${TARGET_PLATFORM}/include/adf/include/)

install(FILES   ${CMAKE_CURRENT_SOURCE_DIR}/include/itc/itc.h
        DESTINATION ${TARGET_PLATFORM}/include/adf/include/itc)

install(FILES   ${CMAKE_CURRENT_SOURCE_DIR}/include/ts_align/ts_align.h
        DESTINATION ${TARGET_PLATFORM}/include/adf/include/ts_align)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/conf/
        DESTINATION ${TARGET_PLATFORM}/conf/adf/)

install(DIRECTORY 
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    DESTINATION ${TARGET_PLATFORM}/lib
)