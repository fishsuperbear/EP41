
cmake_minimum_required(VERSION 2.8.12)

get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME ${CURRENT_DIR_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

file(GLOB_RECURSE
        IDL_SOURCES
        ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach*.cxx

        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase*.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm_method*.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_method*.cxx
)

include_directories(
        ${NETA_MIDDLEWARE_DIR}/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
)

add_executable(${MODULE_NAME} 
        ${IDL_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

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
            nvvideo

            cuda
            cudart
            tegrawfd
            nvidia-kms
            nvdc
            nvddk_2d_v2
            nvimp
            nvddk_vic
            )
endif()
    

target_link_libraries(${MODULE_NAME}
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
        gtest
        gtest_main
    
        ${CAMERA_LIBRARY}
)

# install(
#     TARGETS ${MODULE_NAME} RUNTIME
#     DESTINATION ${TARGET_PLATFORM}/bin
# )
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/nvs_producer_test_unit/${CURRENT_DIR_NAME}/bin
)
