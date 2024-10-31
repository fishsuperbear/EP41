cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "sensor_trans_test")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)

file(GLOB_RECURSE
    IDL_SOURCES
    ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_ota_method*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/genereted/servicebase*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/common*.cxx
    )

include_directories(
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
    ${NETA_MIDDLEWARE_DIR}/log/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/
    # ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
    # ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
)

add_executable(chassis_client_test 
    ${IDL_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/chassis_client_test.cpp    
)

target_link_libraries(chassis_client_test
    pthread
    fastcdr 
    fastrtps 
    neta_cm
    netaos::cm
    dl 
    rt 
    netaos::log 
    globalproto    
)

install(
    TARGETS chassis_client_test RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)