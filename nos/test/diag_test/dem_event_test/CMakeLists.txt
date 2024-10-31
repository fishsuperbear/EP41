
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "dem_event_test")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phmTypeObject.cxx
    )

add_executable(${MODULE_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        )

target_link_libraries(${MODULE_NAME}
        pthread
        netaos::cm
        netaos::log
        )

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
    )
