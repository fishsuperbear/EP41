
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "idl_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmTypeObject.cxx)

add_executable(idl_test ${SOURCES})

target_include_directories(idl_test
        PRIVATE
        ${NETA_MIDDLEWARE_DIR}/idl/generated/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
)
target_link_libraries(idl_test  
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        netaos::log
        pthread
        dl
        rt
        jsoncpp
        zipper
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
