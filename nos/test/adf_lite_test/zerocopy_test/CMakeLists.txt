cmake_minimum_required(VERSION 3.10.2)

project(fusion_test)

include_directories(
    ${PRJ_ROOT_DIR}
    ${NETA_MIDDLEWARE_DIR}
    ${NETA_MIDDLEWARE_DIR}/test/adf_lite_test
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
file(GLOB_RECURSE
        IDL_SOURCES
        ${NETA_MIDDLEWARE_DIR}/idl/generated/*.cxx)

add_library(fusion_test SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fusion_test_executor.cpp
    ${IDL_SOURCES}
    )


target_link_libraries(fusion_test netaos::adf-lite)

install(TARGETS fusion_test LIBRARY DESTINATION ${TARGET_PLATFORM}/test/adf_lite_test/perf_test/lib)
