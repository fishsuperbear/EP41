cmake_minimum_required(VERSION 3.10.2)

project(fisheye_mock)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPILER_SYSROOT}/include
)

add_library(fisheye_mock_test SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fisheye_mock_executor.cpp
    ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc
    ${PRJ_ROOT_DIR}/proto/common/header.pb.cc)

target_link_libraries(fisheye_mock_test netaos::adf-lite)

install(TARGETS fisheye_mock_test LIBRARY DESTINATION ${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib)
