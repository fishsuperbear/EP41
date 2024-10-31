cmake_minimum_required(VERSION 3.10.2)

project(fisheye_perception)

file(COPY ${PRJ_ROOT_DIR}/test/sample/adf_lite_sample/data_type
    DESTINATION  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/sample/adf_lite_sample/  FOLLOW_SYMLINK_CHAIN)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/sample/adf_lite_sample/data_type
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/gflags/include
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/cfg
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
)


link_directories(fisheye_perception
    PUBLIC
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/gflags/lib/
)

add_library(fisheye_perception_test SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fisheye_perception_executor.cpp
    ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc
    ${PRJ_ROOT_DIR}/proto/common/header.pb.cc)

target_link_libraries(fisheye_perception_test netaos::adf-lite gflags)

install(TARGETS fisheye_perception_test LIBRARY DESTINATION ${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib)
