cmake_minimum_required(VERSION 3.10.2)

project(proto_image)

file(COPY ${PRJ_ROOT_DIR}/test/sample/adf_lite_sample/data_type
    DESTINATION  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/sample/proto_image_sample/  FOLLOW_SYMLINK_CHAIN)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/sample/adf_lite_sample/data_type
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/cfg
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/include/
)

link_directories(proto_image
    PUBLIC
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/stubs
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/
)

add_library(proto_image SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/proto_image_executor.cpp
    ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc
    ${PRJ_ROOT_DIR}/proto/common/header.pb.cc)

target_link_libraries(proto_image netaos::adf-lite cuda cudart)

install(TARGETS proto_image LIBRARY DESTINATION ${TARGET_PLATFORM}/test/emproc_adf_test/proto_image_sample/lib)

