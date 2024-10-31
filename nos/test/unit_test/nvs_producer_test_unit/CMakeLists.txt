
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

link_directories(
    ${COMPILER_SYSROOT}/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

#用户可在此添加头文件路径
include_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/orin/fast-dds/include/
    ${NETA_MIDDLEWARE_DIR}/
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
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
if (${TARGET_PLATFORM} MATCHES "orin")
    add_subdirectory(nvs_producer_test_common)
    add_subdirectory(testcase_PRODUCER_NVSTREAM_001)
    add_subdirectory(testcase_PRODUCER_NVSTREAM_002)
    add_subdirectory(testcase_PRODUCER_NVSTREAM_003)
endif()
