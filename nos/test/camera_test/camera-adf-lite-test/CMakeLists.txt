cmake_minimum_required(VERSION 3.10.2)

set(MODULE_NAME "camera_adf_lite")

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

link_directories(${MODULE_NAME}
    PUBLIC
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/stubs
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/
)

add_library(${MODULE_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvs_sensor_executor.cpp
)

target_link_libraries(${MODULE_NAME} 
    netaos::adf-lite
    cuda 
    cudart
)

install(TARGETS 
    ${MODULE_NAME} 
    LIBRARY 
    DESTINATION 
    ${CAMERA_TEST_DIR}/${MODULE_NAME}/lib
)

install(DIRECTORY  
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/conf
)


file(COPY
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/bin/adf-lite-process
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/bin
    FOLLOW_SYMLINK_CHAIN
)

file(COPY
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libadf-lite-ds.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libadf-lite.so
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/lib
    FOLLOW_SYMLINK_CHAIN
)

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/env.sh
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/
    FOLLOW_SYMLINK_CHAIN
)
