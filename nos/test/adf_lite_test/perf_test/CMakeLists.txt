
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "adf_lite_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(fisheye_test)
add_subdirectory(uss_test)
add_subdirectory(fusion_test)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conf
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/adf_lite_test/perf_test  FOLLOW_SYMLINK_CHAIN)