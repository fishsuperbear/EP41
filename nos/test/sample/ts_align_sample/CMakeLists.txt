cmake_minimum_required(VERSION 3.10.2)

project(ts_align_sample)

set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)

add_executable(ts_align_sample 
    ${CMAKE_CURRENT_SOURCE_DIR}/ts_align_sample.cpp
)

target_link_libraries(ts_align_sample
    PRIVATE
    netaos::adf
)

install(
    TARGETS
    ts_align_sample
    RUNTIME
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/ts_align_sample/bin
)