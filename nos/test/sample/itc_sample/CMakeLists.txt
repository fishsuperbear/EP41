cmake_minimum_required(VERSION 3.10.2)

project(itc_sample)

set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)

add_executable(itc_sample 
    ${CMAKE_CURRENT_SOURCE_DIR}/itc_sample.cpp
)

target_link_libraries(itc_sample
    PRIVATE
    netaos::adf
)

install(
    TARGETS
    itc_sample
    RUNTIME
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/itc_sample/bin
)