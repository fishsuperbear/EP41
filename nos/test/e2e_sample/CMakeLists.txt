
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(MODULE_NAME "e2e_sample")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
) 

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
)

add_executable(e2e_sample ${SOURCES})

target_include_directories(e2e_sample
        PRIVATE
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
)
target_link_libraries(e2e_sample
        pthread
        neta_crc
        neta_e2e
        neta_e2exf
        neta_e2exf_cpp
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
