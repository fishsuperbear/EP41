
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "crc_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(crc_test ${SOURCES})

target_include_directories(crc_test
        PRIVATE
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/

)
target_link_libraries(crc_test
        netaos::crc
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
