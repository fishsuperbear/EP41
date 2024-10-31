cmake_minimum_required(VERSION 2.8.12)

include(${PRJ_ROOT_DIR}/cmake/neta_fastddsgen.cmake)

file(GLOB_RECURSE
        IDL_FILES
        ${PRJ_ROOT_DIR}/test/latency/idl/*.idl)

FASTDDS_GENERATE_CPP(
        IDL_ROOT_DIR ${PRJ_ROOT_DIR}/test/latency/idl/
        IDL_OUTPUT_DIR ${PRJ_ROOT_DIR}/test/latency/gen/
        ${IDL_FILES})

include_directories(${PRJ_ROOT_DIR}/middleware)
include_directories(${PRJ_ROOT_DIR}/test/latency)
include_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/)
link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/)

file(GLOB_RECURSE
        IDL_GEN_FILES
        ${PRJ_ROOT_DIR}/test/latency/gen/*.cxx)

add_executable(lat_pub dds_helper.cpp pub_main.cpp ${IDL_GEN_FILES})
add_executable(lat_sub dds_helper.cpp sub_main.cpp ${IDL_GEN_FILES})

target_link_libraries(lat_pub fastcdr fastrtps netaos::log pthread dl rt foonathan_memory-0.7.1 tinyxml2)
target_link_libraries(lat_sub fastcdr fastrtps netaos::log pthread dl rt foonathan_memory-0.7.1 tinyxml2)

install(
    TARGETS lat_pub lat_sub
    DESTINATION ${TARGET_PLATFORM}/bin
)
