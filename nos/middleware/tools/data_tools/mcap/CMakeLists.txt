cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "neta_mcap")
#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()


# -----------------------------------neta_mcap.so------------------------------
add_library(neta_mcap
    SHARED
    src/merge.cpp
    src/split.cpp
    src/filter.cpp
    src/mcap_common.cpp
    )
target_include_directories(neta_mcap
PRIVATE
    ./include/
    ${DATA_TOOLS_ROOT_PATH}/bag/include/)
target_link_libraries(neta_mcap
PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
    rosbag2_storage
    rosbag2_cpp
    rcpputils
    neta_cm
    neta_bag)
install(TARGETS neta_mcap
    ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
    LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

# -----------------------------------mcap bin------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_executable(mcap_command
    src/mcap_command.cpp)
target_include_directories(mcap_command
PRIVATE
    ./include/
    ${DATA_TOOLS_ROOT_PATH}/bag/include/
    ${DATA_TOOLS_ROOT_PATH}/common/argvparser)
target_link_libraries(mcap_command
PUBLIC
    neta_mcap)
install(TARGETS mcap_command RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

# -----------------------------------install hpp------------------------------
install(
    DIRECTORY include/
    DESTINATION ${TARGET_PLATFORM}/include/data_tools/mcap)
