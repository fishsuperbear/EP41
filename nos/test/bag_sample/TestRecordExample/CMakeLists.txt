cmake_minimum_required(VERSION 3.16.3)

project(TestRecordExample VERSION 1 LANGUAGES CXX)

SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/cmake/fastcdr")
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/cmake/tinyxml2")
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/foonathan_memory/cmake/")
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/share/fastrtps/cmake/")

#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()

message(STATUS "Configuring TypeLookup example...")
file(GLOB DDS_RECORDER_EXAMPLE_SOURCES_CPP "*.cpp")

set(CMAKE_CXX_STANDARD 14)

add_executable(TestRecordExample
    ${DDS_RECORDER_EXAMPLE_SOURCES_CPP})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${BAG_ROOT_PATH}/rosbag2/rosbag2_cpp/include
    ${BAG_ROOT_PATH}/rosbag2/rosbag2_storage/include
    ${BAG_ROOT_PATH}/rcutils/include
    ${BAG_ROOT_PATH}/rcpputils/include
)

target_compile_definitions(TestRecordExample PRIVATE
    $<$<AND:$<NOT:$<BOOL:${WIN32}>>,$<STREQUAL:"${CMAKE_BUILD_TYPE}","Debug">>:__DEBUG>
    $<$<BOOL:${INTERNAL_DEBUG}>:__INTERNALDEBUG> # Internal debug activated.
)

target_link_libraries(TestRecordExample
    fastcdr
    fastrtps
    tinyxml2
    rosbag2_storage
    rosbag2_cpp
    rcpputils
    # ${CMAKE_CURRENT_BINARY_DIR}/../../rosbag2/rosbag2_storage/librosbag2_storage.so
    # ${CMAKE_CURRENT_BINARY_DIR}/../../rosbag2/rosbag2_cpp/librosbag2_cpp.so
    # ${CMAKE_CURRENT_BINARY_DIR}/../../rcpputils/librcpputils.so
    )
install(TARGETS TestRecordExample
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example_type.xml
    DESTINATION ${TARGET_PLATFORM}/bin)

file(COPY example_type.xml DESTINATION ${PROJECT_BINARY_DIR})
