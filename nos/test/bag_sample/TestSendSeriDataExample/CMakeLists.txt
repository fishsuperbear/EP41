cmake_minimum_required(VERSION 3.16.3)

project(TestExample VERSION 1 LANGUAGES CXX)

#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()

message(STATUS "Configuring HelloWorld example...")
file(GLOB TEST_EXAMPLE_SOURCES_CXX "*.cxx")
file(GLOB TEST_EXAMPLE_SOURCES_CPP "*.cpp")

set(CMAKE_CXX_STANDARD 14)

add_executable(TestExample ${TEST_EXAMPLE_SOURCES_CXX} ${TEST_EXAMPLE_SOURCES_CPP})
target_compile_definitions(TestExample PRIVATE
    $<$<AND:$<NOT:$<BOOL:${WIN32}>>,$<STREQUAL:"${CMAKE_BUILD_TYPE}","Debug">>:__DEBUG>
    $<$<BOOL:${INTERNAL_DEBUG}>:__INTERNALDEBUG> # Internal debug activated.
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${BAG_ROOT_PATH}/rosbag2/rosbag2_cpp/include
    ${BAG_ROOT_PATH}/rosbag2/rosbag2_storage/include
    ${BAG_ROOT_PATH}/rcutils/include
    ${BAG_ROOT_PATH}/rcpputils/include
)

target_link_libraries(TestExample
    fastrtps
    fastcdr
    # fastdds::optionparser
    rosbag2_cpp
    )
install(TARGETS TestExample
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)