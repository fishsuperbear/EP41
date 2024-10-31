cmake_minimum_required(VERSION 2.8.12)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(helloworld helloworld.cpp)

install(
    TARGETS helloworld RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
