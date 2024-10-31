cmake_minimum_required(VERSION 3.10)

project(ut_test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=c++14 -O3 -Wall -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-command-line-argument")

if (${BUILD_TYPE} MATCHES "release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (${BUILD_TYPE} MATCHES "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
    message(FATAL_ERROR "unsupported build type ${BUILD_TYPE}")
endif()

add_definitions(-DBUILD_TYPE="${BUILD_TYPE}")
add_definitions(-DTARGET_PLATFORM="${TARGET_PLATFORM}")
if (${TARGET_PLATFORM} MATCHES "x86_2004")
    add_definitions(-DBUILD_FOR_X86=1)
elseif (${TARGET_PLATFORM} MATCHES "mdc")
    add_definitions(-DBUILD_FOR_MDC=1)
elseif (${TARGET_PLATFORM} MATCHES "mdc-llvm")
    add_definitions(-DBUILD_FOR_MDC=1)
elseif (${TARGET_PLATFORM} MATCHES "j5")
    add_definitions(-DBUILD_FOR_J5=1)
endif()

set(PRJ_ROOT_DIR  ${CMAKE_SOURCE_DIR}/..)
set(NETA_MIDDLEWARE_DIR ${PRJ_ROOT_DIR}/middleware)
set(NETA_THIRDPARTY_DIR ${PRJ_ROOT_DIR}/netaos_thirdparty)


# include("${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/cmake/netaos/neta_codec-config.cmake")
set(CMAKE_FIND_ROOT_PATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/cmake;${CMAKE_FIND_ROOT_PATH}")
find_package(netaos CONFIG REQUIRED)

option(ENABLE_SOMEIP "Enable someip complier")
set(ENABLE_SOMEIP OFF)
# set(ENABLE_SOMEIP ON)

link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib/
)

add_subdirectory(ut)
