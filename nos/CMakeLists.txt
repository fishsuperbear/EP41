cmake_minimum_required(VERSION 3.10)

project(netaos)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC --std=gnu++14 -O3 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC --std=gnu++14 -O3 -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    set(PLATFORM_3RD "${CMAKE_SOURCE_DIR}/netaos_thirdparty/x86_2004")
elseif (${TARGET_PLATFORM} MATCHES "mdc")
    add_definitions(-DBUILD_FOR_MDC=1)
    set(PLATFORM_3RD "${CMAKE_SOURCE_DIR}/netaos_thirdparty/arm")
elseif (${TARGET_PLATFORM} MATCHES "mdc-llvm")
    add_definitions(-DBUILD_FOR_MDC=1)
    set(PLATFORM_3RD "${CMAKE_SOURCE_DIR}/netaos_thirdparty/mdc-llvm")
elseif (${TARGET_PLATFORM} MATCHES "j5")
    add_definitions(-DBUILD_FOR_J5=1)
    set(PLATFORM_3RD "${CMAKE_SOURCE_DIR}/netaos_thirdparty/j5")
elseif (${TARGET_PLATFORM} MATCHES "orin")
    add_definitions(-DBUILD_FOR_ORIN=1)
    set(PLATFORM_3RD "${CMAKE_SOURCE_DIR}/netaos_thirdparty/orin")
endif()

# message(FATAL_ERROR "the 3rd platform is ${PLATFORM_3RD}")

option(ENABLE_SOMEIP "Enable someip complier")
set(ENABLE_SOMEIP OFF)
# set(ENABLE_SOMEIP ON)

# 默认关闭Log_server压缩的方式，调测不方便，发版时打开
option(ENABLE_LOG_SERVER "Enable log_server")
# set(ENABLE_LOG_SERVER OFF)
set(ENABLE_LOG_SERVER ON)

option(ENABLE_IPC "Enable cpp-ipc")
set(ENABLE_IPC OFF)
# set(ENABLE_IPC ON)

if(ENABLE_IPC)
    add_definitions(-DBUILD_IPC_ENABLE)
endif()

set(NETA_MIDDLEWARE_DIR ${CMAKE_SOURCE_DIR}/middleware)
set(NETA_SERVICE_DIR ${CMAKE_SOURCE_DIR}/service)
set(NETA_THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/netaos_thirdparty)

if (${BUILD_WITH_IDL} MATCHES "true")
    set(BUILD_BY_ADF "true")
    add_subdirectory(proto)
    add_subdirectory(middleware/idl)
    add_subdirectory(service/idl)
endif()

add_subdirectory(middleware)
add_subdirectory(service)
add_subdirectory(scripts)
add_subdirectory(tools)
add_subdirectory(hz_bsp)
