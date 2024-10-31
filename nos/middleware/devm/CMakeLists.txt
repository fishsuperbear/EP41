#module名称
set(MODULE_NAME "neta_devm_client")
set(CLIENT_LIB_NAME neta_devm_client)

#add_definitions("-D EM_DEBUG_ON")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -fPIC)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    )

add_library(${CLIENT_LIB_NAME} SHARED ${SOURCES})

target_include_directories(${CLIENT_LIB_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/devm/include
        ${CMAKE_SOURCE_DIR}/middleware/devm/include/common
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
)

target_link_directories(${CLIENT_LIB_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
        ${CMAKE_SOURCE_DIR}/middleware
)

target_link_libraries(${CLIENT_LIB_NAME}
        PUBLIC
        pthread
        neta_log
        neta_cfg
        neta_zmq_ipc
)

install(TARGETS ${CLIENT_LIB_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${TARGET_PLATFORM}/include/devm)