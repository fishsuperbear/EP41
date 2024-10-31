
cmake_minimum_required(VERSION 3.5)

set(MODULE_NAME "cfg_app")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)


file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${MODULE_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}
        PUBLIC
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/cfg
)

target_link_libraries(${MODULE_NAME}
        pthread
        netaos::log
        netaos::cfg
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
