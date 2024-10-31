
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "camera_internal_sample")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/camera_internal_data*.cxx)

add_executable(${MODULE_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}
    PRIVATE
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
)

target_link_libraries(${MODULE_NAME}
    netaos::cm
    netaos::log
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
