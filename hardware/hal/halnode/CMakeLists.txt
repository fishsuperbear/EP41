cmake_minimum_required(VERSION 3.5)

project(halnode)
include_directories(
    ../code/interface/halnode
)
file(GLOB_RECURSE SOURCES "${HARDWARE_ROOT_PATH}/hal/code/halnode_impl/*.cpp" )
add_library(${PROJECT_NAME} SHARED ${SOURCES})
link_directories(
	/usr/local/cuda/targets/aarch64-linux/lib
	)

set(SOURCE_DIR ${HARDWARE_ROOT_PATH}/hal/code/interface/halnode)
file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.hpp")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)
# install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/halnode)