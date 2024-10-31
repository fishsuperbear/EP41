
cmake_minimum_required(VERSION 3.5.1)

project(hw_platform VERSION 1.0.0)

FIND_INCLUDE_DIR(INCLUDE_VIDEO_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/interface/platform)
FIND_INCLUDE_DIR(INCLUDE_GLOBAL_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/porting/)
FIND_INCLUDE_DIR(INCLUDE_IMPL_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/plat_impl)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")

include_directories(
    ${INCLUDE_VIDEO_DIR_LIST}
    ${INCLUDE_GLOBAL_DIR_LIST}
    ${INCLUDE_IMPL_DIR_LIST}
)

file(GLOB_RECURSE HW_PLATFORM_VS_SRCS
    "${HARDWARE_ROOT_PATH}/hal/code/plat_impl/*.cpp"
    )

#link directorie of lib
link_directories(
    /usr/local/cuda/targets/aarch64-linux/lib
)

add_library(${PROJECT_NAME} SHARED ${HW_PLATFORM_VS_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")

target_link_libraries(${PROJECT_NAME}
    pthread
)


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)
set(SOURCE_DIR ${HARDWARE_ROOT_PATH}/hal/code/interface/platform/)
file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.hpp")
install(FILES  ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/platform/)
set(SOURCE_DIR ${HARDWARE_ROOT_PATH}/hal/code/porting/)
file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.hpp")
install(FILES  ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/porting/)