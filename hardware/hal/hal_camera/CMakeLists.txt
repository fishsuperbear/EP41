
cmake_minimum_required(VERSION 3.5.1)

project(hal_camera VERSION 1.0.0)

FIND_INCLUDE_DIR(INCLUDE_VIDEO_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/interface)
FIND_INCLUDE_DIR(INCLUDE_GLOBAL_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/porting/)
FIND_INCLUDE_DIR(INCLUDE_IMPL_DIR_LIST ${HARDWARE_ROOT_PATH}/hal/code/hal_impl)


set(LIB_DIR ${HARDWARE_ROOT_PATH}/hal/code/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")

include_directories(
    ${HARDWARE_ROOT_PATH}/hal/code/interface/hpp/intf_camera/1.0
    ${INCLUDE_VIDEO_DIR_LIST}
    ${INCLUDE_GLOBAL_DIR_LIST}
    ${INCLUDE_IMPL_DIR_LIST}
)

file(GLOB_RECURSE HW_HAL_VS_STCS
    "${HARDWARE_ROOT_PATH}/hal/code/hal_impl/cpp/impl_camera/*.cpp"
    )

#link directorie of lib
#link_directories(
	#)

add_library(${PROJECT_NAME} SHARED ${HW_HAL_VS_STCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")

target_link_libraries(${PROJECT_NAME}
    hw_platform
    hw_hal
    dl
    pthread
)

set(SOURCE_DIR ${HARDWARE_ROOT_PATH}/hal/code/interface/hpp/intf_camera/1.0)
file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.hpp")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)
install(FILES  ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/camera/)
