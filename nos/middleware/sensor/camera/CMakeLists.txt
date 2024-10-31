#module名称
set(MODULE_NAME "neta_camera")

#lib名称
set(LIB_NAME neta_camera)

link_directories(
    ${COMPILER_PATH}/aarch64-buildroot-linux-gnu/sysroot/lib/
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/aarch64-buildroot-linux-gnu/sysroot/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${NETA_MIDDLEWARE_DIR}/
)

target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        nvsipl
        nvsipl_query
        nvscibuf
        nvscisync
        neta_log
)

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/nv_camera.hpp
        DESTINATION ${TARGET_PLATFORM}/include/sensor/)

add_executable(camera_service
    service/camera_service.cpp
    service/nvs_sender.cpp
    src/nv_camera_impl.cpp
    src/cam_utils.cpp
    src/image_manager.cpp
)

target_include_directories(camera_service PUBLIC
    ${NETA_MIDDLEWARE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${COMPILER_PATH}/aarch64-buildroot-linux-gnu/sysroot/include/
)

target_link_directories(camera_service PUBLIC
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
)

target_link_libraries(camera_service
    nvs
    nvsipl
    nvsipl_query
)

install(TARGETS camera_service RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin)

