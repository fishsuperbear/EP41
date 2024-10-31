get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME ${CURRENT_DIR_NAME})

# setup cuda
set(CMAKE_CUDA_COMPILER ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -arch=sm_87 -Xcompiler -fPIC" CACHE STRING "" FORCE)
enable_language(CUDA)

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

link_directories(
)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/service/camera_venc/src
    ${PRJ_ROOT_DIR}/service/camera_venc/src/orin
    ${NETA_MIDDLEWARE_DIR}
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

target_link_libraries(${MODULE_NAME}
    gtest
    gtest_main
    pthread
    netaos::desen
)

#用户可在此添加链接参数
# set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/${MAIN_DIR_NAME}/${CURRENT_DIR_NAME}/bin
)

