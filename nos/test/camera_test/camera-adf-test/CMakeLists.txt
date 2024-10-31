cmake_minimum_required(VERSION 3.10.2)

set(MODULE_NAME "camera_adf_test")

add_executable(${MODULE_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_test.cpp
)

target_include_directories(${MODULE_NAME}
    PRIVATE
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
)

target_link_libraries(${MODULE_NAME}
    PRIVATE
    netaos::adf
)

install(
    TARGETS
    ${MODULE_NAME}
    RUNTIME
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/bin
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/h265_cam_proto_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/replay_cam_cuda_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf1.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf2.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf3.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf4.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf5.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/env.sh
    DESTINATION ${CAMERA_TEST_DIR}/${MODULE_NAME}/conf
)