cmake_minimum_required(VERSION 3.10.2)

project(cm_protobuf_sample)

# set env before compile
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../netaos_thirdparty/)
set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)

# link_directories(
#     ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/lib
#     ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/lib
#     ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/fast-dds/lib
#     ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/protobuf/lib
#     ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/ffmpeg/lib/
# )

add_executable(proto_send_sample 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf_send_sample.cpp
    ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc)

add_executable(proto_recv_sample 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf_recv_sample.cpp
    ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc)

add_executable(idl_recv_sample 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/idl_recv_sample.cpp
)

target_include_directories(proto_send_sample 
    PRIVATE
    ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${PRJ_ROOT_DIR}
)

target_include_directories(proto_recv_sample
    PRIVATE
    ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
)

target_include_directories(idl_recv_sample 
    PRIVATE
    ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${THIRD_PARTY_DIR}/${TARGET_PLATFORM}/cuda/include
    ${PRJ_ROOT_DIR}
)

target_link_libraries(proto_send_sample
    PRIVATE
    netaos::adf
)

target_link_libraries(proto_recv_sample
    PRIVATE
    netaos::adf
)

target_link_libraries(idl_recv_sample
    PRIVATE
    netaos::adf
)

install(
    TARGETS
    proto_send_sample
    proto_recv_sample
    idl_recv_sample
    RUNTIME
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/adf-proto-sample/bin
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/idl_recv_sample_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/proto_recv_sample_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/proto_send_sample_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf1.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf2.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf3.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf4.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/nvs_cam_sample_conf5.yaml
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/adf-proto-sample/conf
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf_recv_sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf_send_sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/idl_recv_sample.cpp
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/adf-proto-sample/src
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/adf-proto-sample/
)