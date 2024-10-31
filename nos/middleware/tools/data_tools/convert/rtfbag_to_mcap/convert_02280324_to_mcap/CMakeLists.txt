cmake_minimum_required(VERSION 3.12)
# project(rtfbag-0228-0324_mcap)
set(MODULE_NAME "rtfbag-0228-0324_mcap")
set(DEFAULT_COMPILE -fPIC)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")

link_directories(
    ${RTF_TO_MCAP_ROOT_PATH}/MDC_Ubuntu_X86/mdc_platform/lib
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/lib
    )

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)
add_library(${MODULE_NAME} SHARED
  convert_02280324_to_mcap.cpp
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
  ${PROTO_SRC})

target_include_directories(${MODULE_NAME}
  PUBLIC
  ${RTF_TO_MCAP_ROOT_PATH}/include/0228_0324/includes
  ${RTF_TO_MCAP_ROOT_PATH}/MDC_Ubuntu_X86/mdc_platform/include/
  ${NETA_MIDDLEWARE_DIR}/
  ${DATA_TOOLS_ROOT_PATH}/common/
  ${DATA_TOOLS_COMMON_PATH}/util/include
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcpputils/include/
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_cpp/include/
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include/
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcutils/include/
  ${DATA_TOOLS_ROOT_PATH}/bag/src/
  ${DATA_TOOLS_ROOT_PATH}/bag/include/
  ${NETA_MIDDLEWARE_DIR}/../
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include/
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include/
  )

TARGET_LINK_LIBRARIES(${MODULE_NAME}
                    securec rtftools logging ara_core ara_com ament_index_cpp neta_log protobuf)

install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${TARGET_PLATFORM}/conf/bag/convert/rtfbag_to_mcap)
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${TARGET_PLATFORM}/lib)


