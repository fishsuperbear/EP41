cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "proto_idl_util")
#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()


add_library(MODULE_NAME SHARED
    # src/player.cpp
    # src/impl/player_impl.cpp
    # ${DATA_TOOLS_ROOT_PATH}/common/util/src/topic_manager.cpp
    # ${DATA_TOOLS_ROOT_PATH}/common/util/src/adf_lite_utile.cpp
    # ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_utility.cpp
    # ${DATA_TOOLS_ROOT_PATH}/common/util/src/thread_utility.cpp
    # src/dds_player_publisher.cpp
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufTypeObject.cxx
    # src/bag_data_pubsubtype.cpp
    # src/bag_data_type.cpp
    # src/someip_player_publisher.cpp
    # src/message_process.cpp
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
    # ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx
    src/proto_idl_util.cpp
    )

target_include_directories(MODULE_NAME
PRIVATE
  ./include
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
  ${NETA_MIDDLEWARE_DIR}/
  ${DATA_TOOLS_COMMON_PATH}/ros2/ament_index_cpp/include

  # ./src
  # ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_cpp/include
  # ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include
  # ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcutils/include
  # ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcpputils/include
  # ${DATA_TOOLS_ROOT_PATH}/common/proto_factory
  # ${DATA_TOOLS_ROOT_PATH}/common/util/include
  # ${CMAKE_SOURCE_DIR}
  # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
  # # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/include/pcl-1.11
  # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/eigen3/include
  # # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include
  # # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include/boost
  )

# target_link_directories(MODULE_NAME
# PUBLIC
#   ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
#   # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/lib
#   # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib
# )

target_link_libraries(MODULE_NAME
PUBLIC
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
    # rosbag2_storage
    # rosbag2_cpp
    # rcpputils
    # neta_cm
    # nlog_ics
    # ncore_ics
    # perftoolkit_ics
    # dlt_ics
    # ncjson_ics
    # someip_ics
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libssl.so.1.1
    # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libcrypto.so.1.1
    # pcl_io
    # pcl_common
    )
install(TARGETS MODULE_NAME
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)


# -----------------------------------install hpp------------------------------
install(
  FILES include/proto_idl_util.h
  DESTINATION ${TARGET_PLATFORM}/include/data_tools/bag)
