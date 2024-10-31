cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "neta_bag")
#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()


# -----------------------------------neta_play.so------------------------------
# 设置共享库目录
set(SHARED_LIB_DIR "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib")

# 查找所有的共享库文件
file(GLOB SHARED_LIBS "${SHARED_LIB_DIR}/*.so")

add_library(neta_play SHARED
    src/player.cpp
    src/impl/player_impl.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/topic_manager.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/adf_lite_utile.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_utility.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/process_utility.cpp
    src/dds_player_publisher.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufTypeObject.cxx
    src/bag_data_pubsubtype.cpp
    src/bag_data_type.cpp
    src/someip_player_publisher.cpp
    src/message_process.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx)

target_include_directories(neta_play
PRIVATE
  ./include
  ./src
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_cpp/include
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcutils/include
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcpputils/include
  ${DATA_TOOLS_ROOT_PATH}/common/proto_factory
  ${DATA_TOOLS_ROOT_PATH}/common/util/include
  ${CMAKE_SOURCE_DIR}
  ${NETA_MIDDLEWARE_DIR}
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
  # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/include/pcl-1.11
  ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/eigen3/include
  # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include
  # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include/boost
  )

target_link_directories(neta_play
PUBLIC
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
  # ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/lib
  # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib
)

target_link_libraries(neta_play
PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
    rosbag2_storage
    rosbag2_cpp
    rcpputils
    neta_cm
    nlog_ics
    ncore_ics
    perftoolkit_ics
    dlt_ics
    ncjson_ics
    someip_ics
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libssl.so.1.1
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libcrypto.so.1.1
    # pcl_io
    # pcl_common
    )
install(TARGETS neta_play
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
  DESTINATION ${TARGET_PLATFORM})


# -----------------------------------neta_bag.so------------------------------
message(STATUS "Configuring bag...")

file(GLOB IDL_SRC
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx

  ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx

  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodPubSubTypes.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_method.cxx


  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufPubSubTypes.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_someipbuf.cxx

  ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx)

add_library(neta_bag SHARED
    src/bag_info.cpp
    src/convert.cpp
    src/impl/convert_impl.cpp
    src/reader.cpp
    src/impl/reader_impl.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_utility.cpp
    src/message_process.cpp
    ${IDL_SRC}
    src/recorder.cpp
    src/impl/recorder_impl.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/topic_manager.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/adf_lite_utile.cpp
    ${DATA_TOOLS_ROOT_PATH}/common/util/src/proto_factory.cpp
    src/dds_recorder_subscriber.cpp
    src/bag_data_pubsubtype.cpp
    src/bag_data_type.cpp
    ${CMAKE_SOURCE_DIR}/proto/test/soc/dbg_msg.pb.cc
    src/h265_play_handler.cpp
    src/save.cpp
    src/impl/save_impl.cpp
    src/writer.cpp
    src/impl/writer_impl.cpp
    src/attachment.cpp
    src/impl/attachment_impl.cpp
    ${DATA_TOOLS_ROOT_PATH}/topic/src/impl/sub_base.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_image.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imageTypeObject.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/zerocopy_imagePubSubTypes.cxx
    src/impl/stat_impl.cpp
    )
target_include_directories(neta_bag
  PRIVATE
    ./include
    ./src
    ${CMAKE_SOURCE_DIR}
    ${DATA_TOOLS_ROOT_PATH}/common
    ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_cpp/include
    ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include
    ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcutils/include
    ${DATA_TOOLS_ROOT_PATH}/common/ros2/rcpputils/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/include
    ${NETA_MIDDLEWARE_DIR}/../
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/include/pcl-1.11
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/eigen3/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/include
    ${DATA_TOOLS_ROOT_PATH}/topic/src/impl/)

  link_directories(neta_bag
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib)

if(${TARGET_PLATFORM} STREQUAL "orin")
  set(NETA_BAG_LIBRARY
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.70.0
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.70.0
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so.1.11.1)
else (${TARGET_PLATFORM} STREQUAL "x86_2004")
  set(NETA_BAG_LIBRARY
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.73.0
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.73.0)
endif()

target_link_libraries(neta_bag
PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib/libpcl_io_ply.so.1.11
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib/libpcl_io.so.1.11.1
    ${NETA_BAG_LIBRARY}
    rosbag2_storage
    rosbag2_cpp
    rcpputils
    neta_cfg
    neta_cm
    neta_codec)

install(TARGETS neta_bag
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)


# -----------------------------------bag bin------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_executable(bag src/bag_command.cpp)
target_include_directories(bag PRIVATE
  ./include
  src/
  ${DATA_TOOLS_ROOT_PATH}/common/argvparser)
target_link_libraries(bag PUBLIC neta_bag neta_play)
install(TARGETS bag RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

# -----------------------------------install hpp------------------------------
install(
  DIRECTORY include/
  DESTINATION ${TARGET_PLATFORM}/include/data_tools/bag)
install(
FILES
  src/bag_message.hpp
  src/bag_data_type.h
  DESTINATION ${TARGET_PLATFORM}/include/data_tools/bag)

install(
FILES
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include/rosbag2_storage/bag_metadata.hpp
  ${DATA_TOOLS_ROOT_PATH}/common/ros2/rosbag2/rosbag2_storage/include/rosbag2_storage/topic_metadata.hpp
DESTINATION ${TARGET_PLATFORM}/include/data_tools/bag/rosbag2_storage)