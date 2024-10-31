cmake_minimum_required(VERSION 3.14)
set(MODULE_NAME "rosbag2_storage_record")

# Set Release build if no build type was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type for the build. Possible values are: Debug, Release, RelWithDebInfo, MinSizeRel"
      FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Enable additional warnings and warnings as errors
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-error=unused-parameter)
endif()

# Get the ROS_DISTRO environment variable
set(ROS_DISTRO $ENV{ROS_DISTRO})

file(GLOB_RECURSE
    RECORD_FILE
    src/cyber/record/*.cc)

add_library(${MODULE_NAME} SHARED
# add_executable(${MODULE_NAME}
  src/record_storage.cpp
  src/reord_proto_factory.cpp
  ${DATA_TOOLS_COMMON_PATH}/util/src/proto_factory.cpp
  ${RECORD_FILE}
  src/cyber/proto/proto_desc.pb.cc
  src/cyber/proto/record.pb.cc
  src/cyber/common/file.cc

  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx

  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodPubSubTypes.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_methodTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/proto_method.cxx

  ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
)

target_include_directories(${MODULE_NAME} PUBLIC
  ./src
  ./include
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include/
  ${NETA_MIDDLEWARE_DIR}/
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include/
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
  ${DATA_TOOLS_COMMON_PATH}/rosbag2/rosbag2_storage/include
  ${DATA_TOOLS_COMMON_PATH}/ament_index_cpp/include
  ${DATA_TOOLS_COMMON_PATH}/pluginlib/include
  ${DATA_TOOLS_COMMON_PATH}/class_loader/include
  ${DATA_TOOLS_COMMON_PATH}/rcpputils/include
  ${DATA_TOOLS_COMMON_PATH}/rcutils/include
  ${NETA_MIDDLEWARE_DIR}/
)
target_compile_features(${MODULE_NAME} PUBLIC c_std_99 cxx_std_17)
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_STORAGE_RECORD_BUILDING_DLL")

target_link_libraries(${MODULE_NAME}
  rcutils
  neta_log
  # class_loader
  ament_index_cpp
  rosbag2_storage
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfoonathan_memory-0.7.1.so
)

set(RECORD_COMPILE_DEFS)

target_compile_definitions(${MODULE_NAME} PRIVATE ${RECORD_COMPILE_DEFS})

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_STORAGE_RECORD_BUILDING_LIBRARY")

# generate plugin file
include(${DATA_TOOLS_COMMON_PATH}/ros2/pluginlib/cmake/pluginlib_export_plugin_description_file.cmake)
pluginlib_export_plugin_description_file(rosbag2_storage plugin_description_record.xml)

install(
  TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/conf/bag/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/conf/bag/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin
)

# -----------------------------------install  dependent so------------------------------
# file(GLOB_RECURSE
#     LIBFASTRTPS_SO
#     ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fastrtps/lib/libfastrtps.so*)
# install(
#   FILES ${LIBFASTRTPS_SO}
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)