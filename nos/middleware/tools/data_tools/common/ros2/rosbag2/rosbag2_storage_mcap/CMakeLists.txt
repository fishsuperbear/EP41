cmake_minimum_required(VERSION 3.14)
set(MODULE_NAME "rosbag2_storage_mcap")

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
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Get the ROS_DISTRO environment variable
set(ROS_DISTRO $ENV{ROS_DISTRO})

add_library(${MODULE_NAME} SHARED
  src/mcap_storage.cpp
  src/message_definition_cache.cpp
)

target_include_directories(${MODULE_NAME} PUBLIC
  ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
  ${DATA_TOOLS_COMMON_PATH}/
  ${DATA_TOOLS_COMMON_PATH}/util/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/rosbag2/rosbag2_storage/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/ament_index_cpp/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/pluginlib/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/class_loader/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/rcpputils/include
  ${DATA_TOOLS_COMMON_PATH}/rcutils/include
  ${DATA_TOOLS_COMMON_PATH}/mcap/_deps/mcap-src/cpp/mcap/include/mcap
  ${CMAKE_CURRENT_BINARY_DIR}/../../rcutils/include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MODULE_NAME}>
  ${NETA_MIDDLEWARE_DIR}/
)
target_compile_features(${MODULE_NAME} PUBLIC c_std_99 cxx_std_17)
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_STORAGE_MCAP_BUILDING_DLL")

target_link_libraries(${MODULE_NAME}
  rcutils
  mcap
  # ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/mcap/lib/libmcap.so
)

set(MCAP_COMPILE_DEFS)

# COMPATIBILITY(foxy) - 0.3.x is the Foxy release
if(${rosbag2_storage_VERSION} VERSION_GREATER_EQUAL 0.4.0)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_HAS_STORAGE_OPTIONS)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_WRITER_CREATES_DIRECTORY)
endif()
# COMPATIBILITY(galactic) - 0.9.x is the Galactic release
if(${rosbag2_storage_VERSION} VERSION_GREATER_EQUAL 0.10.0)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_OVERRIDE_SEEK_METHOD)
endif()
# COMPATIBILITY(foxy, galactic) - 0.15.x is the Humble release
if(${rosbag2_storage_VERSION} VERSION_GREATER_EQUAL 0.15.0)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_HAS_YAML_HPP)
endif()
# COMPATIBILITY(foxy, galactic, humble, rolling:0.17.x)
if(${rosbag2_storage_VERSION} VERSION_GREATER_EQUAL 0.17.0)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_HAS_STORAGE_FILTER_TOPIC_REGEX)
endif()
# COMPATIBILITY(foxy, galactic, humble, rolling:0.17.x, rolling:0.18.x)
if(${rosbag2_storage_VERSION} VERSION_GREATER_EQUAL 0.18.0)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_HAS_SET_READ_ORDER)
  list(APPEND MCAP_COMPILE_DEFS ROSBAG2_STORAGE_MCAP_HAS_UPDATE_METADATA)
endif()

target_compile_definitions(${MODULE_NAME} PRIVATE ${MCAP_COMPILE_DEFS})

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_STORAGE_MCAP_BUILDING_LIBRARY")

# generate plugin file
include(${DATA_TOOLS_COMMON_PATH}/ros2/pluginlib/cmake/pluginlib_export_plugin_description_file.cmake)
pluginlib_export_plugin_description_file(rosbag2_storage plugin_description_mcap.xml)

install(
  TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/conf/bag/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/conf/bag/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin
)
