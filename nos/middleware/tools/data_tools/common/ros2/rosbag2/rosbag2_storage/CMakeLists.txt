cmake_minimum_required(VERSION 3.5)
set(MODULE_NAME "rosbag2_storage")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Windows supplies macros for min and max by default. We should only use min and max from stl
if(WIN32)
  add_definitions(-DNOMINMAX)
endif()

add_library(
  ${MODULE_NAME}
  SHARED
  src/rosbag2_storage/metadata_io.cpp
  src/rosbag2_storage/ros_helper.cpp
  src/rosbag2_storage/storage_factory.cpp
  src/rosbag2_storage/storage_options.cpp
  src/rosbag2_storage/base_io_interface.cpp)

target_link_libraries(${MODULE_NAME}
  rcutils
  rcpputils
  class_loader
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib/libyaml-cpp.so
  neta_log
  )

target_include_directories(${MODULE_NAME}
  PUBLIC
  ${DATA_TOOLS_COMMON_PATH}/ros2/rcutils/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/rcpputils/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/pluginlib/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/class_loader/include
  ${DATA_TOOLS_COMMON_PATH}/ros2/ament_index_cpp/include
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MODULE_NAME}>
  ${NETA_MIDDLEWARE_DIR}/
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_STORAGE_BUILDING_DLL")

install(
  TARGETS ${MODULE_NAME}
  EXPORT export_${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)