cmake_minimum_required(VERSION 3.5)
set(MODULE_NAME "rosbag2_cpp")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
set(CMAKE_CXX_STANDARD 14)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wthread-safety)
endif()

# Windows supplies macros for min and max by default. We should only use min and max from stl
if(WIN32)
  add_definitions(-DNOMINMAX)
endif()

option(DISABLE_SANITIZERS "disables the use of gcc sanitizers" ON)
if(NOT DISABLE_SANITIZERS AND CMAKE_COMPILER_IS_GNUCXX)
  include(CheckCXXSourceRuns)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS} -fsanitize=leak")
  check_cxx_source_runs("int main() {}" HAVE_SANITIZERS)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if(NOT HAVE_SANITIZERS)
    set(DISABLE_SANITIZERS ON)
    message(WARNING "Sanitizers aren't supported by the compiler or environment - disabling")
  endif()
endif()


add_library(${MODULE_NAME} SHARED
  src/rosbag2_cpp/cache/cache_consumer.cpp
  src/rosbag2_cpp/cache/message_cache_buffer.cpp
  src/rosbag2_cpp/cache/message_cache_circular_buffer.cpp
  src/rosbag2_cpp/cache/message_cache.cpp
  src/rosbag2_cpp/cache/circular_message_cache.cpp
  src/rosbag2_cpp/clocks/time_controller_clock.cpp
  src/rosbag2_cpp/converter.cpp
  src/rosbag2_cpp/info.cpp
  src/rosbag2_cpp/reader.cpp
  src/rosbag2_cpp/readers/sequential_reader.cpp
  src/rosbag2_cpp/rmw_implemented_serialization_format_converter.cpp
  src/rosbag2_cpp/serialization_format_converter_factory.cpp
  src/rosbag2_cpp/types/introspection_message.cpp
  src/rosbag2_cpp/typesupport_helpers.cpp
  src/rosbag2_cpp/types/introspection_message.cpp
  src/rosbag2_cpp/writer.cpp
  src/rosbag2_cpp/writers/sequential_writer.cpp
  src/rosbag2_cpp/reindexer.cpp)

target_link_libraries(${MODULE_NAME}
  class_loader
  ament_index_cpp
  rosbag2_storage
  neta_log
  )

target_include_directories(${MODULE_NAME}
  PUBLIC
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
  # ${DATA_TOOLS_COMMON_PATH}/rosbag2/rosbag2_storage/include
  # ${DATA_TOOLS_COMMON_PATH}/rcpputils/include
  # ${CMAKE_CURRENT_BINARY_DIR}/../../rcutils/include
  # ${DATA_TOOLS_COMMON_PATH}/rcutils/include
  # ${DATA_TOOLS_COMMON_PATH}/ament_index_cpp/include
  # ${DATA_TOOLS_COMMON_PATH}/pluginlib/pluginlib/include
  # ${DATA_TOOLS_COMMON_PATH}/class_loader/include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MODULE_NAME}>
  ${NETA_MIDDLEWARE_DIR}/
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${MODULE_NAME} PRIVATE "ROSBAG2_CPP_BUILDING_DLL")

install(
  TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
