cmake_minimum_required(VERSION 3.5)

set(MODULE_NAME "ament_index_cpp")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)
endif()

add_library(${MODULE_NAME} SHARED
  src/get_package_prefix.cpp
  src/get_search_paths.cpp
  src/has_resource.cpp
)

target_compile_definitions(${MODULE_NAME} PRIVATE "AMENT_INDEX_CPP_BUILDING_DLL")
target_compile_definitions(${MODULE_NAME} PRIVATE -DDEFAULT_PREFIX_PATH="${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag")
target_include_directories(${MODULE_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${MODULE_NAME}>")

install(TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)