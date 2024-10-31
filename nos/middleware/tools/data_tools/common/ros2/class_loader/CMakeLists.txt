cmake_minimum_required(VERSION 3.5)

set(MODULE_NAME "class_loader")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CLASS_LOADER_IGNORE_AMENT FALSE CACHE BOOL
  "Do not use ament when building this package.")
if(NOT CLASS_LOADER_IGNORE_AMENT)
  set(explicit_library_type "")
else()
  set(explicit_library_type "SHARED")
endif()


set(${MODULE_NAME}_SRCS
  src/class_loader.cpp
  src/class_loader_core.cpp
  src/meta_object.cpp
  src/multi_library_class_loader.cpp
)
add_library(${MODULE_NAME} SHARED ${explicit_library_type} ${${MODULE_NAME}_SRCS})
target_include_directories(${MODULE_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${MODULE_NAME}>")

target_include_directories(${MODULE_NAME}
  PUBLIC
  ${DATA_TOOLS_COMMON_PATH}/
  ${DATA_TOOLS_COMMON_PATH}/util/include
  ${DATA_TOOLS_COMMON_PATH}/rcutils/include/
  ${DATA_TOOLS_COMMON_PATH}/ros2/rcpputils/include/
  ${NETA_MIDDLEWARE_DIR}/
  )

target_link_libraries(${MODULE_NAME}
  PUBLIC
  rcutils
  rcpputils
  neta_log
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/${MODULE_NAME}.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pc"
  @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pc" DESTINATION ${TARGET_PLATFORM}/lib/pkgconfig)

install(TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
