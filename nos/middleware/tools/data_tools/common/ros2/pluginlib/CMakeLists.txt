cmake_minimum_required(VERSION 3.5)
set(MODULE_NAME "pluginlib")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

add_library(${MODULE_NAME} INTERFACE)
target_include_directories(${MODULE_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${MODULE_NAME}>"
  ${DATA_TOOLS_COMMON_PATH}/rcpputils/include
  ${NETA_MIDDLEWARE_DIR}/
  )
target_link_libraries(${MODULE_NAME} INTERFACE
  rcutils
  rcpputils
  neta_log
  )