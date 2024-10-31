cmake_minimum_required(VERSION 3.5)

set(MODULE_NAME "rcpputils")

# Default to C11
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # enables building a static library but later link it into a dynamic library
  add_compile_options(-fPIC)
endif()
if(NOT WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(${MODULE_NAME} SHARED
  src/asserts.cpp
  src/filesystem_helper.cpp
  src/find_library.cpp
  src/env.cpp
  src/shared_library.cpp)


target_link_libraries(${MODULE_NAME}
  PUBLIC
  rcutils
)

target_include_directories(${MODULE_NAME} PUBLIC
  ${DATA_TOOLS_COMMON_PATH}/rcutils/include/
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${MODULE_NAME}>")
if(WIN32)
  target_compile_definitions(${MODULE_NAME}
    PRIVATE "RCPPUTILS_BUILDING_LIBRARY")
endif()

install(
  TARGETS ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin
)