cmake_minimum_required(VERSION 3.12)

set(MODULE_NAME "rcutils")

# Default to C11
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

include(CheckLibraryExists)

find_package(Python3 REQUIRED COMPONENTS Interpreter)


if(UNIX AND NOT APPLE)
  include(cmake/check_c_compiler_uses_glibc.cmake)
  check_c_compiler_uses_glibc(USES_GLIBC)
  if(USES_GLIBC)
    # Ensure GNU extended libc API is used, as C++ test code will.
    # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=2082.
    add_definitions(-D_GNU_SOURCE)
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # enables building a static library but later link it into a dynamic library
  add_compile_options(-fPIC)
endif()
if(NOT WIN32)
  # About -Wno-sign-conversion: With Clang, -Wconversion implies -Wsign-conversion. There are a number of
  # implicit sign conversions in gtest.cc, see https://ci.ros2.org/job/ci_osx/9381/clang/.
  # Hence disabling -Wsign-conversion for now until all those have eventually been fixed.
  # (from https://github.com/ros2/rcutils/pull/263#issuecomment-663252537)
  add_compile_options(-Wall -Wextra -Wconversion -Wno-sign-conversion -Wpedantic)
endif()

if(WIN32)
  set(time_impl_c src/time_win32.c)
else()
  set(time_impl_c src/time_unix.c)
endif()

set(rcutils_sources
  src/allocator.c
  src/array_list.c
  src/char_array.c
  src/cmdline_parser.c
  src/env.c
  src/error_handling.c
  src/filesystem.c
  src/find.c
  src/format_string.c
  src/hash_map.c
  src/logging.c
  src/process.c
  src/qsort.c
  src/repl_str.c
  src/shared_library.c
  src/snprintf.c
  src/split.c
  src/strcasecmp.c
  src/strdup.c
  src/strerror.c
  src/string_array.c
  src/string_map.c
  src/testing/fault_injection.c
  src/time.c
  ${time_impl_c}
  src/uint8_array.c
)
set_source_files_properties(
  ${rcutils_sources}
  PROPERTIES language "C")

# "watch" template/inputs for changes, 不需要每次都重新生成
# configure_file(
#   "resource/logging_macros.h.em"
#   "logging_macros.h.em.watch"
#   COPYONLY)
# configure_file(
#   "rcutils/logging.py"
#   "logging.py.watch"
#   COPYONLY)
# # generate header with logging macros
# set(rcutils_module_path ${CMAKE_CURRENT_SOURCE_DIR})
# set(python_code
#   "import em"  # implicitly added ; between python statements due to CMake list
#   "\
# em.invoke( \
#   [ \
#     '-o', 'include/rcutils/logging_macros.h', \
#     '-D', 'rcutils_module_path=\"${rcutils_module_path}\"', \
#     '${CMAKE_CURRENT_SOURCE_DIR}/resource/logging_macros.h.em' \
#   ] \
# )")
# string(REPLACE ";" "$<SEMICOLON>" python_code "${python_code}")

# add_custom_command(OUTPUT include/rcutils/logging_macros.h
#   COMMAND ${CMAKE_COMMAND} -E make_directory "include/rcutils"
#   COMMAND Python3::Interpreter ARGS -B -c "${python_code}"
#   DEPENDS
#     "${CMAKE_CURRENT_BINARY_DIR}/logging_macros.h.em.watch"
#     "${CMAKE_CURRENT_BINARY_DIR}/logging.py.watch"
#   COMMENT "Expanding logging_macros.h.em"
#   VERBATIM
# )
list(APPEND rcutils_sources
  include/rcutils/logging_macros.h)

add_library(
  ${MODULE_NAME} SHARED
  ${rcutils_sources})

target_include_directories(${MODULE_NAME} PUBLIC
  ${DATA_TOOLS_COMMON_PATH}/rcpputils/include
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${MODULE_NAME}>")

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(${MODULE_NAME} PRIVATE "RCUTILS_BUILDING_DLL")

if(BUILD_TESTING AND NOT RCUTILS_DISABLE_FAULT_INJECTION)
  target_compile_definitions(${MODULE_NAME} PUBLIC RCUTILS_ENABLE_FAULT_INJECTION)
endif()

target_link_libraries(${MODULE_NAME} ${CMAKE_DL_LIBS})

# Needed if pthread is used for thread local storage.
if(IOS AND IOS_SDK_VERSION LESS 10.0)
  ament_export_libraries(pthread)
endif()

install(TARGETS ${MODULE_NAME} EXPORT ${MODULE_NAME}
  ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
  LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
  RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
