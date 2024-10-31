cmake_minimum_required(VERSION 3.5)
project(mcap LANGUAGES C CXX ASM)

# Avoid DOWNLOAD_EXTRACT_TIMESTAMP warning for CMake >= 3.24
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

## Compile options
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=unused-variable)
endif()
if (MSVC)
  add_compile_options(/W4 /WX
    /wd4251 # suppress warning about having template instances (such as std::string) as public class members
  )
  # suppress error disallowing `fopen` in favor of `fopen_s`
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

## Define vendor macro
macro(build_mcap_vendor)
  include(FetchContent)
  #下载完源码就注释掉
  # fetchcontent_declare(mcap
  #   URL https://github.com/foxglove/mcap/archive/refs/tags/releases/cpp/v0.8.0.tar.gz
  #   URL_HASH SHA1=b44637791da2c9c1cec61a3ba6994f1ef63a228c # v0.8.0
  # )
  # message(WARNING "download mcap start2")
  # fetchcontent_makeavailable(mcap)
  # message(WARNING "download lz4 start")
  # fetchcontent_declare(lz4
  #   GIT_REPOSITORY https://github.com/lz4/lz4.git
  #   GIT_TAG d44371841a2f1728a3f36839fd4b7e872d0927d3 # v1.9.3
  # )
  # fetchcontent_makeavailable(lz4)

  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
set(lz4_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/_deps/lz4-src)
set(mcap_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/_deps/mcap-src)

  file(GLOB _lz4_srcs
    ${lz4_SOURCE_DIR}/lib/*.c)

  add_library(mcap SHARED
    src/main.cpp
    ${_lz4_srcs}
  )

  set(_mcap_include_dir ${mcap_SOURCE_DIR}/cpp/mcap/include)
  file(GLOB _mcap_installed_headers ${_mcap_include_dir}/mcap/*.hpp)

  target_include_directories(mcap PRIVATE
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zstd/include/
    ${lz4_SOURCE_DIR}/lib
  )

  target_include_directories(mcap PUBLIC

    "$<BUILD_INTERFACE:${_mcap_include_dir}>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
  )
  target_link_libraries(mcap)

  set_target_properties(mcap PROPERTIES LINK_FLAGS "-Wl,--as-needed")

  install(
    FILES ${_mcap_installed_headers}
    DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/mcap
  )

  install(
    TARGETS mcap
    EXPORT mcap
    ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
    LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin
  )

endmacro()

## Call vendor macro
build_mcap_vendor()
