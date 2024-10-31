cmake_minimum_required(VERSION 3.16.3)

set(MODULE_NAME "tsync")
#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()

set(DATA_TOOLS_ROOT_PATH ${CMAKE_SOURCE_DIR}/middleware/tools/data_tools)
add_executable(${MODULE_NAME} src/tsync_command.cpp)
target_include_directories(${MODULE_NAME}
  PRIVATE
    # ${DATA_TOOLS_ROOT_PATH}/common/
    ${DATA_TOOLS_ROOT_PATH}/common/util/include
    ${CMAKE_SOURCE_DIR}/middleware
    )

target_link_libraries(${MODULE_NAME}
    PRIVATE
    neta_cfg)

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
