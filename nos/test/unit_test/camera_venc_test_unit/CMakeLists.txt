
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

get_filename_component(MAIN_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
)

#用户可在此添加头文件路径
include_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include
    ${NETA_MIDDLEWARE_DIR}/sensor/multicast
)

set(CMAKE_SKIP_RPATH  TRUE)

file(GLOB SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(DIR ${SUB_DIRS})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIR})
    add_subdirectory(${DIR})
  endif()
endforeach()

