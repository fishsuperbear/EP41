cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib;${PROJECT_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib;${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
include(${PROJECT_SOURCE_DIR}/../netaos_thirdparty/cmake/Packages.cmake)
set(3RD_ROOT "${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}")
set(CMAKE_FIND_ROOT_PATH "${PROJECT_SOURCE_DIR}/../netaos_thirdparty/cmake;${CMAKE_FIND_ROOT_PATH}")
find_package(Protobuf MODULE REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
# 设置模块名
set(MODULE_NAME per_test)
# 设置中间件头文件路径和lib路径

file(
  GLOB
  SRCS
  "src/*.cpp"
)

add_executable(${MODULE_NAME} ${SRCS} )
target_include_directories(${MODULE_NAME}
  PRIVATE
   ${PROJECT_SOURCE_DIR}/../output/${TARGET_PLATFORM}/include/per
   ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
   ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
)

target_link_directories(${MODULE_NAME} PUBLIC   ${PROJECT_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib 
                        ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
                        ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
)
target_link_libraries(${MODULE_NAME} PUBLIC
  pthread
  neta_per
  netaos::log
  ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so
  ${PROJECT_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/libprotobuf.so
)


install(TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)