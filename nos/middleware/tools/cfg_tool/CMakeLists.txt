cmake_minimum_required(VERSION 3.5)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
# 设置模块名
set(MODULE_NAME cfg)

set(CMAKE_INSTALL_RPATH "MIDDLEWARE_LIB_DIR;${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
file(
  GLOB
  SRCS
  "src/*.cpp"
)

add_executable(${MODULE_NAME} ${SRCS} )

# message(STATUS "---------NETA_THIRDPARTY_DIR: ${NETA_THIRDPARTY_DIR}")

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/cfg/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        )

target_link_directories(${MODULE_NAME} PUBLIC 
                        ${MIDDLEWARE_LIB_DIR}
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib
                        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/tinyxml2/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/jsoncpp/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/libuuid-1.0.3/lib
                        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/sqlite/lib
                        )

target_link_libraries(${MODULE_NAME} PRIVATE
    neta_cfg
)

install(TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)