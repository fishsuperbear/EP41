

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib;${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib;${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -fPIC)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")


#库名称
set(CLIENT_LIB_NAME neta_cfg)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/middleware/cfg/src/config_param.cpp
        ${CMAKE_SOURCE_DIR}/middleware/cfg/src/config_param_impl.cpp
        ${CMAKE_SOURCE_DIR}/middleware/cfg/src/config_param_impl_proto.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/*.cxx
        ${CMAKE_SOURCE_DIR}/service/config_server/proto/cfg.pb.cc
        )
#默认编译c和c++文件
add_library(${CLIENT_LIB_NAME} SHARED ${SOURCES})

include_directories(
        ${NETA_MIDDLEWARE_DIR}/idl/generated/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include
        ${NETA_MIDDLEWARE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/middleware/cfg
        ${CMAKE_SOURCE_DIR}/middleware/cfg/proto/cfg
        ${CMAKE_SOURCE_DIR}/middleware/cfg/include
        ${CMAKE_SOURCE_DIR}/middleware/phm
        ${CMAKE_SOURCE_DIR}/middleware/phm/include
)

target_link_directories(${CLIENT_LIB_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

target_link_libraries(${CLIENT_LIB_NAME}
        PUBLIC
        neta_log
        # neta_phm
        protobuf
        neta_cm
)

install(TARGETS ${CLIENT_LIB_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${TARGET_PLATFORM}/include/cfg)