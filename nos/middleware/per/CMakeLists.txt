
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # 设置 protobuf-config.cmake查找路径。CONFIG模式。
# # set(CMAKE_FIND_ROOT_PATH "${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf;${CMAKE_FIND_ROOT_PATH}")
# set(3RD_ROOT "${CMAKE_CURRENT_LIST_DIR}/../x86")
# set(CMAKE_FIND_ROOT_PATH "${NETA_THIRDPARTY_DIR}/cmake;${CMAKE_FIND_ROOT_PATH}")
# # 强制使用CONFIG模式。protobuf使用到了absl，也需要find。
# # find_package(absl CONFIG REQUIRED)

# # 设置指定protoc路径。非系统目录路径。
# # set(protobuf_PROTOC_EXE "${NETA_THIRDPARTY_DIR}/x86-2004/protobuf/bin/protoc")

# add_library(per-proto-objects OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/proto/per_key_value.proto")
# # target_link_libraries(per-proto-objects PUBLIC protobuf::libprotobuf 
# target_link_libraries(per-proto-objects PUBLIC ${PROTOBUF_LIBRARIES} )

# 在指定目录下生成代码
# protobuf_generate(
#     TARGET per-proto-objects
#     IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/proto"
#     PROTOC_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto" 
#     )
# include(${NETA_THIRDPARTY_DIR}/cmake/Packages.cmake)
# find_package(Protobuf REQUIRED MODULE)
# include_directories(${PROTOBUF_INCLUDE_DIRS})
# include_directories(${CMAKE_CURRENT_BINARY_DIR})
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/proto/per_key_value.proto)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib;${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib;")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
include(${CMAKE_SOURCE_DIR}/netaos_thirdparty/cmake/Packages.cmake)
set(3RD_ROOT "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}")
find_package(Protobuf MODULE REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(MODULE_NAME neta_per)
file(
  GLOB
  SRCS
  "src/*.cpp"
  ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/src/json/*.cpp
  ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/src/protobuf/*.cpp
  ${CMAKE_SOURCE_DIR}/middleware/per/proto/*.cc
  ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/src/thirdParty/json/*.cpp
)

# 添加可执行 target
add_library(${MODULE_NAME} SHARED ${SRCS} ${PROTO_SRCS})

target_include_directories(${MODULE_NAME}
  PRIVATE
  ${CMAKE_SOURCE_DIR}/middleware
  ${CMAKE_SOURCE_DIR}/middleware/per
  ${CMAKE_SOURCE_DIR}/middleware/per/include
  ${CMAKE_SOURCE_DIR}/middleware/per/proto
  ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/include
  ${CMAKE_SOURCE_DIR}/middleware/per/struct2x
  ${CMAKE_SOURCE_DIR}/middleware/log/include
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
  ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
  ${NETA_MIDDLEWARE_DIR}
)

# target_include_directories(${MODULE_NAME} PRIVATE include proto   ${NETA_MIDDLEWARE_DIR} )
target_link_directories(${MODULE_NAME}
                        PUBLIC
                        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib 
                        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
                        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
                        )
target_link_libraries(${MODULE_NAME}
                        PUBLIC
                        pthread
                        jsoncpp
                        protobuf
                        neta_log
                        ) 

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)
install(FILES   
        proto/per_key_value.pb.h
        include/basic_operations.h
        include/file_storage.h
        include/json_object.h
        include/key_value_storage.h
        include/kvs_type.h
        include/per_base_type.h
        include/per_error_domain.h
        include/per_logger.h
        include/read_accessor.h
        include/read_write_accessor.h
        include/serializer_type.h
        include/shared_handle.h
        include/unique_handle.h
        DESTINATION ${TARGET_PLATFORM}/include/per)
install(DIRECTORY ../core
        DESTINATION ${TARGET_PLATFORM}/include/per)
install(DIRECTORY struct2x/include/struct2x
        DESTINATION ${TARGET_PLATFORM}/include/per)
        
