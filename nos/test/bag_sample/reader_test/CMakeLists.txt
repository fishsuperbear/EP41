cmake_minimum_required(VERSION 3.16.3)

# project(ReaderTesting VERSION 1 LANGUAGES CXX)
set(MOUDLE_NAME ReaderTesting)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
message(STATUS "Configuring ComplexIdlTesting example...")

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)

add_executable(${MOUDLE_NAME}
  reader_test.cpp
  ${PROTO_SRC}
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
  )
target_include_directories(${MOUDLE_NAME}
        PRIVATE
        ${NETA_MIDDLEWARE_DIR}/tools/data_tools/bag/include
        ${NETA_MIDDLEWARE_DIR}/../
        ${NETA_MIDDLEWARE_DIR}/
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include/
        )

target_link_libraries(
    ${MOUDLE_NAME}
    netaos::bag
    class_loader
    rcutils
    ament_index_cpp
    )
install(TARGETS ${MOUDLE_NAME}
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)