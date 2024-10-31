cmake_minimum_required(VERSION 3.16.3)

# project(ReaderTesting VERSION 1 LANGUAGES CXX)
set(MOUDLE_NAME check_public_receive_time)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
message(STATUS "Configuring ComplexIdlTesting example...")

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)

add_executable(${MOUDLE_NAME}
check_public_receive_time.cpp
  ${PROTO_SRC}
  )


target_include_directories(${MOUDLE_NAME}
  PRIVATE
  ./fast-dds/include
  ${NETA_MIDDLEWARE_DIR}/tools/data_tools/common
  ${NETA_MIDDLEWARE_DIR}/tools/data_tools/common/ament_index_cpp/include
  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/include/
  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/include/data_tools/bag/
  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/include/protobuf/include
  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/include/fast-dds/
  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/conf/bag
  ./)

target_link_libraries(
    ${MOUDLE_NAME}
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libneta_bag.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libtinyxml2.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libfastrtps.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libfastcdr.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libfoonathan_memory-0.7.1.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/librosbag2_storage.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/librosbag2_cpp.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/librcpputils.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib//libprotobuf.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libyaml-cpp.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libclass_loader.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libneta_cm.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libjsoncpp.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libzmq.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libneta_zmq_ipc.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libneta_log.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/librcutils.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libament_index_cpp.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libzipper.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libuuid.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libglobalproto.so
    ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libpcap.so.1

    #orin
    netaos::codec

    )
install(TARGETS ${MOUDLE_NAME}
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)