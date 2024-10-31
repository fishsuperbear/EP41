cmake_minimum_required(VERSION 3.16.3)

# project(ReaderTesting VERSION 1 LANGUAGES CXX)
set(MOUDLE_NAME sample_reader)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
message(STATUS "Configuring ComplexIdlTesting example...")

file(GLOB PROTO_SRC ${PRJ_ROOT_DIR}/proto/*/*.pb.cc)

add_executable(${MOUDLE_NAME}
sample_reader.cpp
  ${PROTO_SRC}
  )

#编译完成后，跑程序前先执行：source  工具包目录/scripts/scripts/env_setup.sh
#${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/替换成工具包目录


link_directories(pointcloud
  home/sw/work/netaos/output/${TARGET_PLATFORM}/lib/)

target_include_directories(${MOUDLE_NAME}
  PRIVATE
  ./fast-dds/include
  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/data_tools/bag/
  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/protobuf/include
  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/fast-dds/
  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/conf/bag
  ./)

target_link_libraries( ${MOUDLE_NAME}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_bag.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libtinyxml2.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libfastrtps.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libfastcdr.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libfoonathan_memory-0.7.1.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/librosbag2_storage.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/librosbag2_cpp.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/librcpputils.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib//libprotobuf.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libyaml-cpp.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libclass_loader.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_cm.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libjsoncpp.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libzmq.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_zmq_ipc.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_log.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/librcutils.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libament_index_cpp.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libzipper.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libuuid.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libpcap.so.1
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libpcl_common.so
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libboost_filesystem.so
    # ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libboost_iostreams.so.1.70.0
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libpcl_io.so.1.11.1
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libpcl_io_ply.so.1.11.1
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libglobalproto.so
    #orin
    netaos::codec
    )

  if (${TARGET_PLATFORM} MATCHES "x86_2004")
    # target_link_libraries( ${MOUDLE_NAME}  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libboost_iostreams.so.1.70.0)
  elseif (${TARGET_PLATFORM} MATCHES "orin")
    target_link_libraries( ${MOUDLE_NAME}  ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libboost_iostreams.so.1.70.0)
  endif( )


install(TARGETS ${MOUDLE_NAME}
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)