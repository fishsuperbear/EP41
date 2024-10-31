set(DATA_TOOLS_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DATA_TOOLS_COMMON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(common)
add_subdirectory(topic)
add_subdirectory(bag)
add_subdirectory(mcap)

if(${TARGET_PLATFORM} STREQUAL "x86_2004")
    add_subdirectory(convert/rtfbag_to_mcap)
endif()

# -----------------------------------install add_dependencies include------------------------------
install(
  DIRECTORY ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/include/protobuf)
  install(
  DIRECTORY ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/include/fast-dds)
# -----------------------------------install add_dependencies so------------------------------
install(
  DIRECTORY ${NETA_MIDDLEWARE_DIR}/idl/data_type/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag/share/data_type)
install(
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/zstd/lib/libzstd.so
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)
# install(
#   FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/mcap/lib/libmcap.so
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)
install(
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)
install(
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/lib/libprotobuf.so
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)
install(
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so.1.10.4
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so.1
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/libpcap-1.10.4/lib/libpcap.so
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)

install(
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io.so
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io.so.1.11
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io.so.1.11.1
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io_ply.so
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io_ply.so.1.11
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_io_ply.so.1.11.1
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so.1.11
  FILES ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so.1.11.1
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)

  file(GLOB_RECURSE filesystem_file
   "${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so*"
   "${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so*")

install(
  FILES ${filesystem_file}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/lib)

#install proto
 # 设置源目录和目标目录
set(PROTO_SOURCE_DIR "${NETA_MIDDLEWARE_DIR}/../proto")
set(PROTO_DESTINATION_DIR "${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag/proto")
file(GLOB_RECURSE PROTO_FILES
  "${PROTO_SOURCE_DIR}/*.proto")
 # 遍历找到的文件，并保留目录结构进行拷贝
foreach(PROTO_FILE ${PROTO_FILES})
    # 获取文件相对于源目录的路径
    file(RELATIVE_PATH PROTO_RELATIVE_PATH "${PROTO_SOURCE_DIR}" "${PROTO_FILE}")
    # 获取目标路径
    set(PROTO_DESTINATION_PATH "${PROTO_DESTINATION_DIR}/${PROTO_RELATIVE_PATH}")
    # 创建目标目录（如果不存在）
    # file(MAKE_DIRECTORY "${PROTO_DESTINATION_DIR}")
    # 拷贝文件
    configure_file("${PROTO_FILE}" "${PROTO_DESTINATION_PATH}" COPYONLY)
endforeach()

file(COPY bag/conf/topic_classification_config.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag)
file(COPY bag/conf/bag_attachments_config.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag)
