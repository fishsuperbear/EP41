cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "testcase_DATACOLLECTION_COLLECTION_001")


set(PRJ_SOURCE_DIR  "${PRJ_SOURCE_DIR}")
if("${TARGET_PLATFORM}" STREQUAL "")
    set(TARGET_PLATFORM "x86_2004")
    #  单独编译时使用。
    set(PRJ_SOURCE_DIR  "/home/cheng/code/nos-commit")
endif()
set(NETA_THIRDPARTY_DIR "${PRJ_SOURCE_DIR}/thirdparty")
set(NETA_MIDDLEWARE_DIR "${PRJ_SOURCE_DIR}/middleware")
set(CMAKE_INSTALL_RPATH "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(DATA_TOOLS_ROOT_PATH "${NETA_MIDDLEWARE_DIR}/tools/data_tools")
set(DATA_TOOLS_COMMON_PATH "${DATA_TOOLS_ROOT_PATH}/common")
set(BAG_ROOT_PATH "${DATA_TOOLS_ROOT_PATH}/bag")
set(DATA_COLLECTION_ROOT_PATH "${PRJ_SOURCE_DIR}/service/data_collection")
set(DATA_COLLECTION_COMMON_PATH "${DATA_COLLECTION_ROOT_PATH}/common")
#set(CURRENT_PROJECT_LIB_PATH "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
message(STATUS "====target platform :==== ${TARGET_PLATFORM}")
message(STATUS "====DATA_TOOLS_COMMON_PATH : ${DATA_TOOLS_COMMON_PATH}")
message(STATUS "====rcpputils  : ${DATA_TOOLS_COMMON_PATH}/rcpputils/include/rcpputils/filesystem_helper.hpp")


#Check C++11
include(CheckCXXCompilerFlag)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if (NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif ()
endif ()
#set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -dl ")
#set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ltbb  -ldl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g --std=c++17 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-reorder -Wno-sign-compare -Wno-strict-aliasing -Wno-deprecated-declarations ")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")

link_directories(
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/Poco/lib
)


# ---------------------------------------------------------------------------------
file(GLOB_RECURSE
        DC_COLLECTION_SRC
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/collection_factory.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/collection_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/bag_record.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/mcu_bag_record.cpp

        ${DATA_COLLECTION_ROOT_PATH}/common/thread_pool/src/thread_pool_flex.cpp

        ${DATA_COLLECTION_ROOT_PATH}/destination/src/advc_upload.cpp
        ${DATA_COLLECTION_ROOT_PATH}/destination/src/destination_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/manager/src/manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/pipeline/src/pipeline_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/processor_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/compressor.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/mcap_changer.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/add_data.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/mcap_h265_rw.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/all_file_merge.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/path_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/time_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/trans_utils.cpp

        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_info.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoTypeObject.cxx
)

file (GLOB_RECURSE ADVCSDK "${DATA_COLLECTION_ROOT_PATH}/advc-sdk/*.cpp")

set(DC_COLLECTION_INC_DIRS
        ${DATA_COLLECTION_COMMON_PATH}/
        ${DATA_COLLECTION_ROOT_PATH}/
        ${DATA_TOOLS_COMMON_PATH}/ros2/rcpputils/include/
        ${DATA_TOOLS_COMMON_PATH}/ros2/rcutils/include/
        ${DATA_TOOLS_COMMON_PATH}/ros2/rosbag2/rosbag2_cpp/include/
        ${DATA_TOOLS_COMMON_PATH}/ros2/rosbag2/rosbag2_storage/include/
        ${DATA_TOOLS_ROOT_PATH}/bag/include
        ${DATA_TOOLS_ROOT_PATH}/bag/src
        ${NETA_MIDDLEWARE_DIR}/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/common
        ${NETA_MIDDLEWARE_DIR}/crypto/include/common/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/cryp/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/cryp/cryobj/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/keys
        ${NETA_MIDDLEWARE_DIR}/crypto/include/utility/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/utility/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/x509/
        ${NETA_MIDDLEWARE_DIR}/https/
        ${NETA_MIDDLEWARE_DIR}/https/impl/include
        ${NETA_MIDDLEWARE_DIR}/https/include/
        ${NETA_MIDDLEWARE_DIR}/https/include/
        ${NETA_MIDDLEWARE_DIR}/desen/include/
        ${NETA_MIDDLEWARE_DIR}/cfg/include/
        ${NETA_MIDDLEWARE_DIR}/tools/
        ${PRJ_SOURCE_DIR}/service/data_collection/advc-sdk/include/
        ${PRJ_SOURCE_DIR}/service/data_collection/advc-sdk/include/encrypt
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/Poco/include
        ${PRJ_SOURCE_DIR}
        ${PRJ_SOURCE_DIR}/demo/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
)


message(STATUS "========gtest path=========="  ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include)
set(ALL_RELEATED_LIB
        gtest
        gtest_main
        ################### 压缩依赖 ###################
        z
        rcpputils
        ################### advc sdk 依赖 ###################
        PocoCrypto
        PocoFoundation
        PocoJSON
        PocoNet
        PocoNetSSL
        PocoUtil
        PocoXML

        # ################### data collection 依赖 ###################
        #        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/libjsoncpp.so # repeat so found
        yaml-cpp
        #        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/lib/libzipper.so # repeat so found

        crypto
        dl
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        jsoncpp
        #        neta_bag
        neta_cm
        neta_em_client
        neta_compress
        neta_phm
        #        neta_crypto # not used so found
        neta_https
        neta_log
        neta_cfg
        neta_zmq_ipc
        neta_mcap
        protobuf
        pthread
        rt
        ssl
        stdc++
        tinyxml2
        uuid
        zipper
        zmq
        neta_lrzip
        lzo2
        lz4

        neta_sm_client
        neta_crypto
        neta_mcap


        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libneta_bag.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libtinyxml2.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libcurl.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libfastrtps.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libfastcdr.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libfoonathan_memory-0.7.1.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/librosbag2_storage.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/librosbag2_cpp.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/librcpputils.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib//libprotobuf.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libyaml-cpp.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libclass_loader.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libneta_cm.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libjsoncpp.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libzmq.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libneta_zmq_ipc.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libneta_log.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/librcutils.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libament_index_cpp.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libzipper.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libuuid.so
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libpcap.so.1
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libpcl_io.so.1.11.1
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libpcl_io_ply.so.1.11.1
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/libglobalproto.so
        #orin
        netaos::codec

)

if(${TARGET_PLATFORM} STREQUAL "orin")
    set(ALL_RELEATED_LIB
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.70.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.70.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so.1.11.1
            ${ALL_RELEATED_LIB}
            neta_desen
    )
else (${TARGET_PLATFORM} STREQUAL "x86_2004")
    set(ALL_RELEATED_LIB
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.73.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.73.0
            ${ALL_RELEATED_LIB}
    )
endif()


################################################### dc_config begin###########################
set(data_collection dc_config_server)
add_executable(${data_collection}
        dc_config_server.cpp
)
set_target_properties(${data_collection} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${data_collection}
        PUBLIC
        ${NETA_MIDDLEWARE_DIR}
        ${NETA_MIDDLEWARE_DIR}/https/
        ${NETA_MIDDLEWARE_DIR}/https/impl/include
        ${NETA_MIDDLEWARE_DIR}/https/include/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/utility/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/common/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/cryp/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/x509/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/keys
        ${NETA_MIDDLEWARE_DIR}/crypto/include/cryp/cryobj/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/utility/
        ${NETA_MIDDLEWARE_DIR}/https/impl/include
        ${NETA_MIDDLEWARE_DIR}/lrzip-0.631
        ${NETA_MIDDLEWARE_DIR}/cfg/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
)

target_link_libraries(${data_collection}
        pthread
        zmq
        neta_zmq_ipc
        protobuf
        neta_lrzip
        lzo2
        lz4
        dl
        jsoncpp
        crypto
        neta_crypto
        ssl
        zipper
        curl
        neta_https
        netaos::log
        neta_cm
        zipper
        foonathan_memory-0.7.1
        uuid
        fastcdr
        fastrtps
        globalproto
        neta_cfg
        tinyxml2
)
install(TARGETS ${data_collection} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/")
################################################### dc_config end###########################

################################################### dc_config client begin###########################
set(data_collection dc_config_client)
add_executable(${data_collection}
        dc_config_client.cpp
)
set_target_properties(${data_collection} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${data_collection}
        PUBLIC
        ${NETA_MIDDLEWARE_DIR}
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/
)

target_link_libraries(${data_collection}
        pthread
        zmq
        neta_zmq_ipc
        protobuf
)
install(TARGETS ${data_collection} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/")
################################################### dc_config client end###########################
