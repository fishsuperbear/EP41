cmake_minimum_required(VERSION 3.16.3)
project(data_collection)
set(MODULE_NAME "data_collection")
set(PRJ_SOURCE_DIR  "${CMAKE_SOURCE_DIR}")

message(STATUS "====CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
message(STATUS "====TARGET_PLATFORM : ${TARGET_PLATFORM}")

if ("${TARGET_PLATFORM}" STREQUAL "")
    set(TARGET_PLATFORM "x86_2004")
endif ()

if ("${TARGET_PLATFORM}" MATCHES "orin") 
    add_definitions(-DBUILD_FOR_ORIN=1) 
endif()

if (${PRJ_SOURCE_DIR} MATCHES ".*/data_collection")
    set(PRJ_SOURCE_DIR "${PRJ_SOURCE_DIR}/../..")
    set(netaos_DIR ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/cmake/netaos/)
    set(CMAKE_PREFIX_PATH ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/cmake/netaos/)
    message(STATUS "====CMAKE_PREFIX_PATH : ${CMAKE_PREFIX_PATH}")
    message(STATUS "====netaos_DIR : ${netaos_DIR}")
    find_package(netaos CONFIG REQUIRED)
endif ()

set(NETA_THIRDPARTY_DIR "${PRJ_SOURCE_DIR}/thirdparty")
set(NETA_MIDDLEWARE_DIR "${PRJ_SOURCE_DIR}/middleware")
set(DATA_TOOLS_ROOT_PATH "${NETA_MIDDLEWARE_DIR}/tools/data_tools")
set(DATA_TOOLS_COMMON_PATH "${DATA_TOOLS_ROOT_PATH}/common")
message(STATUS "====DATA_TOOLS_COMMON_PATH : ${DATA_TOOLS_COMMON_PATH}")
message(STATUS "====rcpputils  : ${DATA_TOOLS_COMMON_PATH}/rcpputils/include/rcpputils/filesystem_helper.hpp")
set(BAG_ROOT_PATH "${DATA_TOOLS_ROOT_PATH}/bag")

set(DATA_COLLECTION_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(DATA_COLLECTION_COMMON_PATH "${DATA_COLLECTION_ROOT_PATH}/common")
set(CURRENT_PROJECT_LIB_PATH "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")


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
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")

link_directories(
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/Poco/lib
)

# -----------------------------------neta_compress.so------------------------------
add_library(neta_compress
    SHARED
    common/compress/src/dc_compress.cpp
    common/compress/src/archive/archive_write.c
    common/compress/src/archive/archive_write_disk_posix.c
    common/compress/src/archive/archive_write_add_filter_gzip.c
    common/compress/src/archive/archive_write_disk_set_standard_lookup.c
    common/compress/src/archive/archive_check_magic.c
    common/compress/src/archive/archive_util.c
    common/compress/src/archive/archive_virtual.c
    common/compress/src/archive/archive_string.c
    common/compress/src/archive/archive_entry.c
    common/compress/src/archive/archive_string_sprintf.c
    common/compress/src/archive/archive_acl.c
    common/compress/src/archive/archive_entry_xattr.c
    common/compress/src/archive/archive_entry_sparse.c
    common/compress/src/archive/archive_write_open_filename.c
    common/compress/src/archive/archive_write_open_fd.c
    common/compress/src/archive/archive_write_set_format_ustar.c
    common/compress/src/archive/archive_write_set_format.c
    common/compress/src/archive/archive_write_set_format_7zip.c
    common/compress/src/archive/archive_write_set_format_cpio.c
    common/compress/src/archive/archive_write_set_format_cpio_binary.c
    common/compress/src/archive/archive_write_set_format_cpio_odc.c
    common/compress/src/archive/archive_write_set_format_cpio_newc.c
    common/compress/src/archive/archive_write_set_format_iso9660.c
    common/compress/src/archive/archive_write_set_format_mtree.c
    common/compress/src/archive/archive_write_set_format_raw.c
    common/compress/src/archive/archive_write_set_format_shar.c
    common/compress/src/archive/archive_write_set_format_pax.c
    common/compress/src/archive/archive_write_set_format_gnutar.c
    common/compress/src/archive/archive_write_set_format_warc.c
    common/compress/src/archive/archive_write_set_format_xar.c
    common/compress/src/archive/archive_write_set_format_zip.c
    common/compress/src/archive/archive_rb.c
    common/compress/src/archive/archive_ppmd7.c
    common/compress/src/archive/archive_digest.c
    common/compress/src/archive/archive_random.c
    common/compress/src/archive/archive_cryptor.c
    common/compress/src/archive/archive_hmac.c
    common/compress/src/archive/archive_write_set_passphrase.c
    common/compress/src/archive/archive_read.c
    common/compress/src/archive/archive_read_support_format_all.c
    common/compress/src/archive/archive_read_support_format_ar.c
    common/compress/src/archive/archive_read_support_format_cpio.c
    common/compress/src/archive/archive_read_support_format_empty.c
    common/compress/src/archive/archive_read_support_format_lha.c
    common/compress/src/archive/archive_read_support_format_mtree.c
    common/compress/src/archive/archive_read_support_format_tar.c
    common/compress/src/archive/archive_read_support_format_warc.c
    common/compress/src/archive/archive_read_support_format_7zip.c
    common/compress/src/archive/archive_read_support_format_cab.c
    common/compress/src/archive/archive_read_support_format_rar.c
    common/compress/src/archive/archive_read_support_format_rar5.c
    common/compress/src/archive/archive_read_support_format_iso9660.c
    common/compress/src/archive/archive_read_support_format_zip.c
    common/compress/src/archive/archive_ppmd8.c
    common/compress/src/archive/archive_read_support_filter_all.c
    common/compress/src/archive/archive_read_support_filter_bzip2.c
    common/compress/src/archive/archive_read_support_filter_compress.c
    common/compress/src/archive/archive_read_support_filter_gzip.c
    common/compress/src/archive/archive_read_support_filter_xz.c
    common/compress/src/archive/archive_read_support_filter_uu.c
    common/compress/src/archive/archive_read_support_filter_rpm.c
    common/compress/src/archive/archive_read_support_filter_lrzip.c
    common/compress/src/archive/archive_read_support_filter_lzop.c
    common/compress/src/archive/archive_read_support_filter_grzip.c
    common/compress/src/archive/archive_read_support_filter_lz4.c
    common/compress/src/archive/archive_read_support_filter_zstd.c
    common/compress/src/archive/archive_read_open_filename.c
    common/compress/src/archive/archive_entry_link_resolver.c
    common/compress/src/archive/archive_write_add_filter_lz4.c
    common/compress/src/archive/archive_write_add_filter_program.c
    common/compress/src/archive/filter_fork_posix.c
    common/compress/src/archive/archive_cmdline.c
    common/compress/src/archive/archive_read_support_filter_program.c
    common/compress/src/archive/archive_read_support_format_raw.c
    common/compress/src/archive/archive_read_add_passphrase.c
    common/compress/src/archive/archive_blake2s_ref.c
    common/compress/src/archive/archive_blake2sp_ref.c
    common/compress/src/archive/archive_pack_dev.c
    common/compress/src/zipper/CDirEntry.cpp
    common/compress/src/zipper/tools.cpp
    common/compress/src/zipper/unzipper.cpp
    common/compress/src/zipper/zipper.cpp
    common/compress/src/minizip/ioapi_buf.c
    common/compress/src/minizip/ioapi_mem.c
    common/compress/src/minizip/ioapi.c
    common/compress/src/minizip/unzip.c
    common/compress/src/minizip/zip.c
    )

target_include_directories(neta_compress
PRIVATE
    common/compress/include/
    common/compress/src/
    common/compress/src/archive/
    common/compress/src/zipper/
    common/compress/src/minizip/
    ${PRJ_SOURCE_DIR}/service/data_collection/
    ${PRJ_SOURCE_DIR}/middleware/
    ${PRJ_SOURCE_DIR}/middleware/tools/data_tools/common/ros2/rcpputils/include/
    ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include/
    )

target_link_libraries(neta_compress PUBLIC crypto)

install(TARGETS neta_compress
    ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
    LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

# ---------------------------------------------------------------------------------

file (GLOB_RECURSE ADVCSDK "advc-sdk/*.cpp")
add_executable(data_collection
        collection/src/collection_factory.cpp
        collection/src/collection_manager.cpp
        collection/src/impl/bag_record.cpp
        collection/src/impl/mcu_bag_recorder.cpp
        collection/src/impl/mcu_bag_collector.cpp
        collection/src/impl/mcu_log_collector.cpp
        collection/src/impl/can_bag_collector.cpp

        common/thread_pool/src/thread_pool_flex.cpp
        server/dc_server.cpp
        destination/src/advc_upload.cpp
        ${ADVCSDK}
        destination/src/destination_manager.cpp
        manager/src/config_manager.cpp
        manager/src/manager.cpp
        manager/src/cfg_calibstatus.cpp
        pipeline/src/pipeline_manager.cpp
        processor/src/processor_manager.cpp
        processor/src/impl/compressor.cpp
        processor/src/impl/mcap_changer.cpp
        processor/src/impl/mcap_h265_rw.cpp
        processor/src/impl/desense_manager.cpp
        processor/src/impl/add_data.cpp
        processor/src/impl/all_file_merge.cpp
        processor/src/impl/get_dynamic_config.cpp
        utils/src/path_utils.cpp
        utils/src/time_utils.cpp
        utils/src/trans_utils.cpp
        utils/src/sign_utils.cpp
        utils/src/custom_time_util.cpp

        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_info.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoTypeObject.cxx
)
target_compile_options(data_collection PRIVATE -Wall)
set_target_properties(data_collection PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(data_collection
        PUBLIC
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
        ${NETA_MIDDLEWARE_DIR}/tools/
        ${NETA_MIDDLEWARE_DIR}/desen/include/
        ${NETA_MIDDLEWARE_DIR}/cfg/include/
        ${PRJ_SOURCE_DIR}/service/data_collection/advc-sdk/include/
        ${PRJ_SOURCE_DIR}/service/data_collection/advc-sdk/include/encrypt
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/Poco/include
        ${PRJ_SOURCE_DIR}
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
        )
target_link_directories(data_collection
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(data_collection
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
        neta_bag
        neta_cm
        neta_em_client
        neta_compress
        neta_phm
        #        neta_crypto # not used so found
        neta_https
        neta_log
        neta_zmq_ipc
        neta_mcap
        neta_crypto
        neta_cfg
        protobuf
        pthread
        rt
        ssl
        stdc++
        tinyxml2
        uuid
        zipper
        zmq
        globalproto
        )

if(${TARGET_PLATFORM} STREQUAL "orin")
        target_link_libraries(data_collection neta_desen)
endif()

add_library(neta_dc
        SHARED
        client/src/dc_client_trigger.cpp
        client/src/dc_client.cpp
        client/src/dc_client_impl.cpp
        utils/src/path_utils.cpp
        utils/src/time_utils.cpp
        utils/src/trans_utils.cpp

        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_info.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/data_collection_infoTypeObject.cxx
)
target_include_directories(neta_dc
        PUBLIC
        ${PRJ_SOURCE_DIR}/
        ${DATA_COLLECTION_COMMON_PATH}/
        ${DATA_COLLECTION_ROOT_PATH}/
        ${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${NETA_MIDDLEWARE_DIR}/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)
target_link_libraries(neta_dc
        dl
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        jsoncpp
        neta_cm
        neta_log
        pthread
        rt
        tinyxml2
        uuid
)

add_executable(dc_trigger
        client/src/dc_client_trigger.cpp
        utils/src/time_utils.cpp
)
target_include_directories(dc_trigger
        PUBLIC
        ${PRJ_SOURCE_DIR}/
        ${DATA_COLLECTION_COMMON_PATH}/
        ${DATA_COLLECTION_ROOT_PATH}/
)
target_link_libraries(dc_trigger
        neta_dc
)

add_executable(dc_mcu
        mcu/gen/AASWC_McuCANMsgService_Client_SOC/communication_management/src/hozon/netaos/v1/mcucanmsgservice_manifest.cpp
        mcu/gen/AASWC_McuCANMsgService_Client_SOC/communication_management/src/hozon/netaos/v1/mcucanmsgservice_proxy.cpp
        mcu/gen/AASWC_MCUmaintainService_Client_SOC/communication_management/src/hozon/netaos/v1/mcumaintainservice_manifest.cpp
        mcu/gen/AASWC_MCUmaintainService_Client_SOC/communication_management/src/hozon/netaos/v1/mcumaintainservice_proxy.cpp
        mcu/gen/AASWC_TriggerIDService_Server_SOC/communication_management/src/hozon/netaos/v1/triggeridservice_manifest.cpp
        mcu/gen/AASWC_TriggerIDService_Server_SOC/communication_management/src/hozon/netaos/v1/triggeridservice_skeleton.cpp
        mcu/src/MCUClient.cpp
        mcu/src/main.cpp
)
target_include_directories(dc_mcu PRIVATE
        ${DATA_COLLECTION_ROOT_PATH}
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_McuCANMsgService_Client_SOC/communication_management/includes
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_MCUmaintainService_Client_SOC/communication_management/includes
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_TriggerIDService_Server_SOC/communication_management/includes
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
)
target_link_directories(dc_mcu PRIVATE
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/st_someip/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
)
target_link_libraries(dc_mcu PRIVATE
        neta_dc
        neta_em_client
        ara_com_ics
        someip_e2e_ics
        nlog_ics
        aracore_ics
        dlt_ics
        perftoolkit_ics
        ara_serialize_ics
        ncdump_ics
        someip_ics
        ara_someipbinding_ics
        ncjson_ics
        someip_serializer_ics
        ncore_ics
        ssl
        crypto
        pthread
)

add_executable(remote_config
        remote_config/src/main.cpp
        remote_config/src/remote_config.cpp
        remote_config/src/dc_download.cpp
        utils/src/path_utils.cpp
        common/thread_pool/src/thread_pool_flex.cpp
        utils/src/time_utils.cpp
        utils/src/sign_utils.cpp
)

target_include_directories(remote_config PRIVATE
        ${DATA_COLLECTION_ROOT_PATH}
        ${NETA_MIDDLEWARE_DIR}
        ${DATA_COLLECTION_ROOT_PATH}/remote_config/include/
        ${DATA_COLLECTION_ROOT_PATH}/common/
        ${NETA_MIDDLEWARE_DIR}/https/include/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/
        ${NETA_MIDDLEWARE_DIR}/crypto/include/common/
        ${NETA_MIDDLEWARE_DIR}/cfg/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include/
        ${NETA_MIDDLEWARE_DIR}/cfg/include/
        # ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
)
# target_link_directories(remote_config PRIVATE
#         ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
#         ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/st_someip/lib
#         ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
# )
target_link_libraries(remote_config PRIVATE
        neta_dc
        neta_em_client
        pthread
        neta_https
        neta_cfg
)

install(TARGETS neta_dc
        ARCHIVE DESTINATION ${TARGET_PLATFORM}/lib
        LIBRARY DESTINATION ${TARGET_PLATFORM}/lib
        RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(TARGETS data_collection RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(TARGETS dc_trigger RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(TARGETS dc_mcu RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(TARGETS remote_config RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
install(FILES ${DATA_COLLECTION_ROOT_PATH}/client/include/dc_client.h
        DESTINATION ${TARGET_PLATFORM}/include/dc/include)
file(GLOB POCO_LIB_FILES  "${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/Poco/lib/*so*")
install(FILES ${POCO_LIB_FILES} DESTINATION ${TARGET_PLATFORM}/lib)
