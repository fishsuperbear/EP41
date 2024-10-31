cmake_minimum_required(VERSION 3.16.3)
set(MODULE_NAME "testcase_DATACOLLECTION_COLLECTION_001")


set(PRJ_SOURCE_DIR  "${PRJ_ROOT_DIR}")
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
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/mcu_bag_recorder.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/mcu_bag_collector.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/mcu_log_collector.cpp
        ${DATA_COLLECTION_ROOT_PATH}/collection/src/impl/can_bag_collector.cpp

        ${DATA_COLLECTION_ROOT_PATH}/common/thread_pool/src/thread_pool_flex.cpp

        ${DATA_COLLECTION_ROOT_PATH}/destination/src/advc_upload.cpp
        ${DATA_COLLECTION_ROOT_PATH}/destination/src/destination_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/manager/src/manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/manager/src/cfg_calibstatus.cpp
        ${DATA_COLLECTION_ROOT_PATH}/manager/src/config_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/pipeline/src/pipeline_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/processor_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/compressor.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/mcap_changer.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/add_data.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/mcap_h265_rw.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/all_file_merge.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/get_dynamic_config.cpp
        ${DATA_COLLECTION_ROOT_PATH}/processor/src/impl/desense_manager.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/path_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/time_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/trans_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/sign_utils.cpp
        ${DATA_COLLECTION_ROOT_PATH}/utils/src/custom_time_util.cpp

        ${DATA_COLLECTION_ROOT_PATH}/../pki_service/src/tsp_pki_cert_manage.cpp
        ${DATA_COLLECTION_ROOT_PATH}/../pki_service/src/tsp_pki_config.cpp
        ${DATA_COLLECTION_ROOT_PATH}/../pki_service/src/tsp_pki_utils.cpp


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
        ${NETA_MIDDLEWARE_DIR}/devm/include/common
        ${PRJ_SOURCE_DIR}/service/pki_service/include/
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
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include
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
        neta_devm_client

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
        # #orin
        # netaos::codec
)

if(${TARGET_PLATFORM} STREQUAL "orin")
    set(ALL_RELEATED_LIB
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.70.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.70.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/pcl/lib/libpcl_common.so.1.11.1
            ${ALL_RELEATED_LIB}
            netaos::desen
            netaos::codec
    )
else (${TARGET_PLATFORM} STREQUAL "x86_2004")
    set(ALL_RELEATED_LIB
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_filesystem.so.1.73.0
            ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib/libboost_iostreams.so.1.73.0
            ${ALL_RELEATED_LIB}
    )
endif()

################################################### bag record  begin###########################
set(bag_rec_module testcase_DATACOLLECTION_COLLECTION_BAG_RECORDER_001)
add_executable(${bag_rec_module}
        bag_record_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${bag_rec_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${bag_rec_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${bag_rec_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${bag_rec_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${bag_rec_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${bag_rec_module}/bin")

################################################### bag record end###########################

################################################### advc sdk upload begin###########################
set(advc_upload_module testcase_DATACOLLECTION_COLLECTION_ADVC_UPLOAD_001)
add_executable(${advc_upload_module}
        advc_upload_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${advc_upload_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${advc_upload_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${advc_upload_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${advc_upload_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${advc_upload_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${advc_upload_module}/bin")
################################################### advc sdk upload end###########################


################################################### rebase remote config begin###########################
set(remote_config_module testcase_DATACOLLECTION_COLLECTION_REMOTE_CONFIG_001)
add_executable(${remote_config_module}
        remote_config_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${remote_config_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${remote_config_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${remote_config_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${remote_config_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${remote_config_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${remote_config_module}/bin")
################################################### advc sdk upload end###########################


################################################### rebase remote config begin###########################
set(add_data_module testcase_DATACOLLECTION_COLLECTION_ADD_DATA_001)
add_executable(${add_data_module}
        add_data_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${add_data_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${add_data_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${add_data_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${add_data_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${add_data_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${add_data_module}/bin")
################################################### advc data end###########################


################################################### collect all log  begin###########################
set(collect_all_log_module testcase_DATACOLLECTION_COLLECTION_COL_ALL_LOG_001)
add_executable(${collect_all_log_module}
        all_log_collector_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${collect_all_log_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${collect_all_log_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${collect_all_log_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${collect_all_log_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${collect_all_log_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${collect_all_log_module}/bin")
################################################### collect all log end###########################
################################################### add data end###########################

################################################### mcap changer begin###########################
set(mcap_changer_module testcase_DATACOLLECTION_COLLECTION_MCAP_CHANGER_001)
add_executable(${mcap_changer_module}
        mcap_changer_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${mcap_changer_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${mcap_changer_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${mcap_changer_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${mcap_changer_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${mcap_changer_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${mcap_changer_module}/bin")
################################################### mcap changer end###########################

################################################### all file merge begin###########################
set(all_file_merge_module testcase_DATACOLLECTION_COLLECTION_ALL_FILE_MERGE_001)
add_executable(${all_file_merge_module}
        all_file_merge_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${all_file_merge_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${all_file_merge_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${all_file_merge_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${all_file_merge_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${all_file_merge_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${all_file_merge_module}/bin")
################################################### all file merge end###########################

################################################### mcap h265 rw begin###########################
set(mcap_h265_rw_module testcase_DATACOLLECTION_COLLECTION_MCAP_H265_RW_001)
add_executable(${mcap_h265_rw_module}
        mcap_h265_rw_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${mcap_h265_rw_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${mcap_h265_rw_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${mcap_h265_rw_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${mcap_h265_rw_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${mcap_h265_rw_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${mcap_h265_rw_module}/bin")
################################################### mcap h265 rw end###########################

################################################### compressor begin###########################
set(compressor_module testcase_DATACOLLECTION_COLLECTION_COMPRESSOR_001)
add_executable(${compressor_module}
        compressor_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${compressor_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${compressor_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${compressor_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${compressor_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${compressor_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${compressor_module}/bin")
################################################### compressor end###########################

################################################### timer begin###########################
set(timer_test_module testcase_DATACOLLECTION_COLLECTION_TIMER_TEST_001)
add_executable(${timer_test_module}
        get_time_and_timer_test.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${timer_test_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${timer_test_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${timer_test_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${timer_test_module}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${timer_test_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${timer_test_module}/bin")
################################################### timer end###########################

################################################### data_collection begin###########################
set(data_collection data_collection_test)
add_executable(${data_collection}
        ${DATA_COLLECTION_ROOT_PATH}/server/dc_server.cpp
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${data_collection} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${data_collection}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
)
target_link_directories(${data_collection}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${data_collection}
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${data_collection} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/")
################################################### timer end###########################

################################################### mcu begin ###########################
set(mcu_module testcase_DATACOLLECTION_COLLECTION_MCU_001)
file(GLOB_RECURSE MCU_SRC
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_MCUmaintainService_Client_SOC/*cpp
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_TriggerIDService_Server_SOC/*cpp
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_McuCANMsgService_Client_SOC/*cpp
        ${DATA_COLLECTION_ROOT_PATH}/mcu/src/MCUClient.cpp)
add_executable(${mcu_module}
        mcu_test.cpp
        ${MCU_SRC}
        ${DC_COLLECTION_SRC}
        ${ADVCSDK}
)
set_target_properties(${mcu_module} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
target_include_directories(${mcu_module}
        PUBLIC
        ${DC_COLLECTION_INC_DIRS}
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_MCUmaintainService_Client_SOC/communication_management/includes
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_TriggerIDService_Server_SOC/communication_management/includes
        ${DATA_COLLECTION_ROOT_PATH}/mcu/gen/AASWC_McuCANMsgService_Client_SOC/communication_management/includes
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
)
target_link_directories(${mcu_module}
        PUBLIC
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/st_someip/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
        ${PRJ_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)
target_link_libraries(${mcu_module}
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
        ${ALL_RELEATED_LIB}
)
install(TARGETS ${mcu_module} RUNTIME DESTINATION "${PRJ_SOURCE_DIR}/output/${TARGET_PLATFORM}/test/unit_test/data_collection_st/${mcu_module}/bin")
################################################### mcu end ###########################
