set(GTEST_LIB_PATH "${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib")
file(COPY ${GTEST_LIB_PATH}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/  FOLLOW_SYMLINK_CHAIN
)


set(COMMON_SCRIPTS ${CMAKE_SOURCE_DIR}/unit_test/sm_test_unit/sm_test_common)
file(COPY ${COMMON_SCRIPTS}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/sm_test_unit/  FOLLOW_SYMLINK_CHAIN)

set(COMMON_EMSCRIPTS ${CMAKE_SOURCE_DIR}/unit_test/em_test_unit/em_test_common)
file(COPY ${COMMON_EMSCRIPTS}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/em_test_unit/  FOLLOW_SYMLINK_CHAIN)

add_subdirectory(sm_test_unit)
add_subdirectory(example)
add_subdirectory(em_test_unit)
add_subdirectory(adf_test_unit)
add_subdirectory(cfg_test_unit)
add_subdirectory(per_test_unit)
add_subdirectory(system_monitor_test_uint)
add_subdirectory(data_collection_st)
# add_subdirectory(doip_test_unit)
# add_subdirectory(um_test_unit)
add_subdirectory(crc_test_unit)
add_subdirectory(e2e_test_unit)
add_subdirectory(network_capture_test_unit)
# add_subdirectory(log_test_unit)
add_subdirectory(devm_test_unit)
add_subdirectory(codec_test_unit)
add_subdirectory(pki_test_unit)
add_subdirectory(nvs_producer_test_unit)

if(${TARGET_PLATFORM} MATCHES "orin")
add_subdirectory(camera_venc_test_unit)
add_subdirectory(desen_test_unit)
endif()