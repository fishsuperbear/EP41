set(GTEST_LIB_PATH "${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib")
file(COPY ${GTEST_LIB_PATH}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/  FOLLOW_SYMLINK_CHAIN
)

add_subdirectory(diag_server_ut)
add_subdirectory(doip_ut)
add_subdirectory(system_monitor_ut)