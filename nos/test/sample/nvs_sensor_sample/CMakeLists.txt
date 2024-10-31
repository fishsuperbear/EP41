
add_subdirectory(nvs_sensor_test)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conf
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/nvs_sensor_sample  FOLLOW_SYMLINK_CHAIN)

file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libadf-lite-ds.so
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/nvs_sensor_sample/lib  FOLLOW_SYMLINK_CHAIN)