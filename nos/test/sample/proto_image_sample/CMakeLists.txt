
add_subdirectory(proto_image)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conf
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/proto_image_sample  FOLLOW_SYMLINK_CHAIN)

file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libadf-lite-ds.so
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/proto_image_sample/lib  FOLLOW_SYMLINK_CHAIN)