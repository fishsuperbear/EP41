
add_subdirectory(img_zerocopy)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conf
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/zerocopy_img_sample)

file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libadf-lite-ds.so
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/zerocopy_img_sample/lib  FOLLOW_SYMLINK_CHAIN)