
project(perception_12)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(perception_12 SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/perception1/perception1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/perception2/perception2.cpp)

target_link_libraries(perception_12 netaos::adf-lite)

install(TARGETS perception_12 LIBRARY DESTINATION ${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib)


add_subdirectory(fisheye_perception)
add_subdirectory(parking_fusion)
add_subdirectory(fisheye_mock)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conf
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample  FOLLOW_SYMLINK_CHAIN)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/etc
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample  FOLLOW_SYMLINK_CHAIN)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/env.sh
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample  FOLLOW_SYMLINK_CHAIN)
file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/bin/adf-lite-process
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/bin  FOLLOW_SYMLINK_CHAIN)
file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libadf-lite-ds.so
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib  FOLLOW_SYMLINK_CHAIN)
file(COPY ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/lib/libadf-lite-dbg.so
    DESTINATION  ${NETA_MIDDLEWARE_DIR}/../output/${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib  FOLLOW_SYMLINK_CHAIN)