cmake_minimum_required(VERSION 2.8.12)

include_directories(
    ${NETA_SERVICE_DIR}
    ${NETA_MIDDLEWARE_DIR}
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
)

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    # /usr/local/mdc_sdk_0930/dp_gea/mdc_cross_compiler/sysroot/usr/lib
)

add_executable(
    can_tsync_center 
    can_tsync_center_main.cpp
    can_tsync.cpp
    sig_stop.cpp
    can_tsync_config_parser.cpp
)

target_link_libraries(
    can_tsync_center
    neta_log
    pthread
    jsoncpp
    zipper
    # Crc
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/lib/libyaml-cpp.so
)

install(
    TARGETS can_tsync_center
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/can_tsync_config.yaml
    DESTINATION ${TARGET_PLATFORM}/conf
)
