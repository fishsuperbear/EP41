
set(MODULE_NAME "someip_tc8_test")
set(SOMEIP_TC8_TEST someip_tc8_test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${PRJ_ROOT_DIR}/middleware
    ${PRJ_ROOT_DIR}/middleware/someip/include
)

file(GLOB_RECURSE SOMEIP_TC8_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/NESomeIPTc8Process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NESomeIPTc8Server.cpp
)

add_executable(${MODULE_NAME} ${SOMEIP_TC8_FILES})


#用户可在此添加链接库目录
link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)

target_link_libraries(${MODULE_NAME}
    pthread
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_someip.so
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
)

install(TARGETS ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)