#module名称
set(MODULE_NAME "testcase_PKI_001")
add_definitions(-DUT=)

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${PRJ_ROOT_DIR}/service/pki_service/src/*.cpp
        ${PRJ_ROOT_DIR}/middleware/idl/generated/crypto.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/cryptoPubSubTypes.cxx
        ${PRJ_ROOT_DIR}/middleware/idl/generated/cryptoTypeObject.cxx
    )

# 用户需要添加自己的库路径
link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/cm/include
        ${PRJ_ROOT_DIR}/middleware/crypto/include
        ${PRJ_ROOT_DIR}/middleware/crypto/include/common
        ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp
        ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp/cryobj
        ${PRJ_ROOT_DIR}/middleware/crypto/include/keys
        ${PRJ_ROOT_DIR}/middleware/crypto/include/x509
        ${PRJ_ROOT_DIR}/middleware/crypto/include/utility
        ${PRJ_ROOT_DIR}/middleware/https/include
        ${PRJ_ROOT_DIR}/middleware/https/impl/include
        ${PRJ_ROOT_DIR}/middleware/idl/generated
        ${PRJ_ROOT_DIR}/middleware/devm/include
        ${PRJ_ROOT_DIR}/middleware/devm/include/common
        ${PRJ_ROOT_DIR}/service/pki_service/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${PRJ_ROOT_DIR}/middleware/lrzip-0.631
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
        ${PRJ_ROOT_DIR}/middleware/cfg/include/
)

if (${TARGET_PLATFORM} MATCHES "x86_2004" OR ${TARGET_PLATFORM} MATCHES "orin")
    set(CAMERA_LIBRARY
        neta_lrzip
        lzo2
        lz4)
endif()
#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        yaml-cpp
        crypto
        ssl
        curl
        jsoncpp
        pthread
        dl
        rt
        neta_https
        neta_crypto
        netaos::log
        tinyxml2
        neta_zmq_ipc
        zipper
        protobuf
        zmq
        neta_cm
        fastcdr
        fastrtps
        globalproto
        foonathan_memory-0.7.1
        uuid
        ${CAMERA_LIBRARY}
        neta_em_client
        neta_phm
        neta_sm_client
        gtest
        gtest_main
        neta_devm_client
        neta_cfg
)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -pie")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/test/unit_test/pki_test_unit/testcase_PKI_001/bin)
