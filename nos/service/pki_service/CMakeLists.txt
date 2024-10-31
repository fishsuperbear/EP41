#module名称
set(MODULE_NAME "pki_service")

#用户可在此添加编译选项
if(${BUILD_TYPE} STREQUAL "release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -fPIE -Wno-error=unused-variable  -Wno-unused-variable")
elseif(${BUILD_TYPE} STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -fPIE -Wno-error=unused-variable -Wno-unused-variable")
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/crypto.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/cryptoPubSubTypes.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/cryptoTypeObject.cxx
    )

# 用户需要添加自己的库路径
link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
)


#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/common
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp/cryobj
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/keys
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/x509
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/utility
        ${CMAKE_SOURCE_DIR}/middleware/https/include
        ${CMAKE_SOURCE_DIR}/middleware/https/impl/include
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated
        ${CMAKE_SOURCE_DIR}/middleware/cm/include
        ${CMAKE_SOURCE_DIR}/middleware/devm/include
        ${CMAKE_SOURCE_DIR}/middleware/devm/include/common
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/middleware/cfg/include/
)

if (${TARGET_PLATFORM} MATCHES "x86_2004" OR ${TARGET_PLATFORM} MATCHES "orin")
    set(CAMERA_LIBRARY
        neta_lrzip
        #lzo2
        #lz4
        )
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
        neta_log
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
        neta_devm_client
        neta_cfg
)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -pie")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)
# install(FILES ${MODULE_NAME} DESTINATION /app/runtime_service/pki_service/bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/pki_service/conf/
        DESTINATION ${TARGET_PLATFORM}/conf/pki_service/)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/pki_service/etc/
        DESTINATION ${TARGET_PLATFORM}/conf/pki_service/)


# install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/pki_service/preset
#         DESTINATION ${TARGET_PLATFORM}/conf/)
