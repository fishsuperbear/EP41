#module名称
set(MODULE_NAME "crypto_server")
 
if(${TARGET_PLATFORM} STREQUAL "mdc" OR ${TARGET_PLATFORM} STREQUAL "mdc-llvm")
    set(MDC_SDK "/usr/local/mdc_sdk_llvm")
endif()


if(${BUILD_TYPE} STREQUAL "release")
    message("neta_crypto  BUILD_TYPE: ${BUILD_TYPE}")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -DNDEBUG -fPIC -Wsign-compare -fmessage-length=0 -fstack-protector-all  -Wno-error=unused-variable -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
elseif(${BUILD_TYPE} STREQUAL "debug")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -fPIC -Wsign-compare -fmessage-length=0 -fstack-protector-all  -Wno-error=unused-variable -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
endif()
# set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

message("neta_crypto  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/*/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/*/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../keys/*.hpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/crypto*.cxx
        )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${MDC_SDK}/sysroot/usr/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_server
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include
        ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_server/impl/include
        ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_server/src
        ${CMAKE_SOURCE_DIR}/middleware/crypto/common
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
)

target_link_directories(${MODULE_NAME}
    PUBLIC    
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        jsoncpp
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
        neta_log
        tinyxml2
        neta_cm
        neta_em_client
        neta_log
        yaml-cpp
        globalproto
        neta_em_client
        neta_phm
)

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/crypto/include
        DESTINATION ${TARGET_PLATFORM}/include/crypto/)
        
set(OPENSSL3.0.2_LIB
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
)

install(FILES ${OPENSSL3.0.2_LIB} DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_server/etc/
        DESTINATION ${TARGET_PLATFORM}/conf/crypto_server/)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_server/conf/
        DESTINATION ${TARGET_PLATFORM}/conf/crypto_server/)
