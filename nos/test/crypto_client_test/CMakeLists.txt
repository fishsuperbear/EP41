#module名称
set(MODULE_NAME "crypto_client_test")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --std=c++17  -Wno-error=unused-variable")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib

        )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        /usr/local/include
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/crypto/include
        ${PRJ_ROOT_DIR}/middleware/crypto/include/common
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include
        # ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp
        # ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp/cryobj
        # ${PRJ_ROOT_DIR}/middleware/crypto/include/keys
        # ${PRJ_ROOT_DIR}/middleware/crypto/include/x509
        # ${PRJ_ROOT_DIR}/middleware/crypto/crypto_server/impl/include/common
        # ${PRJ_ROOT_DIR}/middleware/crypto/crypto_server/impl/include/keys
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        netaos::log
        neta_crypto
        jsoncpp
        neta_em_client
        yaml-cpp
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        # crypto
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
        uuid
        tinyxml2
        globalproto
        netaos::cm
        )

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")
# 
#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
