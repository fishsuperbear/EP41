#module名称
set(MODULE_NAME "https_test")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --std=c++17  -Wno-error=unused-variable")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

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
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib64
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
        )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/https/include/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/utility/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/common/
        ${PRJ_ROOT_DIR}/middleware/https/
        ${PRJ_ROOT_DIR}/middleware/https/include/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/common
        ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/x509/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/keys
        ${PRJ_ROOT_DIR}/middleware/crypto/include/cryp/cryobj/
        ${PRJ_ROOT_DIR}/middleware/crypto/include/utility/
        ${PRJ_ROOT_DIR}/middleware/https/impl/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${PRJ_ROOT_DIR}/middleware/lrzip-0.631
        ${PRJ_ROOT_DIR}/middleware/cfg/include
        )

if (${TARGET_PLATFORM} MATCHES "x86_2004" OR ${TARGET_PLATFORM} MATCHES "orin")
    set(CAMERA_LIBRARY
        neta_lrzip
        lzo2
        lz4)
endif()
#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        dl
        jsoncpp
        crypto
        neta_crypto
        ssl
        zipper
        curl
        ${CAMERA_LIBRARY}
        neta_https
        netaos::log
        neta_zmq_ipc
        neta_cm
        zipper
        protobuf
        zmq
        foonathan_memory-0.7.1
        uuid
        fastcdr
        fastrtps
        globalproto
        neta_cfg
        tinyxml2
        )

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
