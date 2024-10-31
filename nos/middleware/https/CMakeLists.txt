#module名称
set(MODULE_NAME "neta_https")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
#lib名称
set(LIB_NAME neta_https)
link_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/https
        ${CMAKE_SOURCE_DIR}/middleware/https/include
        ${CMAKE_SOURCE_DIR}/middleware/https/src/impl/include
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/common
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/x509
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/keys
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp/cryobj
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/utility
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated
        ${CMAKE_SOURCE_DIR}/middleware/cm/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/middleware/lrzip-0.631
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
        pthread
        jsoncpp
        crypto
        ssl
        neta_log
        neta_crypto
        tinyxml2
        curl
        neta_cm
        fastcdr
        fastrtps
        globalproto
        foonathan_memory-0.7.1
        ${CAMERA_LIBRARY}
        uuid
        neta_cfg
)

if (${TARGET_PLATFORM} MATCHES "x86_2004")
    add_definitions(-DBUILD_FOR_X86=1)
elseif (${TARGET_PLATFORM} MATCHES "mdc")
    add_definitions(-DBUILD_FOR_MDC=1)
elseif (${TARGET_PLATFORM} MATCHES "mdc-llvm")
    add_definitions(-DBUILD_FOR_MDC=1)
elseif (${TARGET_PLATFORM} MATCHES "j5")
    add_definitions(-DBUILD_FOR_J5=1)
elseif (${TARGET_PLATFORM} MATCHES "orin")
    add_definitions(-DBUILD_FOR_ORIN=1)
endif()

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${TARGET_PLATFORM}/include/https/include)
if(${TARGET_PLATFORM} STREQUAL "mdc-llvm")
    
else()
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib/
    DESTINATION ${TARGET_PLATFORM}/lib)
endif()



