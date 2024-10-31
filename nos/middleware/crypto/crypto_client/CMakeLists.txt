#module名称
set(MODULE_NAME "neta_crypto")

if(${BUILD_TYPE} STREQUAL "release")
    message("neta_crypto  BUILD_TYPE: ${BUILD_TYPE}")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -DNDEBUG -fPIC -Wsign-compare -fmessage-length=0 -fstack-protector-all  -Werror -Wno-error=unused-variable -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
elseif(${BUILD_TYPE} STREQUAL "debug")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -fPIC -Wsign-compare -fmessage-length=0 -fstack-protector-all -Werror -Wno-error=unused-variable -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
endif()

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/crypto*.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/client/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cryp/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cryp/cryobj/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/keys/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/x509/*.cpp
    )

#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/x509
        ${CMAKE_SOURCE_DIR}/middleware/crypto/common
        ${CMAKE_SOURCE_DIR}/middleware/crypto/crypto_client/src
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${NETA_MIDDLEWARE_DIR}
        )

target_link_directories(${MODULE_NAME}
        PRIVATE
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        neta_cm
        neta_em_client
        neta_log
        yaml-cpp
        )


install(
    TARGETS ${MODULE_NAME} LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/crypto/include
        DESTINATION ${TARGET_PLATFORM}/include/crypto/)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib/
        DESTINATION ${TARGET_PLATFORM}/lib)
