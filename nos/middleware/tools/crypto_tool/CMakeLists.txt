#module名称
set(MODULE_NAME "crypto-tool")

set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

if (${TARGET_PLATFORM} MATCHES "orin")
    set(CMAKE_INSTALL_RPATH "/app/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(NV_TOP_DIR "/usr/local/nv-driveos")
    set(NV_PLATFORM_DIR "${NV_TOP_DIR}/drive-linux")
elseif()


endif()

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    ${NV_PLATFORM_DIR}/lib-target
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
message("nv include: ${NV_PLATFORM_DIR}/include")
message("nv include: ${NV_PLATFORM_DIR}/include/nvpkcs11")
#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/middleware
        ${NV_PLATFORM_DIR}/include
        ${NV_PLATFORM_DIR}/include/nvpkcs11
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_log
        tinyxml2
        dl
        rt
        jsoncpp
        zipper
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/libprotobuf.so
        nvpkcs11
        teec
        nvvse
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed,--disable-new-dtags")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)
