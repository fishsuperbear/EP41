#module名称
set(MODULE_NAME "camera_venc")

# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)
set(DEFAULT_COMPILE -std=c++14)

if (${TARGET_PLATFORM} MATCHES "orin")
    set(CAMV_PLAT_DEP_INC_DIRS 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/orin
        ${NETA_MIDDLEWARE_DIR}/sensor/multicast
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
        )
    # set(CAMV_PLAT_DEP_LIBS  nvs_consumer nvs neta_camera cudart nvsipl nvmedia_iep_sci nvscistream nvmedia2d nvscibuf nvscisync nvscievent nvsciipc nvscicommon cuda tegrawfd nvsipl_query nvvic nvdc nvidia-kms nvimp nvddk_2d_v2 pthread)
    set(CAMV_PLAT_DEP_LIBS  nvs_consumer nvs)
    set(CAMV_PLAT_DEP_LIB_DIRS  
        ${COMPILER_SYSROOT}/lib
        ${COMPILER_PATH}/../usr/lib
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
        ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    )

    #把目录下的c和c++所有源文件分别保存在变量中
    file(GLOB
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/orin/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        )
else (${TARGET_PLATFORM} MATCHES "x86_2004")
    set(CODEC_LIBRARY avcodec avutil swresample swscale)
    file(GLOB
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )
endif()

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/fast-dds/include
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/include
        ${NETA_MIDDLEWARE_DIR}
        ${NETA_MIDDLEWARE_DIR}/per/include
        ${NETA_MIDDLEWARE_DIR}/per/include/yaml_cpp_struct/
        ${NETA_MIDDLEWARE_DIR}/per/include/magic_enum/
        ${COMPILER_SYSROOT}/include
        ${CAMV_PLAT_DEP_INC_DIRS}
        )

target_link_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/../usr/lib
        ${NETA_MIDDLEWARE_DIR}/sensor/multicast/lib
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/yaml-cpp/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CAMV_PLAT_DEP_LIB_DIRS}
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        neta_cm
        neta_em_client
        neta_codec
        neta_log
        yaml-cpp
        ${CAMV_PLAT_DEP_LIBS}
        )

# set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

if (${TARGET_PLATFORM} MATCHES "orin")
    file(GLOB CONF_FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/orin/*.yaml)
    # file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf/camera_venc)
endif()
