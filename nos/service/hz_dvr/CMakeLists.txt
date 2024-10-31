# module名称
set(MODULE_NAME "hz_dvr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=gnu++14 -O3 -Wall -Werror")

# 把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattachPubSubTypes.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattachTypeObject.cxx
)

# 默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

# 用户可在此添加头文件路径
include_directories(
    ${NETA_MIDDLEWARE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
    ${CMAKE_SOURCE_DIR}/middleware/cfg/include/
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
)
target_link_directories(${MODULE_NAME}
    PRIVATE
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
)

# 用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
    PRIVATE
    neta_log
    yaml-cpp
    nvs_consumer
    neta_cm
    neta_em_client
)

# 用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS
    ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/${MODULE_NAME}.yaml
    DESTINATION ${TARGET_PLATFORM}/conf/
)

install(
    TARGETS 
    ${MODULE_NAME}
    RUNTIME 
    DESTINATION ${TARGET_PLATFORM}/test/${MODULE_NAME}/bin/
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/${MODULE_NAME}.yaml
    DESTINATION ${TARGET_PLATFORM}/test/${MODULE_NAME}/conf/
)
