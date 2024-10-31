#module名称
set(MODULE_NAME "hz_chassis")

link_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib 
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_SOURCE_DIR}/service/canstack/common/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx)


#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

add_definitions(-DCHASSIS_DEBUG_ON)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${NETA_MIDDLEWARE_DIR}/
        ${CMAKE_SOURCE_DIR}/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${CMAKE_SOURCE_DIR}/service/canstack/hz_chassis/include/
        ${CMAKE_SOURCE_DIR}/service/canstack/common/include/
        ${CMAKE_SOURCE_DIR}/service/canstack/hz_cangen/include/
        ${CMAKE_SOURCE_DIR}/service/hz_common/include/
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        hz_cangen
        # hz_common
        neta_cm
        neta_log
        yaml-cpp
        neta_e2e
        neta_e2exf
        neta_e2exf_cpp
)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -pie")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION  ${TARGET_PLATFORM}/bin
)

#for test
install(
    TARGETS
    ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/hz_chassis/bin
)

install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../config/hz_chassis.yaml
    DESTINATION ${TARGET_PLATFORM}/test/hz_chassis/conf
)
